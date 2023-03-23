import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import trunc_normal_
from torchvision.ops import RoIAlign

from .head import build_box_head
from .language_model import build_bert
from .utils import _init_weights_xavier, _init_weights_trunc_normal
from .visual_model.swin_transformer import build_swin_transformer_backbone
from .visual_model.transformer import build_decoder, build_transformer
from lib.models.visual_model.vl_transformer import build_vl_transformer
from ..utils.box_ops import box_xyxy_to_cxcywh, box_cxcywh_to_xyxy
from ..utils.misc import NestedTensor


class JointNLT(nn.Module):
    def __init__(self, cfg, visual_backbone, language_backbone, vl_joint_trans, temporal_decoder, feature_decoder,
                 box_head, hidden_dim):
        """ Initializes the model.
        """
        super(JointNLT, self).__init__()
        self.hidden_dim = hidden_dim
        self.visual_backbone = visual_backbone
        self.language_backbone = language_backbone
        self.vl_joint_trans = vl_joint_trans
        self.temporal_decoder = temporal_decoder
        self.target_decoder = feature_decoder
        self.box_head = box_head
        self.head_type = cfg.MODEL.HEAD_TYPE

        self.visu_proj = nn.Linear(visual_backbone.num_channels_output[-1], hidden_dim)
        self.text_proj = nn.Linear(self.language_backbone.num_channels, hidden_dim)

        self.divisor = 16
        self.template_size = cfg.TEST.TEMPLATE_SIZE
        self.search_size = cfg.TEST.SEARCH_SIZE
        self.feat_sz_s = self.search_size // self.divisor
        self.num_visu_template_token = int((cfg.TEST.TEMPLATE_SIZE // self.divisor) ** 2)
        self.num_visu_search_token = int((cfg.TEST.SEARCH_SIZE // self.divisor) ** 2)
        self.num_text_token = cfg.MODEL.LANGUAGE.BERT.MAX_QUERY_LEN
        self.USE_VIS_CLS = cfg.MODEL.VL.USE_VIS_CLS
        self.USE_VIS_SEP = cfg.MODEL.VL.USE_VIS_SEP
        self.roi = RoIAlign((6, 6), spatial_scale=1 / self.divisor, sampling_ratio=2)

        # if we use vit as visual encoder, there will be an extract CLS after feature extraction
        if self.USE_VIS_CLS:
            self.num_visu_template_token += 1
            self.num_visu_search_token += 1

        # add 1 for CLS TOKEN
        self.num_total = self.num_visu_template_token + self.num_visu_search_token + self.num_text_token

        if self.USE_VIS_SEP:
            self.num_total += 1
            self.sep_embed = nn.Embedding(1, hidden_dim)

        self.temporal_query_embed = nn.Embedding(1, hidden_dim)
        self.target_query_embed = nn.Embedding(1, hidden_dim)
        self.vl_pos_embed = nn.Embedding(self.num_total, hidden_dim)

        self.local_rank = torch.cuda.current_device()
        self.init = cfg.MODEL.VL.INIT
        self._reset_parameters(self.init)

    def _reset_parameters(self, init):
        # parameters init
        if init == 'xavier':
            self._init_weights_func = _init_weights_xavier
        elif init == 'trunc_norm':
            self._init_weights_func = _init_weights_trunc_normal
        else:
            raise RuntimeError(F"init method should be xavier/trunc_norm, not {init}.")
        self.vl_pos_embed.apply(self._init_weights_func)
        self.text_proj.apply(self._init_weights_func)
        self.visu_proj.apply(self._init_weights_func)

    def forward(self, text_data: NestedTensor, template: NestedTensor, temporal: torch.tensor,
                search_patch: NestedTensor, mode: str):
        '''
            params:
                text_data:  contains token_ids and masks which both shape are (B, LANGUAGE_LENGTH)
                template: contains template_src: (B, C, H, W) template masks: (B, H, W)
                search: contains search_src: (B, C, H, W) search masks: (B, H, W)
                mode: str, choice in ['grounding', 'track']
            return
        '''
        text_src, text_mask = self.forward_text(text_data)
        bs = text_src.shape[0]
        if mode == 'grounding':
            # zero padding as placeholder
            # src: b×c×h×w  mask: b×h×w
            template_src = torch.zeros((bs, self.num_visu_template_token, self.hidden_dim),
                                       requires_grad=True).cuda()
            template_mask = torch.ones((bs, self.num_visu_template_token), requires_grad=True).cuda().to(torch.bool)
        elif mode == 'tracking':
            template = template.to(self.local_rank)
            template_src, template_mask = self.forward_vision_backbone(template)
        else:
            raise ValueError

        # search_src: (B, C, H, W) to (B, H/divisor, HIDDIM_DIM)
        # search_mask: (B, H, W) to (B, HW)
        search_src, search_mask = self.forward_vision_backbone(search_patch)

        return self.forward_joint(text_src, text_mask, template_src, template_mask, search_src, search_mask, temporal)

    def forward_text(self, text_data: NestedTensor):
        # language bert
        text_fea = self.language_backbone(text_data)
        text_src, text_mask = text_fea.decompose()  # seq_len * b * HIDDEN_DIM , seq_len * b
        text_src = self.text_proj(text_src)
        return text_src, text_mask

    def forward_vision_backbone(self, images: NestedTensor):
        img, mask = images.decompose()
        img_sz = img.shape[-1]
        img_src = self.visual_backbone(img)
        img_src = self.visu_proj(img_src[-1].flatten(2).permute(0, 2, 1).contiguous())
        mask = F.interpolate(mask[None].float(), size=img_sz // self.divisor,
                             mode='nearest').to(torch.bool)[0].flatten(1)
        return img_src, mask

    def forward_joint(self, text_src, text_mask, template_src, template_mask, search_src, search_mask, temporal):
        srcs = [text_src, template_src]
        masks = [text_mask, template_mask]
        bs = text_src.shape[0]

        if self.USE_VIS_SEP:
            sep_mask = torch.zeros((bs, 1), requires_grad=True).cuda().to(torch.bool)
            sep_src = self.sep_embed.weight.unsqueeze(1).repeat(bs, 1, 1)
            srcs.append(sep_src)
            masks.append(sep_mask)

        srcs.append(search_src)
        masks.append(search_mask)
        vl_src = torch.cat(srcs, dim=1).permute(1, 0, 2).contiguous()
        vl_mask = torch.cat(masks, dim=1).contiguous()
        vl_pos = self.vl_pos_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        # vl_src: (L, B, C)
        # vl_mask: (B, L)
        # vl_pos: (L, B, C)
        output = self.vl_joint_trans(vl_src, vl_mask, vl_pos)

        search_tokens = output[-self.num_visu_search_token:, :]  # (HW, B, C)
        text_cls = output[0, :].unsqueeze(-1)  # (B, C, L), L = 1
        pos_embed = vl_pos[-self.num_visu_search_token:, :]

        temporal_output = 0
        if temporal is not None:
            temporal_query = self.temporal_query_embed.weight.unsqueeze(1).repeat(1, bs, 1)  # (1, C) -> (1, B, C)
            mix_temporal = torch.cat((text_cls, temporal), dim=-1).permute(2, 0,
                                                                           1).contiguous()  # (B, C,L) -> (L, B, C)
            temporal_output, _ = self.temporal_decoder(src=mix_temporal, mask=None, pos_embed=None,
                                                       query_embed=temporal_query)

        target_query = self.target_query_embed.weight.unsqueeze(1).repeat(1, bs, 1)  # (L, B, C), L = 1
        target_query = target_query + temporal_output
        target_output = self.target_decoder(tgt=target_query, memory=search_tokens, memory_key_padding_mask=search_mask,
                                            pos=pos_embed, query_pos=None)

        search_tokens = search_tokens + search_tokens * target_output

        pred_boxes = self.forward_box_head(search_tokens)

        roi_feature = self.get_target_feature(search_tokens, pred_boxes)

        out = {'pred_boxes': pred_boxes, 'roi_feature': roi_feature}
        return out

    def forward_box_head(self, search_tokens):
        if self.head_type == "CORNER":
            # run the corner head
            opt = (search_tokens.unsqueeze(-1)).permute((1, 3, 2, 0)).contiguous()  # (HW, B, C, 1) >> (B, 1, C, HW)
            b, Nq, C, HW = opt.size()
            opt_feat = opt.view(-1, C, self.feat_sz_s, self.feat_sz_s)  # (B, C, H', W')
            pred_box = box_xyxy_to_cxcywh(self.box_head(opt_feat))
        else:
            raise KeyError

        return pred_box

    def forward_test(self, text_data, template, search_patch: NestedTensor, temporal):
        """
            only for testing
            text_data: the text feature after language backbone
            template: the vision feature after vision backbone
            search_patch: the search image after processing
            temporal: roi feature
        """
        text_src, text_mask = text_data[:]
        template_src, template_mask = template[:]
        search_src, search_mask = self.forward_vision_backbone(search_patch)
        return self.forward_joint(text_src, text_mask, template_src, template_mask, search_src, search_mask, temporal)

    def get_target_feature(self, search_tokens, pred_boxes):
        """
            used for extracting roi feature
        """
        opt = (search_tokens.unsqueeze(-1)).permute((1, 3, 2, 0)).contiguous()  # (HW, B, C, 1) >> (B, 1, C, HW)
        b, Nq, C, HW = opt.size()
        opt_feat = opt.view(-1, C, self.feat_sz_s, self.feat_sz_s)  # (B, C, H', W')
        crop_box = pred_boxes
        crop_box[:, -2:] = crop_box[:, -2:] * 1.5
        target_boxes = box_cxcywh_to_xyxy(crop_box * self.search_size)
        no = torch.tensor(range(b)).unsqueeze(-1).cuda()
        boxes = torch.cat((no, target_boxes), dim=1)
        target_roi_feature = self.roi(opt_feat, boxes)
        return target_roi_feature


def build_jointnlt(cfg):
    if "swin" in cfg.MODEL.VISUAL.BACKBONE:
        visual_backbone = build_swin_transformer_backbone(cfg.MODEL.VISUAL.BACKBONE,
                                                          output_layers=(0, 1, 2))
        for parameter in visual_backbone.parameters():
            parameter.requires_grad_(True)
    else:
        raise NotImplementedError("VISUAL BACKBONE method not implemented")

    language_backbone = build_bert(cfg)

    vl_joint_trans = build_vl_transformer(cfg)

    temporal_decoder = build_transformer(cfg)

    target_decoder = build_decoder(cfg)

    box_head = build_box_head(cfg)
    hidden_dim = cfg.MODEL.VL.HIDDEN_DIM
    model = JointNLT(cfg, visual_backbone, language_backbone, vl_joint_trans, temporal_decoder, target_decoder,
                     box_head, hidden_dim)
    return model
