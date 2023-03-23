import os
import traceback

import cv2
# import ipdb
import ipdb
import numpy as np
import torch
from pytorch_pretrained_bert import BertTokenizer
from torch import Tensor
from torch.nn import functional as F

from lib.models.JointNLT import build_jointnlt
from lib.test.tracker.basetracker import BaseTracker
from lib.test.tracker.tracker_utils import Preprocessor_wo_mask
from lib.train.data.processing_utils import sample_target, grounding_resize, grounding_resize_without_box
from lib.utils.box_ops import clip_box, box_xywh_to_xyxy, return_iou_boxes, box_cxcywh_to_xyxy, box_xyxy_to_xywh
from lib.utils.misc import NestedTensor


class JointNLT(BaseTracker):
    def __init__(self, params, dataset_name):
        super(JointNLT, self).__init__(params)
        network = build_jointnlt(params.cfg)
        network.load_state_dict(torch.load(self.params.checkpoint, map_location='cpu')['net'], strict=False)
        vocab_path = self.params.cfg.MODEL.LANGUAGE.VOCAB_PATH

        if vocab_path is not None and os.path.exists(vocab_path):
            self.tokenizer = BertTokenizer.from_pretrained(vocab_path, do_lower_case=True)
        else:
            self.tokenizer = BertTokenizer.from_pretrained(self.params.cfg.MODEL.LANGUAGE.TYPE, do_lower_case=True)
        self.cfg = params.cfg
        self.network = network.cuda()
        self.network.eval()
        self.preprocessor = Preprocessor_wo_mask()
        self.state = None
        # for debug
        self.debug = False
        self.frame_id = 0
        self.use_first_frame = True if self.params.first_frame == 1 else False
        self.test_method = self.cfg.TEST.TEST_METHOD
        self.template_sz = self.cfg.DATA.TEMPLATE.SIZE
        self.search_num = self.cfg.DATA.SEARCH.NUMBER

        self.MAX_HIS_FRAME = 2
        self.template_roi_feature = None
        self.roi_queue = []

        if self.debug:
            self.save_dir = "debug"
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)
        # for save boxes from all queries
        self.save_all_boxes = params.save_all_boxes

    def initialize(self, image, info: dict):
        # preprocess the groungding patch
        bbox = torch.tensor(info['init_bbox']).cuda()
        self.text_input = self._text_input_process(info['init_nlp'], self.params.cfg.MODEL.LANGUAGE.BERT.MAX_QUERY_LEN)
        with torch.no_grad():
            self.text_dict = self.network.forward_text(self.text_input)

        if self.test_method in ["JOINT", "GROUND"]:
            # (x,y w,h) normalized to [0,1]
            im_crop_padded, box, att_mask, mask_crop_padded, image_top_coords = \
                grounding_resize(image, self.params.grounding_size, bbox, None)
            grounding_img = self.preprocessor.process(im_crop_padded).cuda()
            att_mask = torch.tensor(att_mask).unsqueeze(0).cuda()
            grounding_patch = NestedTensor(grounding_img, att_mask)
            # change shape (H, W) to (B, H, W)
            image_top_coords = torch.tensor(image_top_coords).unsqueeze(0).cuda()
            # forward the template once
            template_src = torch.zeros((1, 64, 256), requires_grad=False).cuda()
            template_mask = torch.ones((1, 64), requires_grad=False).cuda().to(torch.bool)

            self.template_dict = [template_src, template_mask]
            with torch.no_grad():
                out_dict = self.network.forward_test(self.text_dict, self.template_dict, grounding_patch, None)
                scale_box = self._get_templates_and_anno(image, out_dict['pred_boxes'], image_top_coords,
                                                         self.params.grounding_size)
                # return shape: (b,4)
                scale_box = scale_box.squeeze()
            self.template_roi_feature = out_dict['roi_feature'].flatten(2)  # (B, C, H', W') -> (B, C, H'W')
            z_patch_arr, _, z_amask_arr = sample_target(image, scale_box, self.params.template_factor,
                                                        output_sz=self.params.template_size)
            self.template_img, _, _ = sample_target(image, scale_box, 1.0,
                                                    output_sz=144)
            template = self.preprocessor.process(z_patch_arr).cuda()
            self.state = scale_box.tolist()

        elif self.test_method == "TRACK":
            z_patch_arr, _, z_amask_arr = sample_target(image, info['init_bbox'], self.params.template_factor,
                                                        output_sz=self.params.template_size)
            template = self.preprocessor.process(z_patch_arr).cuda()
            self.state = info['init_bbox']
        z_amask_arr = torch.tensor(z_amask_arr).unsqueeze(0).cuda()
        self.template_input = NestedTensor(template, z_amask_arr)
        with torch.no_grad():
            self.template_dict = self.network.forward_vision_backbone(self.template_input)
        self.frame_id = 0
        if self.save_all_boxes:
            '''save all predicted boxes'''
            all_boxes_save = info['init_bbox'] * self.cfg.MODEL.NUM_OBJECT_QUERIES
            return {"all_boxes": all_boxes_save}

    def track(self, image, info: dict = None):
        H, W, _ = image.shape
        self.frame_id += 1

        if self.test_method == "GROUND":
            self.grounding_track(image)
        elif self.test_method in ("JOINT", "TRACK"):
            # crop the search patch according to self.state(last frame predict)
            x_patch_arr, resize_factor, x_amask_arr = sample_target(image, self.state, self.params.search_factor,
                                                                    output_sz=self.params.search_size)  # (x1, y1, w, h)

            search = self.preprocessor.process(x_patch_arr).cuda()
            x_amask_arr = torch.tensor(x_amask_arr).unsqueeze(0).cuda()
            search_patch = NestedTensor(search, x_amask_arr)
            temporal = None
            roi_length = len(self.roi_queue)

            if self.test_method == "JOINT":
                if roi_length >= self.MAX_HIS_FRAME:
                    self.roi_queue = self.roi_queue[-self.MAX_HIS_FRAME + 1:]
                roi_list = [self.template_roi_feature]
                roi_list.extend(self.roi_queue)

            with torch.no_grad():
                out_dict = self.network.forward_test(self.text_dict, self.template_dict, search_patch, temporal)

                self.roi_queue.append(out_dict['roi_feature'].flatten(2))  # (B, C, H', W') -> (B, C, H'W')
            pred_boxes = out_dict['pred_boxes'].view(-1, 4)

            pred_box = (pred_boxes.mean(
                dim=0) * self.params.search_size / resize_factor).tolist()  # (cx, cy, w, h) [0,1]
            # get the final box result
            self.state = clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=10)
        else:
            raise ValueError(f"without test_method{self.test_method}, please choice in  ['GROUND', 'TRACK', 'JOINT']")

        # for debug
        if self.debug:
            x1, y1, w, h = self.state
            image_BGR = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.rectangle(image_BGR, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color=(0, 0, 255), thickness=2)
            save_path = os.path.join(self.save_dir, "%04d.jpg" % self.frame_id)
            cv2.imwrite(save_path, image_BGR)
        if self.save_all_boxes:
            '''save all predictions'''
            all_boxes = self.map_box_back_batch(pred_boxes * self.params.search_size / resize_factor, resize_factor)
            all_boxes_save = all_boxes.view(-1).tolist()  # (4N, )
            return {"target_bbox": self.state,
                    "all_boxes": all_boxes_save}
        else:
            return {"target_bbox": self.state}

    def grounding_track(self, image):
        H, W, _ = image.shape
        self.frame_id += 1
        im_crop_padded, att_mask, mask_crop_padded, image_top_coords, resize_factor = \
            grounding_resize_without_box(image, self.params.grounding_size, None)
        # print(f'resize_factor:{resize_factor}')
        grounding_patch = self.preprocessor.process(im_crop_padded).cuda()
        att_mask = torch.tensor(att_mask).unsqueeze(0).cuda()
        grounding_patch = NestedTensor(grounding_patch, att_mask)
        # change shape (H, W) to (B, H, W)
        # forward the template once
        with torch.no_grad():
            out_dict = self.network.forward_test(self.text_dict, self.template_dict, grounding_patch, None)
        # return shape: (b,4)
        pred_boxes = out_dict['pred_boxes'].view(-1, 4)
        # Baseline: Take the mean of all pred boxes as the final result
        pred_box = (pred_boxes.mean(dim=0) * self.params.search_size / resize_factor)  # (cx, cy, w, h) [0,1]
        pred_box = box_xyxy_to_xywh(box_cxcywh_to_xyxy(pred_box)).tolist()
        # get the final box result
        self.state = clip_box(pred_box, H, W, margin=10)

    def map_box_back(self, pred_box: list, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]

    def map_box_back_batch(self, pred_box: torch.Tensor, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box.unbind(-1)  # (N,4) --> (N,)
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return torch.stack([cx_real - 0.5 * w, cy_real - 0.5 * h, w, h], dim=-1)

    def _extract_token_from_nlp(self, nlp, seq_length):
        """ use tokenizer to convert nlp to tokens
        param:
            nlp:  a sentence of natural language
            seq_length: the max token length, if token length larger than seq_len then cut it,
            elif less than, append '0' token at the reef.
        return:
            token_ids and token_marks
        """
        nlp_token = self.tokenizer.tokenize(nlp)
        if len(nlp_token) > seq_length - 2:
            nlp_token = nlp_token[0:(seq_length - 2)]
        # build tokens and token_ids
        tokens = []
        input_type_ids = []
        tokens.append("[CLS]")
        input_type_ids.append(0)
        for token in nlp_token:
            tokens.append(token)
            input_type_ids.append(0)
        tokens.append("[SEP]")
        input_type_ids.append(0)
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < seq_length:
            input_ids.append(0)
            input_mask.append(0)
            input_type_ids.append(0)
        assert len(input_ids) == seq_length
        assert len(input_mask) == seq_length
        assert len(input_type_ids) == seq_length

        return input_ids, input_mask

    def _get_templates_and_anno(self, image, grounding_dict: torch.Tensor, grounding_image_coords: torch.Tensor,
                                output_sz: int):
        '''
            params:
            grounding_dict - tensor, shape (b, 4) the output of grounding process
            grounding_image_coords - tensor, the coords of the resized grounding image in the input image  shape: (b,4)
            return:
              return the grounding box map to the original image
        '''
        image_shape = image.shape
        # get each original image shape the tensor shape is [b* 3 (h,w,c)]
        original_shapes = torch.tensor(image_shape).cuda().unsqueeze(0)
        # predict boxes
        pred_boxes = torch.round(grounding_dict * output_sz)
        # Compute the IOU boxes between the predict boxes and the resized image in the grounding image
        iou_boxes = return_iou_boxes(box_xywh_to_xyxy(grounding_image_coords), box_cxcywh_to_xyxy(pred_boxes))
        # Compute the iou boxes' relative position in the resize image
        # Compute x y relative postition
        iou_boxes[:, 0:2] = torch.sub(iou_boxes[:, 0:2], grounding_image_coords[:, 0:2]).clamp(min=0)
        scale_factor = torch.div(original_shapes[:, 1], grounding_image_coords[:, 2]).unsqueeze(-1)
        # the correct size of predict boxes in original image
        scale_boxes = iou_boxes * scale_factor
        return scale_boxes

    def _text_input_process(self, nlp, seq_length):
        text_ids, text_masks = self._extract_token_from_nlp(nlp, seq_length)
        text_ids = torch.tensor(text_ids).unsqueeze(0).cuda()
        text_masks = torch.tensor(text_masks).unsqueeze(0).cuda()
        return NestedTensor(text_ids, text_masks)


def get_tracker_class():
    return JointNLT
