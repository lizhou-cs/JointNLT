# import ipdb
import torch

from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy
from . import BaseActor
from ..data.processing import TemplateProcessing
from ...utils.misc import NestedTensor


class JointNLTActor(BaseActor):
    """ Actor for training the JointNLT
        As we propose the Semantic-Guided Temporal Modeling (SGTM) module requiring temporal training approach,
        we employ a multi-frame training approach.
        Specifically, we sample a grounding patch and two search patches.
        Initially, we perform grounding processing on the grounding patch to obtain a template.
        Subsequently, we use the template and natural language to track the target on the first search patch,
        thereby acquiring the historical appearance of the target (ROI feature).
        Finally, we utilize the ROI feature as a temporal clue for training the SGTM module.
    """

    def __init__(self, net, objective, loss_weight, settings, run_score_head=False):
        super().__init__(net, objective)
        self.loss_weight = loss_weight
        self.settings = settings
        self.bs = self.settings.batchsize  # batch size
        self.run_score_head = run_score_head
        self.processing = TemplateProcessing(settings)

    def __call__(self, data):
        """
        args:
            data - The input data, should contain the fields ‘nl_token_ids', 'token_mask','template',
            'search', 'gt_bbox'.
            nl_token_ids: (B, Length)
            template_images: (N_t, batch, 3, H, W)
            search_images: (N_s, batch, 3, H, W)
        returns:
            loss    - the training loss
            status  -  dict containing detailed losses
        """
        # forward pass
        # todo find a better way to fix this
        for key in data.keys():
            if isinstance(data.get(key), torch.Tensor):
                data[key] = data.get(key).squeeze()
        data['nl_token_ids'] = data['nl_token_ids'].permute(1, 0)
        data['nl_token_masks'] = data['nl_token_masks'].permute(1, 0)
        if self.settings.num_search == 1:
            loss, status = self.forward_joint(data)
        elif self.settings.num_search > 1:
            loss, status = self.forward_multiFrame(data)
        else:
            raise ValueError

        return loss, status

    def forward_joint(self, data):
        # Text Input
        text_data = NestedTensor(data['nl_token_ids'], data['nl_token_masks'])
        grounding_path = NestedTensor(data['grounding_images'], data['grounding_att'])
        # for grounding
        ground_dict = self.net(text_data, None, None, grounding_path, 'grounding')
        # Crop template according to ground_dict
        image_coords = data['grounding_coords'].clone().detach().squeeze()
        self.processing(data, data['grounding_frames_path'], ground_dict['pred_boxes'], image_coords)
        # for tracking
        template_patch = NestedTensor(data['template_images'], data['template_att'])
        search_patch = NestedTensor(data['search_images'], data['search_att'])
        track_dict = self.net(text_data, template_patch, None, search_patch, 'tracking')

        # compute losses
        g_loss, g_status, mask_sample = self.adaptive_compute_losses(ground_dict, data['grounding_anno'], "grounding",
                                                                     mask_sample=None)
        t_loss, t_status, _ = self.adaptive_compute_losses(track_dict, data['search_anno'], "tracking", mask_sample)

        loss = g_loss + t_loss
        status = {}
        g_keys = g_status.keys()
        for key in g_keys:
            status['g_' + key] = g_status.get(key)
        t_keys = t_status.keys()
        for key in t_keys:
            status['t_' + key] = t_status.get(key)
        for key in g_keys:
            status[key] = g_status.get(key) + t_status.get(key)

        return loss, status

    def forward_multiFrame(self, data):
        # Text Input
        text_data = NestedTensor(data['nl_token_ids'], data['nl_token_masks'])
        grounding_path = NestedTensor(data['grounding_images'], data['grounding_att'])
        # for grounding
        ground_dict = self.net(text_data, None, None, grounding_path, 'grounding')
        template_roi_feature = ground_dict['roi_feature'].flatten(2)  # (B, C, H', W') -> (B, C, H'W')
        # Crop template according to ground_dict
        image_coords = data['grounding_coords'].clone().detach().squeeze()
        jump_flag = False
        try:
            self.processing(data, data['grounding_frames_path'], ground_dict['pred_boxes'], image_coords)
        except ValueError:
            jump_flag = True
        if not jump_flag:
            # for tracking
            template_patch = NestedTensor(data['template_images'], data['template_att'])
            search_patch_1 = NestedTensor(data['search_images'][0], data['search_att'][0])
            track_dict_1 = self.net(text_data, template_patch, template_roi_feature, search_patch_1, 'tracking')

            last_frame_roi_feature = track_dict_1['roi_feature'].flatten(2)
            last_roi_feature = torch.cat((template_roi_feature, last_frame_roi_feature), dim=-1)
            search_patch_2 = NestedTensor(data['search_images'][1], data['search_att'][1])
            track_dict_2 = self.net(text_data, template_patch, last_roi_feature, search_patch_2, 'tracking')
            g_loss, g_status, mask_sample = self.adaptive_compute_losses(ground_dict, data['grounding_anno'],
                                                                         "grounding",
                                                                         mask_sample=None)
            t_loss_1, t_status_1, _ = self.adaptive_compute_losses(track_dict_1, data['search_anno'][0], "tracking",
                                                                   mask_sample)
            t_loss_2, t_status_2, _ = self.adaptive_compute_losses(track_dict_2, data['search_anno'][1], "tracking",
                                                                   mask_sample)

            loss = g_loss + t_loss_1 + t_loss_2
            status = {}
            g_keys = g_status.keys()
            for key in g_keys:
                status['g_' + key] = g_status.get(key)
            t_keys = t_status_1.keys()
            for key in t_keys:
                status['t1_' + key] = t_status_1.get(key)
            for key in t_keys:
                status['t2_' + key] = t_status_2.get(key)
            for key in g_keys:
                status[key] = g_status.get(key) + t_status_1.get(key) + t_status_2.get(key)
        else:
            g_loss, g_status, mask_sample = self.adaptive_compute_losses(ground_dict, data['grounding_anno'],
                                                                         "grounding",
                                                                         mask_sample=None)
            loss = g_loss
            status = {}
            g_keys = g_status.keys()
            for key in g_keys:
                status['g_' + key] = g_status.get(key)
        return loss, status

    def adaptive_compute_losses(self, pred_dict, gt_bbox, mode, mask_sample, return_status=True, labels=None):
        '''
            # to avoid the negative template affecting tracking learning
            use a loss mask to mask the loss computation whose iou less than threshold
        '''
        # Get boxes
        pred_boxes = pred_dict['pred_boxes']
        b = pred_boxes.shape[0]
        if torch.isnan(pred_boxes).any():
            raise ValueError("Network outputs is NAN! Stop Training")
        # Predict the center of target box
        pred_boxes_vec = box_cxcywh_to_xyxy(pred_boxes.squeeze())
        gt_boxes_vec = box_xywh_to_xyxy(gt_bbox).clamp(min=0.0, max=1.0).squeeze()  # (B,4)
        # compute giou and iou
        try:
            giou_loss, iou = self.objective['giou'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
        except:
            giou_loss, iou = torch.zeros(b).cuda(), torch.zeros(b).cuda()

        # compute l1 loss
        l1_loss = self.objective['l1'""](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)

        if mode == "grounding":
            iou_mask = iou > 0.5
        elif mode == "tracking":
            iou_mask = mask_sample.long()
            giou_loss = giou_loss * iou_mask
            l1_loss = l1_loss * iou_mask

        # weighted sum
        giou_loss = giou_loss.mean()
        l1_loss = l1_loss.mean()
        loss = self.loss_weight['giou'] * giou_loss + self.loss_weight['l1'] * l1_loss

        # compute cls loss if necessary
        if 'pred_scores' in pred_dict:
            score_loss = self.objective['score'](pred_dict['pred_scores'].view(-1), labels)
            loss = score_loss * self.loss_weight['score']

        if return_status:
            # status for log
            mean_iou = iou.detach().mean()
            if 'pred_scores' in pred_dict:
                status = {"Loss/total": loss.item(),
                          "Loss/scores": score_loss.item()}
            else:
                status = {"Loss/total": loss.item(),
                          "Loss/giou": giou_loss.item(),
                          "Loss/l1": l1_loss.item(),
                          "IoU": mean_iou.item()}
            return loss, status, iou_mask
        else:
            return loss

    def compute_losses(self, pred_dict, gt_bbox, return_status=True, labels=None):
        # Get boxes
        pred_boxes = pred_dict['pred_boxes']
        if torch.isnan(pred_boxes).any():
            raise ValueError("Network outputs is NAN! Stop Training")
        # Predict the center of target box
        pred_boxes_vec = box_cxcywh_to_xyxy(pred_boxes.squeeze())
        gt_boxes_vec = box_xywh_to_xyxy(gt_bbox).clamp(min=0.0, max=1.0).squeeze()  # (B,4)

        # compute giou and iou
        try:
            giou_loss, iou = self.objective['giou'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
        except:
            # print(e)
            giou_loss, iou = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()
        # compute l1 loss
        l1_loss = self.objective['l1'""](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
        giou_loss = giou_loss.mean()
        # weighted sum
        loss = self.loss_weight['giou'] * giou_loss + self.loss_weight['l1'] * l1_loss

        # compute cls loss if neccessary
        if 'pred_scores' in pred_dict:
            score_loss = self.objective['score'](pred_dict['pred_scores'].view(-1), labels)
            loss = score_loss * self.loss_weight['score']

        if return_status:
            # status for log
            mean_iou = iou.detach().mean()
            if 'pred_scores' in pred_dict:
                status = {"Loss/total": loss.item(),
                          "Loss/scores": score_loss.item()}
            else:
                status = {"Loss/total": loss.item(),
                          "Loss/giou": giou_loss.item(),
                          "Loss/l1": l1_loss.item(),
                          "IoU": mean_iou.item()}
            return loss, status
        else:
            return loss
