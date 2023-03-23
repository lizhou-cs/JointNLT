import os.path
import traceback
import cv2
#import ipdb
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

import lib.train.data.processing_utils as prutils
import lib.train.data.transforms as tfm
from lib.train.data import opencv_loader
from lib.utils import TensorDict
from lib.utils.box_ops import return_iou_boxes, box_xywh_to_xyxy, box_cxcywh_to_xyxy

torch.set_printoptions(precision=4, sci_mode=False)


def stack_tensors(x):
    if isinstance(x, (list, tuple)) and isinstance(x[0], torch.Tensor):
        return torch.stack(x)
    return x


class BaseProcessing:
    """ Base class for Processing. Processing class is used to process the data returned by a dataset, before passing it
     through the network. For example, it can be used to crop a search region around the object, apply various data
     augmentations, etc."""

    def __init__(self, transform=transforms.ToTensor(), template_transform=None, search_transform=None,
                 joint_transform=None, grounding_transform=None):
        """
        args:
            transform       - The set of transformations to be applied on the images. Used only if template_transform or
                                search_transform is None.
            template_transform - The set of transformations to be applied on the template images. If None, the 'transform'
                                argument is used instead.
            search_transform  - The set of transformations to be applied on the search images. If None, the 'transform'
                                argument is used instead.
            joint_transform - The set of transformations to be applied 'jointly' on the template and search images.  For
                                example, it can be used to convert both template and search images to grayscale.
        """
        self.transform = {'template': transform if template_transform is None else template_transform,
                          'search': transform if search_transform is None else search_transform,
                          'grounding': transform if grounding_transform is None else grounding_transform,
                          'joint': joint_transform}

    def __call__(self, data: TensorDict):
        raise NotImplementedError


class STARKProcessing(BaseProcessing):
    """ The processing class used for training LittleBoy. The images are processed in the following way.
    First, the target bounding box is jittered by adding some noise. Next, a square region (called search region )
    centered at the jittered target center, and of area search_area_factor^2 times the area of the jittered box is
    cropped from the image. The reason for jittering the target box is to avoid learning the bias that the target is
    always at the center of the search region. The search region is then resized to a fixed size given by the
    argument output_sz.

    """

    def __init__(self, search_area_factor, output_sz, center_jitter_factor, scale_jitter_factor,
                 mode='pair', settings=None, *args, **kwargs):
        """
        args:
            search_area_factor - The size of the search region  relative to the target size.
            output_sz - An integer, denoting the size to which the search region is resized. The search region is always
                        square.
            center_jitter_factor - A dict containing the amount of jittering to be applied to the target center before
                                    extracting the search region. See _get_jittered_box for how the jittering is done.
            scale_jitter_factor - A dict containing the amount of jittering to be applied to the target size before
                                    extracting the search region. See _get_jittered_box for how the jittering is done.
            mode - Either 'pair' or 'sequence'. If mode='sequence', then output has an extra dimension for frames
        """
        super().__init__(*args, **kwargs)
        self.search_area_factor = search_area_factor
        self.output_sz = output_sz
        self.center_jitter_factor = center_jitter_factor
        self.scale_jitter_factor = scale_jitter_factor
        self.mode = mode
        self.settings = settings

    def _get_jittered_box(self, box, mode):
        """ Jitter the input box
        args:
            box - input bounding box
            mode - string 'template' or 'search' indicating template or search data

        returns:
            torch.Tensor - jittered box
        """
        jittered_size = box[2:4] * torch.exp(torch.randn(2) * self.scale_jitter_factor[mode])
        max_offset = (jittered_size.prod().sqrt() * torch.tensor(self.center_jitter_factor[mode]).float())
        jittered_center = box[0:2] + 0.5 * box[2:4] + max_offset * (torch.rand(2) - 0.5)

        return torch.cat((jittered_center - 0.5 * jittered_size, jittered_size), dim=0)

    def __call__(self, data: TensorDict):
        """
        args:
            data - The input data, should contain the following fields:
                'template_images', search_images', 'template_anno', 'search_anno'
        returns:
            TensorDict - output data block with following fields:
                'template_images', 'search_images', 'template_anno', 'search_anno', 'test_proposals', 'proposal_iou'
        """
        # Apply joint transforms
        if self.transform['joint'] is not None:
            data['template_images'], data['template_anno'], data['template_masks'] = self.transform['joint'](
                image=data['template_images'], bbox=data['template_anno'], mask=data['template_masks'])
            data['search_images'], data['search_anno'], data['search_masks'] = self.transform['joint'](
                image=data['search_images'], bbox=data['search_anno'], mask=data['search_masks'], new_roll=False)

        for s in ['template', 'search']:
            assert self.mode == 'sequence' or len(data[s + '_images']) == 1, \
                "In pair mode, num train/test frames must be 1"

            # Add a uniform noise to the center pos
            jittered_anno = [self._get_jittered_box(a, s) for a in data[s + '_anno']]

            # 2021.1.9 Check whether data is valid. Avoid too small bounding boxes
            w, h = torch.stack(jittered_anno, dim=0)[:, 2], torch.stack(jittered_anno, dim=0)[:, 3]

            crop_sz = torch.ceil(torch.sqrt(w * h) * self.search_area_factor[s])
            if (crop_sz < 1).any():
                data['valid'] = False
                # print("Too small box is found. Replace it with new data.")
                return data
            # Crop image region centered at jittered_anno box and get the attention mask
            crops, boxes, att_mask, mask_crops = prutils.jittered_center_crop(data[s + '_images'], jittered_anno,
                                                                              data[s + '_anno'],
                                                                              self.search_area_factor[s],
                                                                              self.output_sz[s],
                                                                              masks=data[s + '_masks'])
            # Apply transforms
            data[s + '_images'], data[s + '_anno'], data[s + '_att'], data[s + '_masks'] = self.transform[s](
                image=crops, bbox=boxes, att=att_mask, mask=mask_crops, joint=False)

            # 2021.1.9 Check whether elements in data[s + '_att'] is all 1
            # Note that type of data[s + '_att'] is tuple, type of ele is torch.tensor
            for ele in data[s + '_att']:
                if (ele == 1).all():
                    data['valid'] = False
                    # print("Values of original attention mask are all one. Replace it with new data.")
                    return data
            # 2021.1.10 more strict conditions: require the donwsampled masks not to be all 1
            for ele in data[s + '_att']:
                feat_size = self.output_sz[s] // 16  # 16 is the backbone stride
                # (1,1,128,128) (1,1,256,256) --> (1,1,8,8) (1,1,16,16)
                mask_down = F.interpolate(ele[None, None].float(), size=feat_size).to(torch.bool)[0]
                if (mask_down == 1).all():
                    data['valid'] = False
                    # print("Values of down-sampled attention mask are all one. "
                    #       "Replace it with new data.")
                    return data

        data['valid'] = True
        # if we use copy-and-paste augmentation
        if data["template_masks"] is None or data["search_masks"] is None:
            data["template_masks"] = torch.zeros((1, self.output_sz["template"], self.output_sz["template"]))
            data["search_masks"] = torch.zeros((1, self.output_sz["search"], self.output_sz["search"]))
        # Prepare output
        if self.mode == 'sequence':
            data = data.apply(stack_tensors)
        else:
            data = data.apply(lambda x: x[0] if isinstance(x, list) else x)

        return data


class MixformerProcessing(BaseProcessing):
    """ The processing class used for training LittleBoy. The images are processed in the following way.
    First, the target bounding box is jittered by adding some noise. Next, a square region (called search region )
    centered at the jittered target center, and of area search_area_factor^2 times the area of the jittered box is
    cropped from the image. The reason for jittering the target box is to avoid learning the bias that the target is
    always at the center of the search region. The search region is then resized to a fixed size given by the
    argument output_sz.

    """

    def __init__(self, search_area_factor, output_sz, center_jitter_factor, scale_jitter_factor,
                 mode='pair', settings=None, *args, **kwargs):
        """
        args:
            search_area_factor - The size of the search region  relative to the target size.
            output_sz - An integer, denoting the size to which the search region is resized. The search region is always
                        square.
            center_jitter_factor - A dict containing the amount of jittering to be applied to the target center before
                                    extracting the search region. See _get_jittered_box for how the jittering is done.
            scale_jitter_factor - A dict containing the amount of jittering to be applied to the target size before
                                    extracting the search region. See _get_jittered_box for how the jittering is done.
            mode - Either 'pair' or 'sequence'. If mode='sequence', then output has an extra dimension for frames
        """
        super().__init__(*args, **kwargs)
        self.search_area_factor = search_area_factor
        self.output_sz = output_sz
        self.center_jitter_factor = center_jitter_factor
        self.scale_jitter_factor = scale_jitter_factor
        self.mode = mode
        self.settings = settings
        # self.label_function_params = label_function_params
        self.out_feat_sz = 20  ######## the output feature map size

    def _get_jittered_box(self, box, mode):
        """ Jitter the input box
        args:
            box - input bounding box
            mode - string 'template' or 'search' indicating template or search data

        returns:
            torch.Tensor - jittered box
        """

        jittered_size = box[2:4] * torch.exp(torch.randn(2) * self.scale_jitter_factor[mode])
        max_offset = (jittered_size.prod().sqrt() * torch.tensor(self.center_jitter_factor[mode]).float())
        jittered_center = box[0:2] + 0.5 * box[2:4] + max_offset * (torch.rand(2) - 0.5)

        return torch.cat((jittered_center - 0.5 * jittered_size, jittered_size), dim=0)

    def _generate_neg_proposals(self, box, min_iou=0.0, max_iou=0.3, sigma=0.5):
        """ Generates proposals by adding noise to the input box
        args:
            box - input box
        returns:
            torch.Tensor - Array of shape (num_proposals, 4) containing proposals
            torch.Tensor - Array of shape (num_proposals,) containing IoU overlap of each proposal with the input box. The
                        IoU is mapped to [-1, 1]
        """
        # Generate proposals
        # num_proposals = self.proposal_params['boxes_per_frame']
        # proposal_method = self.proposal_params.get('proposal_method', 'default')

        # if proposal_method == 'default':
        num_proposals = box.size(0)
        proposals = torch.zeros((num_proposals, 4)).to(box.device)
        gt_iou = torch.zeros(num_proposals)
        for i in range(num_proposals):
            proposals[i, :], gt_iou[i] = prutils.perturb_box(box[i], min_iou=min_iou, max_iou=max_iou,
                                                             sigma_factor=sigma)
        # elif proposal_method == 'gmm':
        #     proposals, _, _ = prutils.sample_box_gmm(box, self.proposal_params['proposal_sigma'],
        #                                                                      num_samples=num_proposals)
        #     gt_iou = prutils.iou(box.view(1,4), proposals.view(-1,4))

        # # Map to [-1, 1]
        # gt_iou = gt_iou * 2 - 1
        return proposals

    def __call__(self, data: TensorDict):
        """
        args:
            data - The input data, should contain the following fields:
                'template_images', search_images', 'template_anno', 'search_anno'
        returns:
            TensorDict - output data block with following fields:
                'template_images', 'search_images', 'template_anno', 'search_anno', 'test_proposals', 'proposal_iou'
        """
        # Apply joint transforms
        if self.transform['joint'] is not None:
            data['template_images'], data['template_anno'], data['template_masks'] = self.transform['joint'](
                image=data['template_images'], bbox=data['template_anno'], mask=data['template_masks'])
            data['search_images'], data['search_anno'], data['search_masks'] = self.transform['joint'](
                image=data['search_images'], bbox=data['search_anno'], mask=data['search_masks'], new_roll=False)

        for s in ['template', 'search']:
            assert self.mode == 'sequence' or len(data[s + '_images']) == 1, \
                "In pair mode, num train/test frames must be 1"

            # Add a uniform noise to the center pos - noise term 抖动box
            jittered_anno = [self._get_jittered_box(a, s) for a in data[s + '_anno']]

            # 2021.1.9 Check whether data is valid. Avoid too small bounding boxes
            w, h = torch.stack(jittered_anno, dim=0)[:, 2], torch.stack(jittered_anno, dim=0)[:, 3]

            crop_sz = torch.ceil(torch.sqrt(w * h) * self.search_area_factor[s])
            if (crop_sz < 1).any():
                data['valid'] = False
                # print("Too small box is found. Replace it with new data.")
                return data

            # Crop image region centered at jittered_anno box and get the attention mask
            crops, boxes, att_mask, mask_crops = prutils.jittered_center_crop(data[s + '_images'], jittered_anno,
                                                                              data[s + '_anno'],
                                                                              self.search_area_factor[s],
                                                                              self.output_sz[s],
                                                                              masks=data[s + '_masks'])
            # Apply transforms
            data[s + '_images'], data[s + '_anno'], data[s + '_att'], data[s + '_masks'] = self.transform[s](
                image=crops, bbox=boxes, att=att_mask, mask=mask_crops, joint=False)

            # 2021.1.9 Check whether elements in data[s + '_att'] is all 1
            # Note that type of data[s + '_att'] is tuple, type of ele is torch.tensor
            for ele in data[s + '_att']:
                if (ele == 1).all():
                    data['valid'] = False
                    # print("Values of original attention mask are all one. Replace it with new data.")
                    return data
            # 2021.1.10 more strict conditions: require the donwsampled masks not to be all 1
            for ele in data[s + '_att']:
                feat_size = self.output_sz[s] // 16  # 16 is the backbone stride
                # (1,1,128,128) (1,1,256,256) --> (1,1,8,8) (1,1,16,16)
                mask_down = F.interpolate(ele[None, None].float(), size=feat_size).to(torch.bool)[0]
                if (mask_down == 1).all():
                    data['valid'] = False
                    # print("Values of down-sampled attention mask are all one. "
                    #       "Replace it with new data.")
                    return data

        data['valid'] = True
        # if we use copy-and-paste augmentation
        if data["template_masks"] is None or data["search_masks"] is None:
            data["template_masks"] = torch.zeros((1, self.output_sz["template"], self.output_sz["template"]))
            data["search_masks"] = torch.zeros((1, self.output_sz["search"], self.output_sz["search"]))
        # Process NLP  grounding_nl_token_ids, grounding_nl_token_masks
        data['nl_token_ids'] = torch.tensor(data['nl_token_ids'])
        data['nl_token_masks'] = torch.tensor(data['nl_token_masks'])
        if (data['nl_token_masks'] == 0).all():
            data['valid'] = False
            print('nl_token_masks is error')
            return data
        # Prepare output
        if self.mode == 'sequence':
            data = data.apply(stack_tensors)
        else:
            data = data.apply(lambda x: x[0] if isinstance(x, list) else x)

        return data

    def _generate_regression_mask(self, target_center, mask_w, mask_h, mask_size=20):
        """
        NHW format
        :return:
        """
        k0 = torch.arange(mask_size, dtype=torch.float32, device=target_center.device).view(1, 1, -1)
        k1 = torch.arange(mask_size, dtype=torch.float32, device=target_center.device).view(1, -1, 1)

        d0 = (k0 - target_center[:, 0].view(-1, 1, 1)).abs()  # w, (b, 1, w)
        d1 = (k1 - target_center[:, 1].view(-1, 1, 1)).abs()  # h, (b, h, 1)
        # dist = d0.abs() + d1.abs()
        mask_w = mask_w.view(-1, 1, 1)
        mask_h = mask_h.view(-1, 1, 1)

        mask0 = torch.where(d0 <= mask_w * 0.5, torch.ones_like(d0), torch.zeros_like(d0))  # (b, 1, w)
        mask1 = torch.where(d1 <= mask_h * 0.5, torch.ones_like(d1), torch.zeros_like(d1))  # (b, h, 1)

        return mask0 * mask1  # (b, h, w)


# TODO: add TransProcessing, the first frame use original picture to predict box
# todo：修改call部分 传入template 和 search 其中 template不需要corp 只需要resize 和 transform
# todo 覆写一个Processing类
class TransNLTProcessing(BaseProcessing):
    """ The processing class used for training LittleBoy. The images are processed in the following way.
    First, the target bounding box is jittered by adding some noise. Next, a square region (called search region )
    centered at the jittered target center, and of area search_area_factor^2 times the area of the jittered box is
    cropped from the image. The reason for jittering the target box is to avoid learning the bias that the target is
    always at the center of the search region. The search region is then resized to a fixed size given by the
    argument output_sz.

    """

    def __init__(self, search_area_factor, output_sz, center_jitter_factor, scale_jitter_factor,
                 mode='pair', settings=None, *args, **kwargs):
        """
        args:
            search_area_factor - The size of the search region  relative to the target size.
            output_sz - An integer, denoting the size to which the search region is resized. The search region is always
                        square.
            center_jitter_factor - A dict containing the amount of jittering to be applied to the target center before
                                    extracting the search region. See _get_jittered_box for how the jittering is done.
            scale_jitter_factor - A dict containing the amount of jittering to be applied to the target size before
                                    extracting the search region. See _get_jittered_box for how the jittering is done.
            mode - Either 'pair' or 'sequence'. If mode='sequence', then output has an extra dimension for frames
        """
        super().__init__(*args, **kwargs)
        self.search_area_factor = search_area_factor
        self.output_sz = output_sz
        self.center_jitter_factor = center_jitter_factor
        self.scale_jitter_factor = scale_jitter_factor
        self.mode = mode
        self.settings = settings
        # self.label_function_params = label_function_params
        self.out_feat_sz = 20  # the output feature map size

    def _get_jittered_box(self, box, mode):
        """ Jitter the input box
        args:
            box - input bounding box
            mode - string 'template' or 'search' indicating template or search data

        returns:
            torch.Tensor - jittered box
        """
        jittered_size = box[2:4] * torch.exp(torch.randn(2) * self.scale_jitter_factor[mode])
        max_offset = (jittered_size.prod().sqrt() * torch.tensor(self.center_jitter_factor[mode]).float())
        jittered_center = box[0:2] + 0.5 * box[2:4] + max_offset * (torch.rand(2) - 0.5)
        return torch.cat((jittered_center - 0.5 * jittered_size, jittered_size), dim=0)

    def _generate_neg_proposals(self, box, min_iou=0.0, max_iou=0.3, sigma=0.5):
        """ Generates proposals by adding noise to the input box
        args:
            box - input box
        returns:
            torch.Tensor - Array of shape (num_proposals, 4) containing proposals
            torch.Tensor - Array of shape (num_proposals,) containing IoU overlap of each proposal with the input box. The
                        IoU is mapped to [-1, 1]
        """
        # Generate proposals
        # num_proposals = self.proposal_params['boxes_per_frame']
        # proposal_method = self.proposal_params.get('proposal_method', 'default')

        # if proposal_method == 'default':
        num_proposals = box.size(0)
        proposals = torch.zeros((num_proposals, 4)).to(box.device)
        gt_iou = torch.zeros(num_proposals)
        for i in range(num_proposals):
            proposals[i, :], gt_iou[i] = prutils.perturb_box(box[i], min_iou=min_iou, max_iou=max_iou,
                                                             sigma_factor=sigma)
        # elif proposal_method == 'gmm':
        #     proposals, _, _ = prutils.sample_box_gmm(box, self.proposal_params['proposal_sigma'],
        #                                                                      num_samples=num_proposals)
        #     gt_iou = prutils.iou(box.view(1,4), proposals.view(-1,4))

        # # Map to [-1, 1]
        # gt_iou = gt_iou * 2 - 1
        return proposals

    def __call__(self, data: TensorDict):
        """Generates proposals by adding noise to the input box
        args:
            data - The input data, should contain the following fields:
                'grounding_images', search_images', 'grounding_anno', 'search_anno'
        returns:
            TensorDict - output data block with following fields:
                'grounding_images', 'search_images', 'grounding_anno', 'search_anno', 'test_proposals', 'proposal_iou'
        """

        # Apply joint transforms
        if self.transform['joint'] is not None:
            data['search_images'], data['search_anno'], data['search_masks'] = self.transform['joint'](
                image=data['search_images'], bbox=data['search_anno'], mask=data['search_masks'])

        # for grounding image
        grounding_resize = [prutils.grounding_resize(im, self.output_sz['grounding'], box, ma)
                            for im, box, ma in zip(data['grounding_images'], data['grounding_anno'],
                                                   data['grounding_masks'])]

        resize_grounding_frames, resize_grounding_box, grounding_att_mask, resize_grounding_mask, \
        resize_grounding_top_left_coord = zip(*grounding_resize)
        data['grounding_coords'] = torch.tensor(resize_grounding_top_left_coord)
        # self.sava_visual_img(data['grounding_frames_path'][0], 'grounding',
        #                      resize_grounding_frames[0], resize_grounding_box[0], grounding_att_mask[0])
        if self.transform['grounding'] is not None:
            data['grounding_images'], data['grounding_anno'], data['grounding_att'], data['grounding_masks'] = \
                self.transform['grounding'](image=resize_grounding_frames, bbox=resize_grounding_box,
                                            att=grounding_att_mask, mask=resize_grounding_mask, joint=False)

        for s in ['search']:
            assert self.mode == 'sequence' or len(data[s + '_images']) == 1, \
                "In pair mode, num train/test frames must be 1"
            # Add a uniform noise to the center pos - noise term 抖动box
            jittered_anno = [self._get_jittered_box(a, s) for a in data[s + '_anno']]
            # get template box and search box
            # 2021.1.9 Check whether data is valid. Avoid too small bounding boxes
            w, h = torch.stack(jittered_anno, dim=0)[:, 2], torch.stack(jittered_anno, dim=0)[:, 3]
            crop_sz = torch.ceil(torch.sqrt(w * h) * self.search_area_factor[s])
            if (crop_sz < 1).any():
                data['valid'] = False
                # print("Too small box is found. Replace it with new data.")
                return data
            # Crop image region centered at jittered_anno box and get the attention mask
            crops, boxes, att_mask, mask_crops = prutils.jittered_center_crop(data[s + '_images'], jittered_anno,
                                                                              data[s + '_anno'],
                                                                              self.search_area_factor[s],
                                                                              self.output_sz[s],
                                                                              masks=data[s + '_masks'])
            # self.sava_visual_img(data['grounding_frames_path'][0], 'grounding', crops[0], boxes[0])
            # Apply transforms
            data[s + '_images'], data[s + '_anno'], data[s + '_att'], data[s + '_masks'] = self.transform[s](
                image=crops, bbox=boxes, att=att_mask, mask=mask_crops, joint=False)
            # 2021.1.9 Check whether elements in data[s + '_att'] is all 1
            # Note that type of data[s + '_att'] is tuple, type of ele is torch.tensor
        for s in ['search', 'grounding']:
            for ele in data[s + '_att']:
                if (ele == 1).all():
                    data['valid'] = False
                    print("Values of original attention mask are all one. Replace it with new data.")
                    return data
            # 2021.1.10 more strict conditions: require the donwsampled masks not to be all 1
            for ele in data[s + '_att']:
                feat_size = self.output_sz[s] // 16  # 16 is the backbone stride
                # (1,1,128,128) (1,1,256,256) --> (1,1,8,8) (1,1,16,16)
                mask_down = F.interpolate(ele[None, None].float(), size=feat_size).to(torch.bool)[0]
                if (mask_down == 1).all():
                    data['valid'] = False
                    print("Values of down-sampled attention mask are all one. Replace it with new data.")
                    return data
        data['valid'] = True
        # if we use copy-and-paste augmentation
        if data["grounding_masks"] is None or data["search_masks"] is None:
            data["grounding_masks"] = torch.zeros((1, self.output_sz["template"], self.output_sz["template"]))
            data["search_masks"] = torch.zeros((1, self.output_sz["search"], self.output_sz["search"]))

        # Process NLP  grounding_nl_token_ids, grounding_nl_token_masks
        data['nl_token_ids'] = torch.tensor(data['nl_token_ids'])
        data['nl_token_masks'] = torch.tensor(data['nl_token_masks'])
        if (data['nl_token_masks'] == 0).all():
            data['valid'] = False
            print('nl_token_masks is error')
            return data
        # Prepare output
        if self.mode == 'sequence':
            data = data.apply(stack_tensors)
        else:
            data = data.apply(lambda x: x[0] if isinstance(x, list) else x)

        return data

    def _generate_regression_mask(self, target_center, mask_w, mask_h, mask_size=20):
        """
        NHW format
        :return:
        """
        k0 = torch.arange(mask_size, dtype=torch.float32, device=target_center.device).view(1, 1, -1)
        k1 = torch.arange(mask_size, dtype=torch.float32, device=target_center.device).view(1, -1, 1)

        d0 = (k0 - target_center[:, 0].view(-1, 1, 1)).abs()  # w, (b, 1, w)
        d1 = (k1 - target_center[:, 1].view(-1, 1, 1)).abs()  # h, (b, h, 1)
        # dist = d0.abs() + d1.abs()
        mask_w = mask_w.view(-1, 1, 1)
        mask_h = mask_h.view(-1, 1, 1)

        mask0 = torch.where(d0 <= mask_w * 0.5, torch.ones_like(d0), torch.zeros_like(d0))  # (b, 1, w)
        mask1 = torch.where(d1 <= mask_h * 0.5, torch.ones_like(d1), torch.zeros_like(d1))  # (b, h, 1)

        return mask0 * mask1  # (b, h, w)


class TemplateProcessing(BaseProcessing):
    def __init__(self, settings, *args, **kwargs):
        """Crops templates by the predicted boxes of grounding process from original images
        args:
            search_area_factor - The size of the search region  relative to the target size.
            output_sz - An integer, denoting the size to which the search region is resized. The search region is always
                        square.
            center_jitter_factor - A dict containing the amount of jittering to be applied to the target center before
                                    extracting the search region. See _get_jittered_box for how the jittering is done.
            scale_jitter_factor - A dict containing the amount of jittering to be applied to the target size before
                                    extracting the search region. See _get_jittered_box for how the jittering is done.
            mode - Either 'pair' or 'sequence'. If mode='sequence', then output has an extra dimension for frames
        """
        super().__init__(*args, **kwargs)
        self.search_area_factor = settings.search_area_factor
        self.output_sz = settings.output_sz
        self.template_sz = self.output_sz['template']
        self.center_jitter_factor = settings.center_jitter_factor
        self.scale_jitter_factor = settings.scale_jitter_factor
        self.settings = settings
        self.local_rank = settings.local_rank if settings.local_rank > -1 else 0
        # self.label_function_params = label_function_params
        self.transform['joint'] = tfm.Transform(tfm.ToGrayscale(probability=0.05),
                                                tfm.RandomHorizontalFlip(probability=0.5))

        self.transform['template'] = tfm.Transform(tfm.ToTensorAndJitter(0.2),
                                                   tfm.RandomHorizontalFlip_Norm(probability=0.5),
                                                   tfm.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)))
        self.out_feat_sz = 20  # the output feature map size

    def _get_jittered_box(self, box, mode):
        """ Jitter the input box
        args:
            box - input bounding box
            mode - string 'template' or 'search' indicating template or search data

        returns:
            torch.Tensor - jittered box
        """
        jittered_size = box[2:4] * torch.exp(torch.randn(2) * self.scale_jitter_factor[mode])
        max_offset = (jittered_size.prod().sqrt() * torch.tensor(self.center_jitter_factor[mode]).float())
        jittered_center = box[0:2] + 0.5 * box[2:4] + max_offset * (torch.rand(2) - 0.5)
        return torch.cat((jittered_center - 0.5 * jittered_size, jittered_size), dim=0)

    def _get_templates_and_anno(self, grounding_path, grounding_dict, grounding_image_coords):
        for path in grounding_path:
            if isinstance(path, tuple):
                frames = [opencv_loader(p) for p in path]
        original_shapes = []
        for img in frames:
            original_shapes.append(img.shape)
        # get each original image shape the tensor shape is [b* 3 (h,w,c)]
        original_shapes = torch.tensor(original_shapes, device=self.local_rank)
        # the predict boxes
        pred_boxes = torch.round(grounding_dict * self.output_sz['grounding'])
        # Compute the IOU boxes between the predict boxes and the resized image in the grounding image
        iou_boxes = return_iou_boxes(box_xywh_to_xyxy(grounding_image_coords), box_cxcywh_to_xyxy(pred_boxes))
        # Compute the iou boxes' relative position in the resize image
        # Compute x y relative postition
        iou_boxes[:, 0:2] = torch.sub(iou_boxes[:, 0:2], grounding_image_coords[:, 0:2]).clamp(min=0)
        scale_factor = torch.div(original_shapes[:, 1], grounding_image_coords[:, 2]).unsqueeze(-1)
        # the correct size of predict boxes in original image
        scale_boxes = iou_boxes * scale_factor
        return frames, scale_boxes

    def __call__(self, data: TensorDict, grounding_path, grounding_dict, grounding_iamge_coords):
        """
        args:
            data - The input data, should contain the following fields:
                'template_images', search_images', 'template_anno', 'search_anno'
        returns:
            TensorDict - output data block with following fields:
                'template_images', 'search_images', 'template_anno', 'search_anno', 'test_proposals', 'proposal_iou'
        """
        frames, template_boxes = self._get_templates_and_anno(grounding_path, grounding_dict, grounding_iamge_coords)
        valid = False
        templates, annos, atts, masks = [], [], [], []
        for i, img in enumerate(frames):
            s = 'template'
            H, W, _ = img.shape
            template_mask = torch.zeros((H, W))
            # Apply joint transforms
            if self.transform['joint'] is not None:
                template_images, template_anno, template_masks = self.transform['joint'](
                    image=[img], bbox=[template_boxes[i]], mask=[template_mask])

            jittered_anno = [self._get_jittered_box(a, s) for a in template_anno]

            # 2021.1.9 Check whether data is valid. Avoid too small bounding boxes
            # w, h = jittered_anno[2], jittered_anno[3]
            w, h = torch.stack(jittered_anno, dim=0)[:, 2], torch.stack(jittered_anno, dim=0)[:, 3]
            crop_sz = torch.ceil(torch.sqrt(w * h) * self.search_area_factor[s])
            if (crop_sz < 1).any():
                valid = False
            # Crop image region centered at jittered_anno box and get the attention mask
            crops, boxes, att_mask, mask_crops = prutils.jittered_center_crop(template_images, jittered_anno,
                                                                              template_anno,
                                                                              self.search_area_factor[s],
                                                                              self.output_sz[s],
                                                                              masks=template_masks)
            # Apply transforms
            template_images, template_anno, template_att, template_mask = self.transform[s](
                image=crops, bbox=boxes, att=att_mask, mask=mask_crops, joint=False)
            # todo check the data is correct or not
            templates.extend(template_images)
            annos.extend(template_anno)
            atts.extend(template_att)
            masks.extend(template_mask)

        templates = torch.stack(templates, dim=0)
        annos = torch.stack(annos, dim=0)
        atts = torch.stack(atts, dim=0)
        masks = torch.stack(masks, dim=0)

        data['template_images'] = templates
        # Pay Attention  template_anno 是假标签 基于grounding产生的
        # data['template_anno'] = annos
        # data['template_masks'] = masks
        data['template_att'] = atts
        return data
