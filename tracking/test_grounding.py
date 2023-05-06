import argparse
import importlib
import os
import sys

import torch
from pytorch_pretrained_bert import BertTokenizer
from tqdm import tqdm

lib_path = os.path.join(os.path.dirname(__file__), '../')
if lib_path not in sys.path:
    sys.path.append(lib_path)
from lib.train.dataset.refcoco_seq import RefCOCOSeq
import lib.train.admin.settings as ws_settings
from lib.models.JointNLT import build_jointnlt
from lib.test.evaluation.environment import env_settings
from lib.test.tracker.tracker_utils import Preprocessor_wo_mask
from lib.train.data import opencv_loader
from lib.train.data.processing_utils import grounding_resize
from lib.utils.box_ops import box_iou, box_xywh_to_xyxy, box_cxcywh_to_xyxy
from lib.utils.misc import NestedTensor


def extract_token_from_nlp(tokenizer, nlp, seq_length):
    """ use tokenizer to convert nlp to tokens
    param:
        nlp:  a sentence of natural language
        seq_length: the max token length, if token length larger than seq_len then cut it,
        elif less than, append '0' token at the reef.
    return:
        token_ids and token_marks
    """
    nlp_token = tokenizer.tokenize(nlp)
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
    input_ids = tokenizer.convert_tokens_to_ids(tokens)

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


def get_text_input(tokenizer, nlp, max_length):
    text_ids, text_mask = extract_token_from_nlp(tokenizer, nlp, max_length)
    t_ids = torch.tensor(text_ids).unsqueeze(0).cuda()
    t_mask = torch.tensor(text_mask).unsqueeze(0).cuda()
    return NestedTensor(t_ids, t_mask)


def get_vis_input(preprocessor, img, gt_box, grounding_sz):
    if isinstance(img, str):
        image = opencv_loader(img)
    else:
        image = img
    im_crop_padded, scale_gt, att_mask, mask_crop_padded, image_top_coords = grounding_resize(image, grounding_sz,
                                                                                              gt_box, None)
    grounding_patch = preprocessor.process(im_crop_padded).cuda()
    att_mask = torch.tensor(att_mask).unsqueeze(0).cuda()
    grounding_patch = NestedTensor(grounding_patch, att_mask)

    return grounding_patch, scale_gt


def load_config(script_name, config_name):
    prj_dir = env_settings().prj_dir
    settings = ws_settings.Settings()
    settings.script_name = script_name
    settings.config_name = config_name

    settings.cfg_file = os.path.join(prj_dir, 'experiments/%s/%s.yaml' % (script_name, config_name))
    config_module = importlib.import_module("lib.config.%s.config" % script_name)
    cfg = config_module.cfg

    config_module.update_config_from_file(settings.cfg_file)
    return settings, cfg


def eval_coco(script_name, config_name, checkpoint_path, dataset_name='refcocog', split='val', splitBy='google'):
    # load config
    settings, cfg = load_config(script_name, config_name)

    # build and load the network
    net = build_jointnlt(cfg).cuda()
    net = net.eval()
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    net.load_state_dict(checkpoint_dict['net'])

    # Load the COCO set.
    coco = RefCOCOSeq(settings.env.coco_dir, split=split, image_loader=opencv_loader,
                      name=dataset_name, splitBy=splitBy)

    # load parameters
    GROUNDING_SZ = cfg.TEST.GROUNDING_SIZE
    MAX_LENGTH_TEXT = cfg.MODEL.LANGUAGE.BERT.MAX_QUERY_LEN

    # preprocessors
    preprocessor = Preprocessor_wo_mask()
    tokenizer = BertTokenizer.from_pretrained('pretrained/bert/bert-base-uncased-vocab.txt', do_lower_case=True)

    seqs_result = []
    seqs_gt = []
    sequence_list = coco.sequence_list

    for seq_id in tqdm(sequence_list):
        img, _, _ = coco.get_frames(seq_id, [0])
        anno = coco.get_sequence_info(seq_id)
        nlp = anno['nlp']
        gt_box = anno['bbox'].squeeze(0).cuda()
        text_input = get_text_input(tokenizer, nlp, max_length=MAX_LENGTH_TEXT)
        vision_input, scale_gt = get_vis_input(preprocessor, img[0], gt_box, grounding_sz=GROUNDING_SZ)
        with torch.no_grad():
            rs = net(text_input, None, None, vision_input, 'grounding')
            pred_box = box_cxcywh_to_xyxy(rs['pred_boxes']).squeeze()
        seqs_result.append(pred_box)
        seqs_gt.append(box_xywh_to_xyxy(scale_gt))

    # compute result
    gt_boxes = torch.stack(seqs_gt, dim=0)
    pred_boxes = torch.stack(seqs_result, dim=0)
    iou, _ = box_iou(gt_boxes, pred_boxes)
    print('{} grounding performance evaluation'.format(dataset_name))
    avg_iou = iou.mean()
    threholds = [0.25, 0.5, 0.75]
    nums = gt_boxes.shape[0]
    acc = [(iou > t).sum() / nums for t in threholds]
    for j in range(len(threholds)):
        print('ACC({}) : {:4f}    '.format(threholds[j], acc[j]), end="")
    print('\nAVG_IOU : {:4f}'.format(avg_iou))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run tracker on sequence or dataset.')
    parser.add_argument('--script', type=str, help='Name of tracking method.')
    parser.add_argument('--config', type=str, help='Name of config file.')
    parser.add_argument('--ckpt', type=str, default=None, help="Tracking model path.")
    args = parser.parse_args()

    eval_coco(args.script, args.config, args.ckpt)
