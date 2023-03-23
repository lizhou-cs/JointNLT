import argparse
import cProfile
import importlib
import os
import sys
import time

import torch
from thop import clever_format
from thop import profile

prj_path = os.path.join(os.path.dirname(__file__), '..')
if prj_path not in sys.path:
    sys.path.append(prj_path)

from lib.utils.misc import NestedTensor
from lib.models.JointNLT import build_jointnlt


def parse_args():
    """
    args for training.
    """
    parser = argparse.ArgumentParser(description='Parse args for training')
    # for train
    parser.add_argument('--script', type=str, default='jointnlt', choices=['jointnlt'],
                        help='training script name')
    parser.add_argument('--config', type=str, default='baseline', help='yaml configure file name')
    parser.add_argument('--display_name', type=str, default='TransNLT')
    parser.add_argument('--online_skip', type=int, default=200, help='the skip interval of mixformer-online')
    args = parser.parse_args()

    return args


def evaluate(model, text, template, search, display_info='JointNLT'):
    """Compute Macs,FLOPs, Params, and Speed"""
    # compmut grounding process
    print("<==== grounding process ====>")
    macs, params = profile(model, inputs=(text, template, None, search, "grounding"),
                           custom_ops={}, verbose=False)
    macs, params = clever_format([macs, params], "%.3f")
    print('==>Macs is ', macs)
    print('==>Params is ', params)
    # test speed
    # T_w = 10
    T_t = 1000
    print("testing speed ...")
    with torch.no_grad():
        # overall
        start = time.time()
        for i in range(T_t):
            _ = model(text, None, None, search, "grounding")

        end = time.time()
        cost_time = end - start
        avg_lat = (end - start) / T_t
        print("Cost time: {}".format(cost_time))
        print("\033[0;32;40m The average overall FPS of {} is {}.\033[0m".format(display_info, 1.0 / avg_lat))

    # compmut tracking process
    print("<==== tracking process ====>")
    flops, params = profile(model, inputs=(text, template, None, search, "tracking"),
                            custom_ops={}, verbose=False)
    flops, params = clever_format([flops, params], "%.3f")
    print("FLOPs:{}".format(flops))
    print("Params:{}".format(params))
    # test speed
    # T_w = 10
    T_t = 1000
    print("testing speed ...")
    with torch.no_grad():
        # overall
        start = time.time()
        text_dict = model.forward_text(text)
        template_dict = model.forward_vision_backbone(template)
        for i in range(T_t):
            _ = model.forward_test(text_dict, template_dict, search, None)

        end = time.time()
        cost_time = end - start
        avg_lat = (end - start) / T_t
        print("Cost time: {}".format(cost_time))
        print("\033[0;32;40m The average overall FPS of {} is {}.\033[0m".format(display_info, 1.0 / avg_lat))


def get_data(bs, sz):
    img_patch = torch.randn(bs, 3, sz, sz)
    img_mask = torch.randint(0, 2, (bs, sz, sz)).type(torch.float)
    img_input = NestedTensor(img_patch, img_mask)
    return img_input


def get_text(bs, text_length):
    text_ids, text_mask = torch.randint(5, (bs, text_length)), torch.randint(5, (bs, text_length))
    text_input = NestedTensor(text_ids, text_mask)
    return text_input


def main():
    device = "cuda:0"
    torch.cuda.set_device(device)
    args = parse_args()
    '''update cfg'''
    prj_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    yaml_fname = os.path.join(prj_dir, 'experiments/%s/%s.yaml' % (args.script, args.config))
    print("yaml_fname: {}".format(yaml_fname))
    config_module = importlib.import_module('lib.config.%s.config' % args.script)
    cfg = config_module.cfg
    config_module.update_config_from_file(yaml_fname)
    print("cfg: {}".format(cfg))
    '''set some values'''
    bs = 1
    z_sz = cfg.TEST.TEMPLATE_SIZE
    x_sz = cfg.TEST.SEARCH_SIZE
    t_len = cfg.MODEL.LANGUAGE.BERT.MAX_QUERY_LEN

    if args.script == "jointnlt":
        model = build_jointnlt(cfg)
        # get the template and search
        template = get_data(bs, z_sz)
        search = get_data(bs, x_sz)
        text = get_text(bs, t_len)

        # transfer to device
        model = model.to(device)
        template = template.to(device)
        search = search.to(device)
        text = text.to(device)
        # evaluate the model properties
        evaluate(model, text, template, search, args.display_name)


if __name__ == "__main__":
    # cProfile.run('main()', 'restats')
    main()
