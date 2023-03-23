from easydict import EasyDict as edict
import yaml

# TODO 删掉没用参数 重新整理一遍参数作用
"""
Add default config for TransNLT.
"""
cfg = edict()

# MODEL
cfg.MODEL = edict()
# BOX_HEAD
cfg.MODEL.HEAD_TYPE = "MLP"
cfg.MODEL.BOX_HEAD_HIDDEN_DIM = 256


# VISUAL
cfg.MODEL.VISUAL = edict()
cfg.MODEL.VISUAL.DILATION = False
cfg.MODEL.VISUAL.BACKBONE = 'VIT'
cfg.MODEL.VISUAL.LR = 10e-5

# LANGUAGE
cfg.MODEL.LANGUAGE = edict()
cfg.MODEL.LANGUAGE.IMPLEMENT = 'pytorch'
cfg.MODEL.LANGUAGE.TYPE = 'bert-base-uncased'
cfg.MODEL.LANGUAGE.PATH = 'pretrained/bert/bert-base-uncased.tar.gz'
cfg.MODEL.LANGUAGE.VOCAB_PATH = 'pretrained/bert/bert-base-uncased-vocab.txt'
# BERT
cfg.MODEL.LANGUAGE.BERT = edict()
cfg.MODEL.LANGUAGE.BERT.LR = 10e-5
cfg.MODEL.LANGUAGE.BERT.ENC_NUM = 12
cfg.MODEL.LANGUAGE.BERT.HIDDEN_DIM = 256
cfg.MODEL.LANGUAGE.BERT.MAX_QUERY_LEN = 40

# VISION-LANGUAGE MODEL
cfg.MODEL.VL = edict()
cfg.MODEL.VL.HIDDEN_DIM = 256
cfg.MODEL.VL.DROPOUT = 0.1
cfg.MODEL.VL.NHEAD = 8
cfg.MODEL.VL.DIM_FEEDFORWARD = 1024
cfg.MODEL.VL.ENC_LAYERS = 6
cfg.MODEL.VL.DEC_LAYERS = 6
cfg.MODEL.VL.NORM_BEFORE = False
cfg.MODEL.VL.RETURN_INTERMEDIATE = False
cfg.MODEL.VL.USE_VIS_CLS = True
cfg.MODEL.VL.USE_VIS_SEP = False
cfg.MODEL.VL.ACTIVATION = 'QUICK_GELU'
cfg.MODEL.VL.INIT = 'trunc_norm'

# TRAIN Hyper parameter
cfg.TRAIN = edict()
cfg.TRAIN.LR = 0.0001
cfg.TRAIN.WEIGHT_DECAY = 0.0001
cfg.TRAIN.EPOCH = 300
cfg.TRAIN.LR_DROP_EPOCH = 200
cfg.TRAIN.BATCH_SIZE = 16
cfg.TRAIN.NUM_WORKER = 8
cfg.TRAIN.OPTIMIZER = "ADAMW"
cfg.TRAIN.BACKBONE_MULTIPLIER = 0.1
cfg.TRAIN.GIOU_WEIGHT = 2.0
cfg.TRAIN.L1_WEIGHT = 5.0
cfg.TRAIN.DEEP_SUPERVISION = False
cfg.TRAIN.FREEZE_STAGE0 = False
cfg.TRAIN.AMP = False
cfg.TRAIN.PRINT_INTERVAL = 50
cfg.TRAIN.VAL_EPOCH_INTERVAL = 20
cfg.TRAIN.GRAD_CLIP_NORM = 0.1
# TRAIN.SCHEDULER
cfg.TRAIN.SCHEDULER = edict()
cfg.TRAIN.SCHEDULER.TYPE = "Mstep"
cfg.TRAIN.SCHEDULER.WARM_EPOCH = 20
cfg.TRAIN.SCHEDULER.MILESTONES = [200, 300, 500]
cfg.TRAIN.SCHEDULER.GAMMA = 0.1
cfg.TRAIN.SCHEDULER.DECAY_RATE = 0.1
cfg.TRAIN.USE_WANDB = False
# DATA
cfg.DATA = edict()
cfg.DATA.MEAN = [0.485, 0.456, 0.406]
cfg.DATA.STD = [0.229, 0.224, 0.225]
cfg.DATA.MAX_SAMPLE_INTERVAL = 200
cfg.DATA.MAX_SEQ_LENGTH = 40
# DATA.TRAIN
cfg.DATA.TRAIN = edict()
cfg.DATA.TRAIN.DATASETS_NAME = ["TNL2K", "LASOT"]  # ["LASOT", "GOT10K_vottrain"]
cfg.DATA.TRAIN.DATASETS_RATIO = [1, 1]
cfg.DATA.TRAIN.SAMPLE_PER_EPOCH = 60000
# DATA.VAL
cfg.DATA.VAL = edict()
cfg.DATA.VAL.DATASETS_NAME = ["LASOT_val"]
cfg.DATA.VAL.DATASETS_RATIO = [1]
cfg.DATA.VAL.SAMPLE_PER_EPOCH = 10000
# DATA.GROUNDING
cfg.DATA.GROUNDING = edict()
cfg.DATA.GROUNDING.SIZE = 320
cfg.DATA.GROUNDING.FACTOR = 2.0
cfg.DATA.GROUNDING.NUMBER = 1
cfg.DATA.GROUNDING.CENTER_JITTER = 1.5
cfg.DATA.GROUNDING.SCALE_JITTER = 0.5
cfg.DATA.SAMPLER_MODE = 'grounding'
# DATA.SEARCH
cfg.DATA.SEARCH = edict()
cfg.DATA.SEARCH.SIZE = 320
cfg.DATA.SEARCH.FACTOR = 4.0
cfg.DATA.SEARCH.NUMBER = 1
cfg.DATA.SEARCH.CENTER_JITTER = 4.5
cfg.DATA.SEARCH.SCALE_JITTER = 0.5
# DATA.TEMPLATE
cfg.DATA.TEMPLATE = edict()
cfg.DATA.TEMPLATE.SIZE = 128
cfg.DATA.TEMPLATE.FACTOR = 2.0
cfg.DATA.TEMPLATE.NUMBER = 1
cfg.DATA.TEMPLATE.CENTER_JITTER = 0
cfg.DATA.TEMPLATE.SCALE_JITTER = 0
cfg.DATA.SAMPLER_MODE = 'grounding'

# TEST
cfg.TEST = edict()
cfg.TEST.TEMPLATE_FACTOR = 2.0
cfg.TEST.TEMPLATE_SIZE = 128
cfg.TEST.SEARCH_FACTOR = 5.0
cfg.TEST.SEARCH_SIZE = 320
cfg.TEST.GROUNDING_SIZE = 320
cfg.TEST.GROUNDING_FACTOR = 2.0
cfg.TEST.EPOCH = 300

cfg.TEST.TEST_METHOD = "GROUND"    # choice in ['GROUND', 'TRACK', 'JOINT']


def _edict2dict(dest_dict, src_edict):
    if isinstance(dest_dict, dict) and isinstance(src_edict, dict):
        for k, v in src_edict.items():
            if not isinstance(v, edict):
                dest_dict[k] = v
            else:
                dest_dict[k] = {}
                _edict2dict(dest_dict[k], v)
    else:
        return


def gen_config(config_file):
    cfg_dict = {}
    _edict2dict(cfg_dict, cfg)
    with open(config_file, 'w') as f:
        yaml.dump(cfg_dict, f, default_flow_style=False)


def _update_config(base_cfg, exp_cfg):
    if isinstance(base_cfg, dict) and isinstance(exp_cfg, edict):
        for k, v in exp_cfg.items():
            if k in base_cfg:
                if not isinstance(v, dict):
                    base_cfg[k] = v
                else:
                    _update_config(base_cfg[k], v)
            else:
                raise ValueError("{} not exist in config.py".format(k))
    else:
        return


# load the information of filename' config to this config files
def update_config_from_file(filename):
    exp_config = None
    with open(filename) as f:
        exp_config = edict(yaml.safe_load(f))
        _update_config(cfg, exp_config)
