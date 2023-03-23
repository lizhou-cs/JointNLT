# for import modules
import importlib
import os

# loss function related
from torch.distributed.elastic.multiprocessing.errors import record
from torch.nn.functional import l1_loss
# distributed training related
from torch.nn.parallel import DistributedDataParallel as DDP

# forward propagation related
from lib.train.actors import JointNLTActor
# train pipeline related
from lib.train.trainers import LTRTrainer
from lib.utils.box_ops import giou_loss
# some more advanced functions
from .base_functions import *
# network related
from ..models.JointNLT import build_jointnlt


def prepare_input(res):
    res_t, res_s = res
    t = torch.FloatTensor(1, 3, res_t, res_t).cuda()
    s = torch.FloatTensor(1, 3, res_s, res_s).cuda()
    return dict(template=t, search=s)


@record
def run(settings):
    settings.description = 'Training script for JointNLT'
    # Record the training log
    log_dir = os.path.join(settings.save_dir, 'logs')
    if settings.local_rank in [-1, 0]:
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
    settings.log_file = os.path.join(log_dir, "%s-%s.log" % (settings.script_name, settings.config_name))
    # update the default configs with config file
    # setting from yaml and local.py and args
    if not os.path.exists(settings.cfg_file):
        raise ValueError("%s doesn't exist." % settings.cfg_file)

    # cfg file located in lib/config/scipt_name/config.py
    config_module = importlib.import_module("lib.config.%s.config" % settings.script_name)
    cfg = config_module.cfg

    config_module.update_config_from_file(settings.cfg_file)

    # update settings based on cfg
    update_settings(settings, cfg)

    if settings.local_rank in [0, -1]:
        print(cfg)
        with open(settings.log_file, 'a') as f:
            f.write(str(cfg)+'\n')
            f.close()
    # Build dataloaders
    loader_train, _ = build_dataloaders(cfg, settings)
    # Create network
    if settings.script_name == "jointnlt":
        net = build_jointnlt(cfg)
    else:
        raise ValueError("illegal script name")

    net.cuda()

    if settings.local_rank != -1:
        net = DDP(net, device_ids=[settings.local_rank], find_unused_parameters=True, broadcast_buffers=False)
        settings.device = torch.device("cuda:%d" % settings.local_rank)
    else:
        settings.device = torch.device("cuda:0")
    # Loss functions and Actors
    if settings.script_name == 'jointnlt':
        objective = {'giou': giou_loss, 'l1': l1_loss}
        loss_weight = {'giou': cfg.TRAIN.GIOU_WEIGHT, 'l1': cfg.TRAIN.L1_WEIGHT}
        actor = JointNLTActor(net=net, objective=objective, loss_weight=loss_weight, settings=settings)
    else:
        raise ValueError("illegal script name")
    # Optimizer, parameters, and learning rates
    optimizer, lr_scheduler = get_optimizer_scheduler(net, cfg)
    # Mixed Precision Training
    use_amp = getattr(cfg.TRAIN, "AMP", True)
    trainer = LTRTrainer(actor, [loader_train], optimizer, settings, lr_scheduler, use_amp=use_amp)
    print("Start train")
    # train process
    trainer.train(cfg.TRAIN.EPOCH, load_latest=True, fail_safe=True)
