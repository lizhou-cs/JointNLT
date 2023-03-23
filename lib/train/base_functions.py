import torch
from torch.optim import RAdam
from torch.utils.data.distributed import DistributedSampler

import lib.train.data.transforms as tfm
from lib.train.data import sampler, opencv_loader, processing, LTRLoader
# datasets related
from lib.train.dataset import Lasot
from lib.train.dataset import Lasot_lmdb
from lib.train.dataset.lasottext import LasotText
from lib.train.dataset.otb99_lang import Otb99_lang
from lib.train.dataset.refcoco_seq import RefCOCOSeq
from lib.train.dataset.tnl2k import Tnl2k
from lib.utils.WarmupMultiStepLR import WarmupMultiStepLR


def update_settings(settings, cfg):
    settings.print_interval = cfg.TRAIN.PRINT_INTERVAL
    settings.search_area_factor = {'template': cfg.DATA.TEMPLATE.FACTOR,
                                   'search': cfg.DATA.SEARCH.FACTOR,
                                   'grounding': cfg.DATA.GROUNDING.FACTOR}
    settings.output_sz = {'template': cfg.DATA.TEMPLATE.SIZE,
                          'search': cfg.DATA.SEARCH.SIZE,
                          'grounding': cfg.DATA.GROUNDING.SIZE}
    settings.center_jitter_factor = {'template': cfg.DATA.TEMPLATE.CENTER_JITTER,
                                     'search': cfg.DATA.SEARCH.CENTER_JITTER,
                                     'grounding': cfg.DATA.GROUNDING.CENTER_JITTER}
    settings.scale_jitter_factor = {'template': cfg.DATA.TEMPLATE.SCALE_JITTER,
                                    'search': cfg.DATA.SEARCH.SCALE_JITTER,
                                    'grounding': cfg.DATA.GROUNDING.SCALE_JITTER}
    settings.grad_clip_norm = cfg.TRAIN.GRAD_CLIP_NORM
    settings.print_stats = None
    settings.batchsize = cfg.TRAIN.BATCH_SIZE
    settings.scheduler_type = cfg.TRAIN.SCHEDULER.TYPE


def names2datasets(name_list: list, settings, image_loader):
    print("dataset namelist:{}".format(name_list))
    assert isinstance(name_list, list)
    datasets = []
    for name in name_list:
        assert name in ["LASOT", "LASOT_VAL", "TNL2K", "OTB99", "REFCOCO", "REFCOCO+", "REFCOCOG", "LASOTTEXT", "OTB99",
                        "OTB99_TEST"]
        if name == "LASOT":
            if settings.use_lmdb:
                print("Building lasot dataset from lmdb")
                datasets.append(Lasot_lmdb(settings.env.lasot_lmdb_dir, split='train', image_loader=image_loader))
            else:
                datasets.append(Lasot(settings.env.lasot_dir, split='train', image_loader=image_loader))
        elif name == "TNL2K":
            if settings.use_lmdb:
                print("Building tnl2k dataset from lmdb")
                # datasets.append(Lasot_lmdb(settings.env.lasot_lmdb_dir, split='train', image_loader=image_loader))
            else:
                datasets.append(Tnl2k(settings.env.tnl2k_dir, split=None, image_loader=image_loader))
        elif name == "REFCOCO":
            if settings.use_lmdb:
                print("Building tnl2k dataset from lmdb")
                # datasets.append(Lasot_lmdb(settings.env.lasot_lmdb_dir, split='train', image_loader=image_loader))
            else:
                datasets.append(RefCOCOSeq(settings.env.coco_dir, split="train", image_loader=image_loader,
                                           name="refcoco", splitBy="google"))
        elif name == "REFCOCO+":
            if settings.use_lmdb:
                print("Building tnl2k dataset from lmdb")
                # datasets.append(Lasot_lmdb(settings.env.lasot_lmdb_dir, split='train', image_loader=image_loader))
            else:
                datasets.append(RefCOCOSeq(settings.env.coco_dir, split="train", image_loader=image_loader,
                                           name="refcoco+", splitBy="unc"))
        elif name == "REFCOCOG":
            if settings.use_lmdb:
                print("Building tnl2k dataset from lmdb")
                # datasets.append(Lasot_lmdb(settings.env.lasot_lmdb_dir, split='train', image_loader=image_loader))
            else:
                datasets.append(RefCOCOSeq(settings.env.coco_dir, split="train", image_loader=image_loader,
                                           name="refcocog", splitBy="google"))
        elif name == "LASOTTEXT":
            if settings.use_lmdb:
                print("Building lasottext dataset from lmdb")
                # datasets.append(Lasot_lmdb(settings.env.lasot_lmdb_dir, split='train', image_loader=image_loader))
            else:
                datasets.append(LasotText(settings.env.lasottext_dir, split='val', image_loader=image_loader))
        elif name == "OTB99":
            if settings.use_lmdb:
                print("Building lasottext dataset from lmdb")
                # datasets.append(Lasot_lmdb(settings.env.lasot_lmdb_dir, split='train', image_loader=image_loader))
            else:
                datasets.append(Otb99_lang(settings.env.otb99_dir, split='train', image_loader=image_loader))
        elif name == "OTB99_TEST":
            if settings.use_lmdb:
                print("Building lasottext dataset from lmdb")
                # datasets.append(Lasot_lmdb(settings.env.lasot_lmdb_dir, split='train', image_loader=image_loader))
            else:
                datasets.append(Otb99_lang(settings.env.otb99_dir, split='test', image_loader=image_loader))
        else:
            raise ValueError(f"Wrong dataset name:{name}")

    return datasets


def build_dataloaders(cfg, settings):
    # The tracking pairs processing module
    output_sz = settings.output_sz
    search_area_factor = settings.search_area_factor

    # Data transform
    # can‘t use RandomHorizontalFlip it would change the pic would not fit the language description.,
    transform_joint = tfm.Transform(tfm.ToGrayscale(probability=0.05),
                                    tfm.RandomHorizontalFlip(probability=0.5))

    # only for grounding pic, as possible as not change the original pic tfm.ToGrayscale(probability=0.05),
    transform_grounding = tfm.Transform(tfm.ToTensorAndJitter(0.2),
                                        tfm.Normalize(mean=cfg.DATA.MEAN, std=cfg.DATA.STD))
    # for template and search area
    transform_train = tfm.Transform(tfm.ToTensorAndJitter(0.2),
                                    tfm.RandomHorizontalFlip_Norm(probability=0.5),
                                    tfm.Normalize(mean=cfg.DATA.MEAN, std=cfg.DATA.STD))

    transform_val = tfm.Transform(tfm.ToTensor(),
                                  tfm.Normalize(mean=cfg.DATA.MEAN, std=cfg.DATA.STD))
    #
    settings.IOU_THREHOLD = 0.1
    # Train sampler and loader
    settings.num_grounding = getattr(cfg.DATA.GROUNDING, "NUMBER", 1)
    settings.num_search = getattr(cfg.DATA.SEARCH, "NUMBER", 1)
    sampler_mode = getattr(cfg.DATA, "SAMPLER_MODE", "grounding")
    train_score = getattr(cfg.TRAIN, "TRAIN_SCORE", False)

    data_processing_train = processing.TransNLTProcessing(search_area_factor=search_area_factor,
                                                          output_sz=output_sz,
                                                          center_jitter_factor=settings.center_jitter_factor,
                                                          scale_jitter_factor=settings.scale_jitter_factor,
                                                          mode='sequence',
                                                          transform=transform_train,
                                                          joint_transform=transform_joint,
                                                          grounding_transform=transform_grounding,
                                                          settings=settings)

    dataset_train = sampler.GroundingAndTrackingSampler(
        datasets=names2datasets(cfg.DATA.TRAIN.DATASETS_NAME, settings, opencv_loader),
        p_datasets=cfg.DATA.TRAIN.DATASETS_RATIO,
        samples_per_epoch=cfg.DATA.TRAIN.SAMPLE_PER_EPOCH,
        max_gap=cfg.DATA.MAX_SAMPLE_INTERVAL, num_search_frames=settings.num_search,
        num_template_frames=settings.num_grounding, processing=data_processing_train,
        frame_sample_mode=sampler_mode, train_cls=train_score, pos_prob=0.5,
        max_seq_len=cfg.DATA.MAX_SEQ_LENGTH, bert_model=cfg.MODEL.LANGUAGE.TYPE,
        bert_path=cfg.MODEL.LANGUAGE.VOCAB_PATH)

    train_sampler = DistributedSampler(dataset_train) if settings.local_rank != -1 else None
    shuffle = False if settings.local_rank != -1 else True

    loader_train = LTRLoader('train', dataset_train, training=True, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=shuffle,
                             num_workers=cfg.TRAIN.NUM_WORKER, drop_last=True, stack_dim=1, sampler=train_sampler)
    # Validation samplers and loaders
    data_processing_val = processing.TransNLTProcessing(search_area_factor=search_area_factor,
                                                        output_sz=output_sz,
                                                        center_jitter_factor=settings.center_jitter_factor,
                                                        scale_jitter_factor=settings.scale_jitter_factor,
                                                        mode='sequence',
                                                        transform=transform_val,
                                                        joint_transform=transform_joint,
                                                        grounding_transform=transform_grounding,
                                                        settings=settings)

    dataset_val = sampler.GroundingAndTrackingSampler(
        datasets=names2datasets(cfg.DATA.TRAIN.DATASETS_NAME, settings, opencv_loader),
        p_datasets=cfg.DATA.TRAIN.DATASETS_RATIO,
        samples_per_epoch=cfg.DATA.TRAIN.SAMPLE_PER_EPOCH,
        max_gap=cfg.DATA.MAX_SAMPLE_INTERVAL, num_search_frames=settings.num_search,
        num_template_frames=settings.num_grounding, processing=data_processing_val,
        frame_sample_mode=sampler_mode, train_cls=train_score, pos_prob=0.5,
        max_seq_len=cfg.DATA.MAX_SEQ_LENGTH, bert_model=cfg.MODEL.LANGUAGE.TYPE,
        bert_path=cfg.MODEL.LANGUAGE.VOCAB_PATH)

    val_sampler = DistributedSampler(dataset_val) if settings.local_rank != -1 else None
    loader_val = LTRLoader('val', dataset_val, training=False, batch_size=cfg.TRAIN.BATCH_SIZE,
                           num_workers=cfg.TRAIN.NUM_WORKER, drop_last=True, stack_dim=1, sampler=val_sampler,
                           epoch_interval=cfg.TRAIN.VAL_EPOCH_INTERVAL)

    return loader_train, loader_val


def get_optimizer_scheduler(net, cfg):
    VISUAL_LR = getattr(cfg.MODEL.VISUAL, "LR", 10e-5)
    LANGUAGE_LR = getattr(cfg.MODEL.LANGUAGE.BERT, "LR", 10e-5)

    param_dicts = [
        {
            "params": [p for n, p in net.named_parameters() if "backbone" not in n and p.requires_grad],
            "lr": cfg.TRAIN.LR * cfg.TRAIN.BACKBONE_MULTIPLIER,
        },
    ]
    if VISUAL_LR > 0:
        param_dicts.append({
            "params": [p for n, p in net.named_parameters() if "visual_backbone" in n and p.requires_grad],
            "lr": VISUAL_LR,
        })
    if LANGUAGE_LR > 0:
        param_dicts.append({
            "params": [p for n, p in net.named_parameters() if "language_backbone" in n and p.requires_grad],
            "lr": LANGUAGE_LR,
        })

    if cfg.TRAIN.OPTIMIZER == "ADAMW":
        optimizer = torch.optim.AdamW(param_dicts, lr=cfg.TRAIN.LR,
                                      weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    elif cfg.TRAIN.OPTIMIZER == "RADAM":
        optimizer = RAdam(param_dicts, lr=cfg.TRAIN.LR, weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    else:
        raise ValueError("Unsupported Optimizer")
    if cfg.TRAIN.SCHEDULER.TYPE == 'step':
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, cfg.TRAIN.LR_DROP_EPOCH)
    elif cfg.TRAIN.SCHEDULER.TYPE == "Mstep":
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                            milestones=cfg.TRAIN.SCHEDULER.MILESTONES,
                                                            gamma=cfg.TRAIN.SCHEDULER.GAMMA)
    elif cfg.TRAIN.SCHEDULER.TYPE == "WarmMstep":
        lr_scheduler = WarmupMultiStepLR(optimizer, milestones=cfg.TRAIN.SCHEDULER.MILESTONES,
                                         gamma=cfg.TRAIN.SCHEDULER.GAMMA, warmup_iters=cfg.TRAIN.SCHEDULER.WARM_EPOCH)
    else:
        raise ValueError("Unsupported scheduler")
    return optimizer, lr_scheduler
