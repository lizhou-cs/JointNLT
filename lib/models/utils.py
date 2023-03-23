"""
Mixformer utils.
"""
import logging

import torch
from torch import nn
from torch.nn.init import trunc_normal_


class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other checkpoints than torchvision.checkpoints.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()  # rsqrt(x): 1/sqrt(x), r: reciprocal
        bias = b - rm * scale
        return x * scale + bias


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


def _init_weights_trunc_normal(m):
    if isinstance(m, nn.Linear):
        logging.info('=> init weight of Linear from trunc norm')
        trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            logging.info('=> init bias of Linear to zeros')
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)
    elif isinstance(m, nn.Embedding):
        logging.info('=> init weight of Embedding from trunc norm')
        nn.init.trunc_normal_(m.weight, std=0.02)


def _init_weights_xavier(m):
    if isinstance(m, nn.Linear):
        logging.info('=> init weight of Linear from xavier uniform')
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            logging.info('=> init bias of Linear to zeros')
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)