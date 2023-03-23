import torch.nn as nn
import torch
import torch.nn.functional as F
from lib.models.utils import FrozenBatchNorm2d


def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1,
         freeze_bn=False):
    if freeze_bn:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, bias=True),
            FrozenBatchNorm2d(out_planes),
            nn.ReLU(inplace=False))
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, bias=True),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=False))


class Corner_Predictor(nn.Module):
    """ Corner Predictor module"""

    def __init__(self, inplanes=64, channel=256, feat_sz=20, stride=16, freeze_bn=False):
        super(Corner_Predictor, self).__init__()
        self.feat_sz = feat_sz
        self.stride = stride
        self.img_sz = self.feat_sz * self.stride
        '''top-left corner'''
        self.conv1_tl = conv(inplanes, channel, freeze_bn=freeze_bn)
        self.conv2_tl = conv(channel, channel // 2, freeze_bn=freeze_bn)
        self.conv3_tl = conv(channel // 2, channel // 4, freeze_bn=freeze_bn)
        self.conv4_tl = conv(channel // 4, channel // 8, freeze_bn=freeze_bn)
        self.conv5_tl = nn.Conv2d(channel // 8, 1, kernel_size=1)

        '''bottom-right corner'''
        self.conv1_br = conv(inplanes, channel, freeze_bn=freeze_bn)
        self.conv2_br = conv(channel, channel // 2, freeze_bn=freeze_bn)
        self.conv3_br = conv(channel // 2, channel // 4, freeze_bn=freeze_bn)
        self.conv4_br = conv(channel // 4, channel // 8, freeze_bn=freeze_bn)
        self.conv5_br = nn.Conv2d(channel // 8, 1, kernel_size=1)

        '''about coordinates and indexs'''
        with torch.no_grad():
            self.indice = torch.arange(0, self.feat_sz).view(-1, 1) * self.stride
            # generate mesh-grid
            self.coord_x = self.indice.repeat((self.feat_sz, 1)) \
                .view((self.feat_sz * self.feat_sz,)).float().cuda()
            self.coord_y = self.indice.repeat((1, self.feat_sz)) \
                .view((self.feat_sz * self.feat_sz,)).float().cuda()

    def forward(self, x, return_dist=False, softmax=True):
        """ Forward pass with input x. """
        score_map_tl, score_map_br = self.get_score_map(x)
        if return_dist:
            coorx_tl, coory_tl, prob_vec_tl = self.soft_argmax(score_map_tl, return_dist=True, softmax=softmax)
            coorx_br, coory_br, prob_vec_br = self.soft_argmax(score_map_br, return_dist=True, softmax=softmax)
            return torch.stack((coorx_tl, coory_tl, coorx_br, coory_br), dim=1) / self.img_sz, prob_vec_tl, prob_vec_br
        else:
            coorx_tl, coory_tl = self.soft_argmax(score_map_tl)
            coorx_br, coory_br = self.soft_argmax(score_map_br)
            return torch.stack((coorx_tl, coory_tl, coorx_br, coory_br), dim=1) / self.img_sz

    def get_score_map(self, x):
        # top-left branch
        x_tl1 = self.conv1_tl(x)
        x_tl2 = self.conv2_tl(x_tl1)
        x_tl3 = self.conv3_tl(x_tl2)
        x_tl4 = self.conv4_tl(x_tl3)
        score_map_tl = self.conv5_tl(x_tl4)

        # bottom-right branch
        x_br1 = self.conv1_br(x)
        x_br2 = self.conv2_br(x_br1)
        x_br3 = self.conv3_br(x_br2)
        x_br4 = self.conv4_br(x_br3)
        score_map_br = self.conv5_br(x_br4)
        return score_map_tl, score_map_br

    def soft_argmax(self, score_map, return_dist=False, softmax=True):
        """ get soft-argmax coordinate for a given heatmap """
        score_vec = score_map.view((-1, self.feat_sz * self.feat_sz))  # (batch, feat_sz * feat_sz)
        prob_vec = nn.functional.softmax(score_vec, dim=1)
        exp_x = torch.sum((self.coord_x * prob_vec), dim=1)
        exp_y = torch.sum((self.coord_y * prob_vec), dim=1)
        if return_dist:
            if softmax:
                return exp_x, exp_y, prob_vec
            else:
                return exp_x, exp_y, score_vec
        else:
            return exp_x, exp_y


class Hierarchical_Corner_Predictor(nn.Module):
    """ Corner Predictor module"""

    def __init__(self, inplanes=64, channel=256, feat_sz=20, stride=16, freeze_bn=False):
        super(Hierarchical_Corner_Predictor, self).__init__()
        self.feat_sz = feat_sz
        self.stride = stride
        self.img_sz = self.feat_sz * self.stride
        '''top-left corner'''
        self.conv1_tl = conv(inplanes, inplanes, freeze_bn=freeze_bn)
        self.conv2_tl = conv(inplanes * 2, inplanes, freeze_bn=freeze_bn)
        self.conv3_tl = conv(inplanes * 2, inplanes, freeze_bn=freeze_bn)
        self.conv4_tl = conv(inplanes * 2, inplanes // 2, freeze_bn=freeze_bn)
        self.conv5_tl = conv(inplanes // 2, inplanes // 8, freeze_bn=freeze_bn)
        self.conv6_tl = nn.Conv2d(inplanes // 8, 1, kernel_size=1)

        '''bottom-right corner'''
        self.conv1_br = conv(inplanes, inplanes, freeze_bn=freeze_bn)
        self.conv2_br = conv(inplanes * 2, inplanes, freeze_bn=freeze_bn)
        self.conv3_br = conv(inplanes * 2, inplanes, freeze_bn=freeze_bn)
        self.conv4_br = conv(inplanes * 2, inplanes // 2, freeze_bn=freeze_bn)
        self.conv5_br = conv(inplanes // 2, inplanes // 8, freeze_bn=freeze_bn)
        self.conv6_br = nn.Conv2d(inplanes // 8, 1, kernel_size=1)

        self.layer1_conv = conv(inplanes // 2, inplanes, freeze_bn=freeze_bn)
        self.layer2_conv = conv(inplanes, inplanes, freeze_bn=freeze_bn)
        self.layer3_conv = conv(inplanes * 2, inplanes, freeze_bn=freeze_bn)

        '''about coordinates and indexs'''
        with torch.no_grad():
            self.indice = torch.arange(0, self.feat_sz * 4).view(-1, 1) * (self.stride // 4)
            # generate mesh-gridd
            self.coord_x = self.indice.repeat((self.feat_sz * 4, 1)) \
                .view((self.feat_sz * self.feat_sz * 16,)).float().cuda()
            self.coord_y = self.indice.repeat((1, self.feat_sz * 4)) \
                .view((self.feat_sz * self.feat_sz * 16,)).float().cuda()

    def forward(self, x, img_feas, return_dist=False, softmax=True):
        """ Forward pass with input x. """
        score_map_tl, score_map_br = self.get_score_map(x, img_feas)
        if return_dist:
            coorx_tl, coory_tl, prob_vec_tl = self.soft_argmax(score_map_tl, return_dist=True, softmax=softmax)
            coorx_br, coory_br, prob_vec_br = self.soft_argmax(score_map_br, return_dist=True, softmax=softmax)
            return torch.stack((coorx_tl, coory_tl, coorx_br, coory_br), dim=1) / self.img_sz, prob_vec_tl, prob_vec_br
        else:
            coorx_tl, coory_tl = self.soft_argmax(score_map_tl)
            coorx_br, coory_br = self.soft_argmax(score_map_br)
            return torch.stack((coorx_tl, coory_tl, coorx_br, coory_br), dim=1) / self.img_sz

    def get_score_map(self, x, img_feas):
        # input (B, C, H', W')
        # top-left branch
        # (B, C, H', W') ->  (B, C, H', W')
        layer3 = self.layer3_conv(img_feas[-1])  # (B, 2*C, H/16, W/16)
        layer2 = self.layer2_conv(img_feas[-2])  # (B, C, H/8, W/8)
        layer1 = self.layer1_conv(img_feas[-3])  # (B, C/2, H/4, W/4)

        x_tl1 = self.conv1_tl(x)  # (B, C, H/16, W/16)
        x_tl1 = torch.concat((x_tl1, layer3), dim=1)
        x_tl2 = self.conv2_tl(x_tl1)
        x_tl2 = F.interpolate(x_tl2, size=(self.feat_sz * 2, self.feat_sz * 2), mode='bilinear')

        x_tl2 = torch.concat((x_tl2, layer2), dim=1)
        x_tl3 = self.conv3_tl(x_tl2)
        x_tl3 = F.interpolate(x_tl3, size=(self.feat_sz * 4, self.feat_sz * 4), mode='bilinear')

        x_tl3 = torch.concat((x_tl3, layer1), dim=1)
        x_tl4 = self.conv4_tl(x_tl3)
        x_tl5 = self.conv5_tl(x_tl4)
        # (B, C/8, H', W') ->  (B, 1, H', W')
        score_map_tl = self.conv6_tl(x_tl5)

        # bottom-right branch
        x_br1 = self.conv1_br(x)  # (B, C, H/16, W/16)
        x_br1 = torch.concat((x_br1, layer3), dim=1)
        x_br2 = self.conv2_br(x_br1)
        x_br2 = F.interpolate(x_br2, size=(self.feat_sz * 2, self.feat_sz * 2), mode='bilinear')

        x_br2 = torch.concat((x_br2, layer2), dim=1)
        x_br3 = self.conv3_br(x_br2)
        x_br3 = F.interpolate(x_br3, size=(self.feat_sz * 4, self.feat_sz * 4), mode='bilinear')

        x_br3 = torch.concat((x_br3, layer1), dim=1)
        x_br4 = self.conv4_br(x_br3)
        x_br5 = self.conv5_br(x_br4)
        # (B, C/8, H', W') ->  (B, 1, H', W')
        score_map_br = self.conv6_br(x_br5)
        return score_map_tl, score_map_br

    def soft_argmax(self, score_map, return_dist=False, softmax=True):
        """ get soft-argmax coordinate for a given heatmap """
        score_vec = score_map.view((-1, self.feat_sz * self.feat_sz * 16))  # (batch, feat_sz * feat_sz)
        prob_vec = nn.functional.softmax(score_vec, dim=1)
        exp_x = torch.sum((self.coord_x * prob_vec), dim=1)
        exp_y = torch.sum((self.coord_y * prob_vec), dim=1)
        if return_dist:
            if softmax:
                return exp_x, exp_y, prob_vec
            else:
                return exp_x, exp_y, score_vec
        else:
            return exp_x, exp_y


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, BN=False):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        if BN:
            self.layers = nn.ModuleList(nn.Sequential(nn.Linear(n, k), nn.BatchNorm1d(k))
                                        for n, k in zip([input_dim] + h, h + [output_dim]))
        else:
            self.layers = nn.ModuleList(nn.Linear(n, k)
                                        for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build_box_head(cfg):
    if cfg.MODEL.HEAD_TYPE == "MLP":
        hidden_dim = cfg.MODEL.BOX_HEAD_HIDDEN_DIM
        mlp_head = MLP(hidden_dim, hidden_dim, 4, 3)  # dim_in, dim_hidden, dim_out, 3 layers
        return mlp_head
    elif cfg.MODEL.HEAD_TYPE == "CORNER":
        stride = 16
        feat_sz = int(cfg.DATA.SEARCH.SIZE / stride)
        channel = getattr(cfg.MODEL, "HEAD_DIM", 256)
        # print("head channel: %d" % channel)
        if cfg.MODEL.HEAD_TYPE == "CORNER":
            corner_head = Corner_Predictor(inplanes=cfg.MODEL.VL.HIDDEN_DIM, channel=channel,
                                           feat_sz=feat_sz, stride=stride)
        else:
            raise ValueError()
        return corner_head
    else:
        raise ValueError("HEAD TYPE %s is not supported." % cfg.MODEL.BOX_HEAD.HEAD_TYPE)
