import math
import numpy as np
import torch.nn as nn

from detectron2.utils.registry import Registry
from projects.CenterNet.centernet.modeling.backbone.dla import DLAUp, IDAUp
from projects.CenterNet.centernet.deform_conv import ModulatedDeformConvWithOff, DeformConvWithOff

UPSAMPLE_LAYER_REGISTRY = Registry("UPSAMPLE_LAYER")


class DeconvLayer(nn.Module):

    def __init__(
        self, in_planes,
        out_planes, deconv_kernel,
        deconv_stride=2, deconv_pad=1,
        deconv_out_pad=0, modulate_deform=True,
    ):
        super(DeconvLayer, self).__init__()
        if modulate_deform:
            self.dcn = ModulatedDeformConvWithOff(
                in_planes, out_planes,
                kernel_size=3, deformable_groups=1,
            )
        else:
            self.dcn = DeformConvWithOff(
                in_planes, out_planes,
                kernel_size=3, deformable_groups=1,
            )

        self.dcn_bn = nn.BatchNorm2d(out_planes)
        self.up_sample = nn.ConvTranspose2d(
            in_channels=out_planes,
            out_channels=out_planes,
            kernel_size=deconv_kernel,
            stride=deconv_stride, padding=deconv_pad,
            output_padding=deconv_out_pad,
            bias=False,
        )
        self._deconv_init()
        self.up_bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.dcn(x)
        x = self.dcn_bn(x)
        x = self.relu(x)
        x = self.up_sample(x)
        x = self.up_bn(x)
        x = self.relu(x)
        return x

    def _deconv_init(self):
        w = self.up_sample.weight.data
        f = math.ceil(w.size(2) / 2)
        c = (2 * f - 1 - f % 2) / (2. * f)
        for i in range(w.size(2)):
            for j in range(w.size(3)):
                w[0, 0, i, j] = \
                    (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
        for c in range(1, w.size(0)):
            w[c, 0, :, :] = w[0, 0, :, :]


@UPSAMPLE_LAYER_REGISTRY.register()
class CenternetDeconv(nn.Module):
    """
    The head used in CenterNet for object classification and box regression.
    It has three subnet, with a common structure but separate parameters.
    """
    def __init__(self, cfg):
        super(CenternetDeconv, self).__init__()
        # modify into config
        channels = cfg.MODEL.CENTERNET.DECONV_CHANNEL
        deconv_kernel = cfg.MODEL.CENTERNET.DECONV_KERNEL
        modulate_deform = cfg.MODEL.CENTERNET.MODULATE_DEFORM
        self.out_channels = channels[-1]

        self.deconv1 = DeconvLayer(
            channels[0], channels[1],
            deconv_kernel=deconv_kernel[0],
            modulate_deform=modulate_deform,
        )
        self.deconv2 = DeconvLayer(
            channels[1], channels[2],
            deconv_kernel=deconv_kernel[1],
            modulate_deform=modulate_deform,
        )
        self.deconv3 = DeconvLayer(
            channels[2], channels[3],
            deconv_kernel=deconv_kernel[2],
            modulate_deform=modulate_deform,
        )

    def forward(self, x):
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        return x


@UPSAMPLE_LAYER_REGISTRY.register()
class DLASegUp(nn.Module):
    def __init__(self, cfg):
        super(DLASegUp, self).__init__()
        channels = cfg.MODEL.DLA.CHANNELS
        down_ratio = cfg.MODEL.CENTERNET.DOWN_RATIO
        last_level = cfg.MODEL.DLA.LAST_LEVEL
        out_channel = cfg.MODEL.DLA.OUT_CHANNEL

        self.first_level = int(np.log2(down_ratio))
        self.last_level = last_level
        self.out_channels = channels[self.first_level]

        scales = [2 ** i for i in range(len(channels[self.first_level:]))]
        self.dla_up = DLAUp(self.first_level, channels[self.first_level:], scales)

        if out_channel == 0:
            out_channel = channels[self.first_level]
        ## aggregation P2~P4
        self.ida_up = IDAUp(out_channel, channels[self.first_level:self.last_level],
                            [2 ** i for i in range(self.last_level - self.first_level)])
        # ## aggregation P2~P5
        # self.ida_up = IDAUp(out_channel, channels[self.first_level:self.last_level+1],
        #                     [2 ** i for i in range(self.last_level - self.first_level+1)])
    def forward(self, x):
        x = self.dla_up(x)

        y = []
        for i in range(self.last_level - self.first_level):
            y.append(x[i].clone())
        y = self.ida_up(y, 0, len(y))
        return y[-1]
