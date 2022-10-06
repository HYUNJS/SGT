import torch
import torch.nn as nn

from projects.CenterNet.centernet.layers.aspp import ASPP


class SingleHead(nn.Module):

    def __init__(self, in_channel, inter_channels, out_channel, bias_fill=False, bias_value=0, aspp_on=False):
        super(SingleHead, self).__init__()
        self.feat_conv = nn.Conv2d(in_channel, inter_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.out_conv = nn.Conv2d(inter_channels, out_channel, kernel_size=1)
        self.aspp_on = aspp_on
        if self.aspp_on:
            self.aspp = ASPP(in_channel, [6, 12, 18], in_channel)

        if bias_fill:
            self.out_conv.bias.data.fill_(bias_value)
        else:
            self.init_weights(self.feat_conv)
            self.init_weights(self.out_conv)

    def forward(self, x):
        if self.aspp_on:
            x = self.aspp(x)
        x = self.feat_conv(x)
        x = self.relu(x)
        x = self.out_conv(x)
        return x

    def init_weights(self, layers):
        for m in layers.modules():
            if isinstance(m, nn.Conv2d):
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class CenternetHead(nn.Module):
    """
    The head used in CenterNet for object classification and box regression.
    It has three subnet, with a common structure but separate parameters.
    """
    def __init__(self, cfg, in_channels):
        super(CenternetHead, self).__init__()
        self.cls_head = SingleHead(
            in_channels,
            cfg.MODEL.CENTERNET.HEAD.INTER_CHANNELS,
            cfg.MODEL.CENTERNET.NUM_CLASSES,
            bias_fill=True,
            bias_value=cfg.MODEL.CENTERNET.BIAS_VALUE,
        )
        if cfg.MODEL.CENTERNET.TLBR_FLAG:
            self.wh_head = SingleHead(in_channels, cfg.MODEL.CENTERNET.HEAD.INTER_CHANNELS, 4)
        else:
            self.wh_head = SingleHead(in_channels, cfg.MODEL.CENTERNET.HEAD.INTER_CHANNELS, 2)
        self.reg_head = SingleHead(in_channels, cfg.MODEL.CENTERNET.HEAD.INTER_CHANNELS, 2)
        self.identity_on = cfg.MODEL.IDENTITY_ON and cfg.MODEL.CENTERNET.ID_LOSS
        if self.identity_on:
            self.id_head = SingleHead(
                in_channels,
                cfg.MODEL.CENTERNET.HEAD.INTER_CHANNELS,
                cfg.MODEL.CENTERNET.HEAD.REID_DIM,
                aspp_on=cfg.MODEL.CENTERNET.HEAD.REID_ASPP_ON,
            )

    def forward(self, x):
        hm = self.cls_head(x)
        # hm = torch.sigmoid(hm)
        wh = self.wh_head(x)
        reg = self.reg_head(x)
        id = self.id_head(x) if self.identity_on else None
        pred = {
            'hm': hm,
            'wh': wh,
            'reg': reg,
            'id': id,
            'fmap': x
        }
        return pred