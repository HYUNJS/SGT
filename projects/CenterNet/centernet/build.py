import torch
import torch.nn as nn

from detectron2.modeling.backbone import BACKBONE_REGISTRY
from detectron2.modeling.backbone.backbone import Backbone
from projects.CenterNet.centernet.layers.centernet_deconv import UPSAMPLE_LAYER_REGISTRY
from projects.CenterNet.centernet.layers.centernet_head import CenternetHead


def build_centernet_backbone(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    backbone = BACKBONE_REGISTRY.get(backbone_name)(cfg)
    assert isinstance(backbone, Backbone)
    return backbone


def build_centernet_upsample_layer(cfg):
    upsample_layer_name = cfg.MODEL.CENTERNET.UPSAMPLE_LAYER.NAME
    if upsample_layer_name == '': ## currently, only for hg104 backbone
        upsample_layer = nn.Identity()
        upsample_layer.out_channels = 256
    else:
        upsample_layer = UPSAMPLE_LAYER_REGISTRY.get(upsample_layer_name)(cfg)
    return upsample_layer


def build_centernet_head(cfg, in_channels):
    return CenternetHead(cfg, in_channels)

