import torch.nn as nn
import torchvision.models.resnet as resnet
from collections import OrderedDict

from detectron2.modeling.backbone.backbone import Backbone
from detectron2.modeling.backbone import BACKBONE_REGISTRY

_resnet_mapper = {
    18: resnet.resnet18,
    50: resnet.resnet50,
    101: resnet.resnet101,
}


@BACKBONE_REGISTRY.register()
class ResnetBackbone(Backbone):
    def __init__(self, cfg):
        super().__init__()
        depth = cfg.MODEL.RESNETS.DEPTH
        pretrained = cfg.MODEL.RESNETS.PRETRAINED
        backbone = _resnet_mapper[depth](pretrained=pretrained)
        # self.stage0 = nn.Sequential(
        #     backbone.conv1,
        #     backbone.bn1,
        #     backbone.relu,
        #     backbone.maxpool
        # )
        self.stage0 = nn.Sequential(OrderedDict([
            ('conv1',backbone.conv1),
            ('bn1', backbone.bn1),
            ('act1', backbone.relu),
            ('maxpool1', backbone.maxpool)
        ]))
        self.stage1 = backbone.layer1
        self.stage2 = backbone.layer2
        self.stage3 = backbone.layer3
        self.stage4 = backbone.layer4

    def forward(self, x):
        x = self.stage0(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        return x
