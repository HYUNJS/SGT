import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .generator.centernet_decode import gather_feature
from .losses import FocalLoss, RegL1Loss

def _sigmoid(x):
  y = torch.clamp(x.sigmoid_(), min=1e-4, max=1-1e-4)
  return y

class CenterNetLoss(torch.nn.Module):
    def __init__(self, cfg):
        super(CenterNetLoss, self).__init__()
        self.crit = FocalLoss()
        self.crit_reg = RegL1Loss()
        self.cfg = cfg
        self.hm_weight = cfg.MODEL.CENTERNET.HM_WEIGHT
        self.wh_weight = cfg.MODEL.CENTERNET.WH_WEIGHT
        self.off_weight = cfg.MODEL.CENTERNET.OFF_WEIGHT
        self.id_weight = cfg.MODEL.CENTERNET.ID_WEIGHT
        self.identity_on = cfg.MODEL.IDENTITY_ON and cfg.DATALOADER.NUM_IDS > 0 and cfg.MODEL.CENTERNET.ID_LOSS
        self.auto_weight_flag = False
        if hasattr(cfg.MODEL, 'LOSS'):
            self.auto_weight_flag = cfg.MODEL.LOSS.AUTO_WEIGHT_FLAG
            self.det_weight = cfg.MODEL.LOSS.DET_WEIGHT
            self.id_weight = cfg.MODEL.LOSS.ID_WEIGHT

        if self.identity_on:
            self.num_ids = cfg.DATALOADER.NUM_IDS
            self.emb_scale = math.sqrt(2) * math.log(self.num_ids - 1)
            self.reid_dim = cfg.MODEL.CENTERNET.HEAD.REID_DIM
            self.classifier = nn.Linear(self.reid_dim, self.num_ids)
            self.IDLoss = nn.CrossEntropyLoss(ignore_index=-1)
            if self.auto_weight_flag:
                self.s_det = nn.Parameter(-1.85 * torch.ones(1))
                self.s_id = nn.Parameter(-1.05 * torch.ones(1))
            else:
                self.s_det = nn.Parameter(self.det_weight * torch.ones(1), requires_grad=False)
                self.s_id = nn.Parameter(self.id_weight * torch.ones(1), requires_grad=False)

    def forward(self, outputs, batch):
        cur_device = outputs['hm'].device
        for k in batch:
            batch[k] = batch[k].to(cur_device)

        hm_outputs = outputs['hm'].clone()
        # hm_outputs = torch.clamp(hm_outputs, min=1e-4, max=1 - 1e-4)
        hm_outputs = _sigmoid(hm_outputs)

        det_loss, id_loss, weight_loss = 0, 0, 0
        hm_loss = self.crit(hm_outputs, batch['hm'])
        wh_loss = self.crit_reg(outputs['wh'], batch['reg_mask'], batch['ind'], batch['wh'])
        off_loss = self.crit_reg(outputs['reg'], batch['reg_mask'], batch['ind'], batch['reg'])
        det_loss = self.hm_weight * hm_loss + self.wh_weight * wh_loss + self.off_weight * off_loss
        losses_dict = {'hm_loss': hm_loss.detach(), 'wh_loss': wh_loss.detach(), 'off_loss': off_loss.detach()}
        losses_dict.update({'det_loss': det_loss})
        if self.identity_on:
            pred = gather_feature(outputs['id'], batch['ind'], use_transform=True)
            pred = pred[batch['reg_mask'] > 0].contiguous()
            pred = self.emb_scale * F.normalize(pred)
            id_target = batch['id'][batch['reg_mask'] > 0]
            id_output = self.classifier(pred).contiguous()
            id_loss = self.IDLoss(id_output, id_target)
            losses_dict.update({'id_loss': id_loss})
            if self.auto_weight_flag:
                weight_loss = self.s_det + self.s_id
                losses_dict.update({'s_det_loss': self.s_det.detach(), 's_id_loss': self.s_id.detach()})

        if self.identity_on:
            if self.auto_weight_flag:
                det_loss = torch.exp(-self.s_det) * det_loss * 0.5
                id_loss = torch.exp(-self.s_id) * self.id_weight * id_loss * 0.5
                weight_loss = weight_loss * 0.5
            else:
                det_loss = det_loss * self.s_det
                id_loss = id_loss * self.s_id
        total_loss = det_loss + id_loss + weight_loss
        losses_dict.update({'total_loss': total_loss})

        return losses_dict