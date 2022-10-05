import torch, logging
import torch.nn.functional as F
from torch import nn
from fvcore.nn import sigmoid_focal_loss_jit



class MultiLossNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.auto_weight_flag = cfg.MODEL.GRAPHSPARSETRACK.AUTO_WEIGHT_FLAG
        nce_config_flag = hasattr(cfg.MODEL.TRACKER.GNN.NODE_MODEL.CLASSIFY, 'NCE')
        node_cls_config_flag = hasattr(cfg.MODEL.TRACKER.GNN.NODE_MODEL.CLASSIFY, 'CLS')
        self.id_loss_flag = cfg.MODEL.CENTERNET.ID_LOSS
        if self.auto_weight_flag:
            s_det = cfg.MODEL.GRAPHSPARSETRACK.AUTO_WEIGHT.S_DET
            s_cls = cfg.MODEL.GRAPHSPARSETRACK.AUTO_WEIGHT.S_MATCH
            s_nce = cfg.MODEL.GRAPHSPARSETRACK.AUTO_WEIGHT.S_NCE
            self.s_det = nn.Parameter(torch.tensor([s_det]))
            self.s_cls = nn.Parameter(torch.tensor([s_cls]))
            self.s_nce = nn.Parameter(torch.tensor([s_nce]))
        else:
            self.s_id = cfg.MODEL.LOSS.ID_WEIGHT if self.id_loss_flag else 0.0
            self.s_det = cfg.MODEL.GRAPHSPARSETRACK.DET_WEIGHT
            self.s_cls = cfg.MODEL.TRACKER.GNN.EDGE_MODEL.CLASSIFY.LOSS_WEIGHT
            if nce_config_flag:
                self.s_nce = cfg.MODEL.TRACKER.GNN.NODE_MODEL.CLASSIFY.NCE.LOSS_WEIGHT
            if node_cls_config_flag:
                self.s_node_cls = cfg.MODEL.TRACKER.GNN.NODE_MODEL.CLASSIFY.CLS.LOSS_WEIGHT

    def forward(self, det_loss_dict, trk_loss_dict):
        det_loss = det_loss_dict['det_loss']
        match_loss = trk_loss_dict['edge__loss_match']
        node_cls_flag = 'node__loss_cls' in trk_loss_dict
        nce_flag = 'edge__loss_nce' in trk_loss_dict
        losses_dict = {}
        if self.auto_weight_flag:
            ## TODO. need to modify - add reid loss
            total_loss = torch.exp(-self.s_det) * det_loss + torch.exp(-self.s_cls) * match_loss + self.s_det + self.s_cls
            losses_dict.update({'s_det': self.s_det, 's_cls': self.s_cls})
            if nce_flag:
                total_loss += torch.exp(-self.s_nce) * trk_loss_dict['edge__loss_nce'] + self.s_nce
                losses_dict.update({'s_nce': self.s_nce})
        else:
            total_loss = det_loss * self.s_det + match_loss * self.s_cls
            if self.id_loss_flag:
                total_loss += det_loss_dict['id_loss'] * self.s_id
            if node_cls_flag:
                total_loss += trk_loss_dict['node__loss_cls'] * self.s_node_cls
            if nce_flag:
                total_loss += trk_loss_dict['edge__loss_nce'] * self.s_nce
        losses_dict.update({'det_loss': det_loss, 'edge__loss_match': match_loss, 'total_loss': total_loss})
        if self.id_loss_flag:
            losses_dict.update({'id_loss': det_loss_dict['id_loss']})
        if node_cls_flag:
            losses_dict.update({'node__loss_cls': trk_loss_dict['node__loss_cls']})
        if nce_flag:
            losses_dict.update({'edge__loss_nce': trk_loss_dict['edge__loss_nce']})
        return losses_dict

class EdgeClsLoss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.focal_loss_alpha = cfg.FOCAL_ALPHA
        self.focal_loss_gamma = cfg.FOCAL_GAMMA
        self.logger = logging.getLogger(__name__)

    def forward(self, src_logits, labels, num_connection):
        edge_cls_loss = sigmoid_focal_loss_jit(
            src_logits,
            labels,
            alpha=self.focal_loss_alpha,
            gamma=self.focal_loss_gamma,
            reduction="sum",
        )
        if torch.isnan(edge_cls_loss):
            self.logger.info("NAN loss at Edge classification")
            return torch.tensor(0.0, device=edge_cls_loss.device)
        if labels.sum() == 0:  # for training stability
            return torch.tensor(0.0, device=edge_cls_loss.device)
        edge_cls_loss = edge_cls_loss / max(1.0, labels.sum()) # divide by positive connection; +1 for the case of no positive connection
        return edge_cls_loss

class NodeClsLoss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.focal_loss_alpha = cfg.FOCAL_ALPHA
        self.focal_loss_gamma = cfg.FOCAL_GAMMA
        self.logger = logging.getLogger(__name__)

    def forward(self, pred_logits, gt_labels):
        node_cls_loss = sigmoid_focal_loss_jit(
            pred_logits,
            gt_labels,
            alpha=self.focal_loss_alpha,
            gamma=self.focal_loss_gamma,
            reduction="sum",
        )
        if torch.isnan(node_cls_loss):
            self.logger.info("NAN loss at Node classification")
            return torch.tensor(0.0, device=node_cls_loss.device)
        if gt_labels.sum() == 0:  # for training stability
            return torch.tensor(0.0, device=node_cls_loss.device)
        return node_cls_loss / max(1.0, gt_labels.sum())