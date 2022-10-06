import torch, logging
import torch.nn.functional as F
from torch import nn
from fvcore.nn import sigmoid_focal_loss_jit


class MultiLossNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.node_cls_flag = cfg.MODEL.TRACKER.GNN.NODE_MODEL.CLASSIFY.FLAG
        self.s_det = 1.0
        self.s_edge_cls = cfg.MODEL.TRACKER.GNN.EDGE_MODEL.CLASSIFY.LOSS_WEIGHT
        self.s_node_cls = cfg.MODEL.TRACKER.GNN.NODE_MODEL.CLASSIFY.LOSS_WEIGHT if self.node_cls_flag else 0.0

    def forward(self, det_loss_dict, trk_loss_dict):
        det_loss = det_loss_dict['det_loss']
        match_loss = trk_loss_dict['edge__loss_match']
        losses_dict = {}

        total_loss = det_loss * self.s_det + match_loss * self.s_edge_cls
        if self.node_cls_flag:
            total_loss += trk_loss_dict['node__loss_cls'] * self.s_node_cls

        losses_dict.update({'det_loss': det_loss, 'edge__loss_match': match_loss, 'total_loss': total_loss})
        if self.node_cls_flag:
            losses_dict.update({'node__loss_cls': trk_loss_dict['node__loss_cls']})

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
        node_cls_loss = node_cls_loss / max(1.0, gt_labels.sum())

        return node_cls_loss / max(1.0, gt_labels.sum())
