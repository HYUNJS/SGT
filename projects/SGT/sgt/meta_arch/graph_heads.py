import torch
import numpy as np
from torch import nn
import torch.nn.functional as F

from projects.SGT.sgt.meta_arch.loss import EdgeClsLoss, NodeClsLoss
from projects.SGT.sgt.meta_arch.graph_net import build_layers


class EdgeClsHead(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        edge_dim = cfg.IN_DIM
        edge_fc_dims = cfg.FC_DIMS
        edge_dropout_p = cfg.DROPOUT_P
        norm_layer = nn.LayerNorm
        act_func = nn.ReLU(inplace=True)
        edge_predictor = nn.Sequential(
                        *build_layers(edge_dim, edge_fc_dims, edge_dropout_p, norm_layer, act_func),
                        nn.Linear(edge_fc_dims[-1], 1))
        self.inf_thresh = cfg.INF_THRESH
        self.edge_predictor = edge_predictor
        self.match_loss = EdgeClsLoss(cfg)
        self.edge_out_dim = edge_fc_dims[-1]
        self.edge_feat_dim = edge_dim

    def forward(self, edge_dict, mems, new_mems):
        pred_logits, edge_idxs, mask_dict = self.forward_edge_cls(edge_dict, mems, new_mems)
        match_loss, gt_labels = self.loss_edge_cls(pred_logits, edge_idxs, mask_dict['mask_pred'], mems, new_mems)
        edge_info = {'edge_idxs': edge_idxs, 'gt_labels': gt_labels, 'mask_dict': mask_dict}
        return match_loss, edge_info

    @torch.no_grad()
    def inference(self, edge_dict):
        edge_features, edge_idxs = self.avg_edge_features(edge_dict)
        pred_logits = self.edge_predictor(edge_features[0])
        pred_probs = pred_logits.sigmoid()
        return pred_probs, edge_idxs

    def avg_edge_features(self, edge_dict):
        edge_dim = self.edge_feat_dim
        edge_features_list, edge_idxs = edge_dict['features'], edge_dict['index']
        batch_mask_flag = edge_dict.get('batch_masks', None) is not None
        num_deep_loss = len(edge_features_list)
        ## suppose batch masks is not None
        bs = len(edge_dict['batch_masks'])
        if batch_mask_flag:  ## suppose different number of edges in each batch
            edge_features_avg_list, edge_idxs_avg_list = [], []
            for b_idx in range(bs):
                batch_edge_mask = edge_dict['batch_masks'][b_idx]
                for i in range(num_deep_loss):
                    edge_features_avg = edge_features_list[i][batch_edge_mask].reshape(2, -1, edge_dim).mean(dim=0).reshape(-1, edge_dim)
                    edge_features_avg_list.append(edge_features_avg)
                edge_idxs_avg = edge_idxs[:, batch_edge_mask].chunk(2, dim=1)[0].permute(1, 0)
                edge_idxs_avg_list.append(edge_idxs_avg)
            edge_features_list = [torch.cat(edge_features_avg_list[i::num_deep_loss]) for i in range(num_deep_loss)]
            edge_idxs = torch.cat(edge_idxs_avg_list)
        else:  ## suppose same number of edges in each batch
            raise Exception("No more supporting w/o batch mask")

        return edge_features_list, edge_idxs


    def forward_edge_cls(self, edge_dict, mems, new_mems):
        edge_features_list, edge_idxs = self.avg_edge_features(edge_dict)
        bs = len(mems['box_nums'])

        ## GT mask
        t1_gt_ids, t2_gt_ids = mems['tensor']['gt_ids'], new_mems['tensor']['gt_ids']
        t1_box_nums, t2_box_nums = mems['box_nums'], new_mems['box_nums']
        t1_idx, t2_idx = 0, 0
        t1_gt_node_idx_list, t2_gt_node_idx_list = [], []
        for b_idx in range(bs):
            t1_box_num, t2_box_num = t1_box_nums[b_idx], t2_box_nums[b_idx]
            t1_gt_node_idx = torch.nonzero(t1_gt_ids[t1_idx:t1_idx+t1_box_num])
            t2_gt_node_idx = torch.nonzero(t2_gt_ids[t2_idx:t2_idx+t2_box_num])
            t2_gt_node_idx_list.append(t2_gt_node_idx + t1_idx + t2_idx)
            t1_gt_node_idx_list.append(t1_gt_node_idx + t1_idx + t2_idx + t2_box_num)
            t1_idx += t1_box_num
            t2_idx += t2_box_num
        t1_gt_node_idx = torch.cat(t1_gt_node_idx_list).permute(1, 0)
        t2_gt_node_idx = torch.cat(t2_gt_node_idx_list).permute(1, 0)
        gt_t2_node_mask = edge_idxs[:, 0].unsqueeze(1) == t2_gt_node_idx
        gt_t1_node_mask = edge_idxs[:, 1].unsqueeze(1) == t1_gt_node_idx
        gt_t2_node_mask = gt_t2_node_mask.any(dim=1)
        gt_t1_node_mask = gt_t1_node_mask.any(dim=1)
        mask_gt = torch.logical_and(gt_t2_node_mask, gt_t1_node_mask) ## TODO. Check isn't too much?
        mask_pred = torch.logical_or(gt_t2_node_mask, gt_t1_node_mask)

        num_deep_loss = len(edge_features_list)
        pred_logits = []
        for i in range(num_deep_loss):
            edge_feats = edge_features_list[i][mask_pred]
            pred_logits.append(self.edge_predictor(edge_feats))
        mask_dict = {'mask_pred': mask_pred, 'mask_all_gt': mask_gt, 'mask_trk_gt': gt_t1_node_mask, 'mask_det_gt':gt_t2_node_mask} ## TODO. Check is mask_gt necessary in this funciton?
        return pred_logits, edge_idxs, mask_dict

    def loss_edge_cls(self, pred_logits, edge_idxs, mask_pred, mems, new_mems):
        ## Find GT labels for each edge
        num_edge = edge_idxs.size(0)  # bs * num_prop * k
        gt_labels = torch.zeros(num_edge, device=edge_idxs.device)

        ## connection curr_frame-prev_frame
        t1_gt_ids, t2_gt_ids = mems['tensor']['gt_ids'].clone(), new_mems['tensor']['gt_ids'].clone()
        t1_box_nums, t2_box_nums = mems['box_nums'], new_mems['box_nums']
        bs = len(t1_box_nums)
        t1_gt_node_idx_list, t2_gt_node_idx_list = [], []
        t1_idx, t2_idx = 0, 0
        for b_idx in range(bs):
            t1_box_num, t2_box_num = t1_box_nums[b_idx], t2_box_nums[b_idx]
            t1_gt_id = t1_gt_ids[t1_idx:t1_idx+t1_box_nums[b_idx]]
            t2_gt_id = t2_gt_ids[t2_idx:t2_idx+t2_box_nums[b_idx]]

            ## find t1-t2 ID matching node index pairs
            t2_gt_id[t2_gt_id == 0] = -2 # fill t2's zero ID by -2 to avoid matching with t1's zero ID
            t1_gt_id[t1_gt_id == 0] = -1
            t2_gt_node_idx, t1_gt_node_idx = torch.nonzero(t2_gt_id.unsqueeze(1) == t1_gt_id.unsqueeze(0), as_tuple=True)

            ## add node offset
            t2_gt_node_idx += t1_idx + t2_idx
            t1_gt_node_idx += t1_idx + t2_idx + t2_box_num
            t1_gt_node_idx_list.append(t1_gt_node_idx)
            t2_gt_node_idx_list.append(t2_gt_node_idx)

            ## save t1 and t2 node index offset
            t1_idx += t1_box_nums[b_idx]
            t2_idx += t2_box_nums[b_idx]

        t1_gt_node_idx_list = torch.cat(t1_gt_node_idx_list, dim=0)
        t2_gt_node_idx_list = torch.cat(t2_gt_node_idx_list, dim=0)

        t2_gt_edge_mask = edge_idxs[:, 0].unsqueeze(1) == t2_gt_node_idx_list.unsqueeze(0)
        t1_gt_edge_mask = edge_idxs[:, 1].unsqueeze(1) == t1_gt_node_idx_list.unsqueeze(0)
        gt_label_mask = torch.logical_and(t2_gt_edge_mask, t1_gt_edge_mask)
        gt_label_idx = torch.nonzero(gt_label_mask.any(dim=1)).squeeze(1)
        gt_labels[gt_label_idx] = 1

        masked_gt_labels = gt_labels
        if mask_pred is not None:
            masked_gt_labels = gt_labels[mask_pred]
        num_conn = masked_gt_labels.size(0)

        ## compute focal loss
        masked_gt_labels = masked_gt_labels.unsqueeze(1)
        masked_gt_labels = masked_gt_labels.repeat(len(pred_logits), 1)
        pred_logits = torch.cat(pred_logits, dim=0)
        match_loss = self.match_loss(pred_logits, masked_gt_labels, num_conn)
        return match_loss, gt_labels

class NodeClsHead(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        in_dim = cfg.UPDATE.IN_DIM
        fc_dims = cfg.CLASSIFY.FC_DIMS
        dropout_p = cfg.CLASSIFY.DROPOUT_P
        norm_layer = nn.LayerNorm
        act_func = nn.ReLU(inplace=True)
        self.inf_thresh = cfg.CLASSIFY.INF_THRESH
        self.node_cls_net = nn.Sequential(
                                *build_layers(in_dim, fc_dims, dropout_p, norm_layer, act_func),
                                nn.Linear(fc_dims[-1], 1))
        self.node_cls_loss = NodeClsLoss(cfg.CLASSIFY)

    def forward(self, node_feats_list, gt_ids):
        '''
        Get input of each time's (T1 or T2) node feature and assigned gt ids
        :param node_feat: List[Tensor(#img * #topk_det, #feat_dim)] - len(List) = #deep supervision
        :param gt_ids: Tensor(#img, #topk_det)
        :return: classification loss of given time's node detection classification loss
        '''
        gt_labels = (gt_ids != 0).reshape(-1, 1).type(torch.float)
        pred_logits = self.node_cls_net(node_feats_list[-1])
        node_cls_loss = self.node_cls_loss(pred_logits, gt_labels)
        return node_cls_loss

    def inference(self, node_feats):
        pred_logits = self.node_cls_net(node_feats).sigmoid().squeeze(1)
        pos_node_mask = pred_logits >= self.inf_thresh
        return pos_node_mask
