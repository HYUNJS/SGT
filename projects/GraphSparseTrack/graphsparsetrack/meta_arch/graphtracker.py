import torch, os, logging, shutil
import numpy as np
from scipy.optimize import linear_sum_assignment
from torch import nn
import torch.nn.functional as F
import torchvision.ops as ops

from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.structures import Boxes, Instances
from projects.FairMOT.fairmot.tracker import matching
from projects.GraphSparseTrack.graphsparsetrack.meta_arch.graph_net import CrossFrameInteractionGNN
from projects.GraphSparseTrack.graphsparsetrack.utils import *
from projects.GraphSparseTrack.graphsparsetrack.tracker.tracklet_manager import TrackletManager
from projects.GraphSparseTrack.graphsparsetrack.meta_arch.graph_heads import NodeClsHead, NodeNCEHead, EdgeClsHead
from projects.GraphSparseTrack.graphsparsetrack.vis_utils import vis_input, vis_topk_input_pairs
from projects.GraphSparseTrack.graphsparsetrack.meta_arch.graphtracker_centernet import GST_CenterNet_Decoder

__all__ = ["GraphTracker"]


class NodeCNN(nn.Module):
    def __init__(self, cfg, backbone_out_feat_dim):
        super().__init__()

        in_dim = backbone_out_feat_dim
        inter_dims = cfg.MODEL.TRACKER.NODE_PRE_ENCODE.FC_DIMS
        out_dim = cfg.MODEL.TRACKER.GNN.NODE_MODEL.UPDATE.IN_DIM

        self.feat_conv = nn.Conv2d(in_dim, inter_dims[0], kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.out_conv = nn.Conv2d(inter_dims[0], out_dim, kernel_size=1)
        self.out_conv.bias.data.fill_(0)

    def forward(self, x):
        x = self.feat_conv(x)
        x = self.relu(x)
        x = self.out_conv(x)
        return x

@META_ARCH_REGISTRY.register()
class GraphTracker(nn.Module):
    def __init__(self, cfg, backbone_out_feat_dim):
        super().__init__()
        self.logger = logging.getLogger(__name__)

        self.cfg = cfg
        self.gnn = CrossFrameInteractionGNN(cfg)
        self.edge_cls_head = EdgeClsHead(cfg.MODEL.TRACKER.GNN.EDGE_MODEL.CLASSIFY)
        self.detection_decoder = GST_CenterNet_Decoder(cfg) # only support CenterNet

        ## top-K for previous frame
        self.prev_det_proposal_flag = cfg.MODEL.TRACKER.GNN.GRAPH.PREV_DET_PROPOSAL_FLAG

        ## storing tracklets
        self.history_manager = TrackletManager(cfg)

        ## smoothing feature
        self.SF_flag = cfg.MODEL.TRACKER.SMOOTH_FEAT.FLAG
        self.SF_alpha = cfg.MODEL.TRACKER.SMOOTH_FEAT.ALPHA
        self.SF_weight_by_score_flag = cfg.MODEL.TRACKER.SMOOTH_FEAT.WEIGHT_BY_SCORE

        ## second stage matching using IoU
        self.second_match_flag = cfg.MODEL.TRACKER.SECOND_MATCH.FLAG
        self.second_match_iou_thresh = cfg.MODEL.TRACKER.SECOND_MATCH.IOU_THRESH
        self.second_match_recovery = cfg.MODEL.TRACKER.SECOND_MATCH.RECOVERY_FLAG


        ## detection threshold values
        self.topk_det_flag = cfg.MODEL.TRACKER.GNN.TOPK_DET_FLAG
        self.topk_det = cfg.MODEL.TRACKER.GNN.TOPK_DET
        self.det_low_thresh = cfg.MODEL.TRACKER.GNN.DET_LOW_THRESH
        self.init_det_thresh = cfg.MODEL.TRACKER.INIT_DET_THRESH
        self.inference_gt = cfg.MODEL.GRAPHSPARSETRACK.INFERENCE_GT

        ## detection feature extractor config
        self.reid_flag = cfg.MODEL.CENTERNET.ID_LOSS

        ## detach gradient flag
        self.cut_det_grad_flag = cfg.MODEL.TRACKER.CUT_DET_GRAD
        self.cut_id_grad_flag = cfg.MODEL.TRACKER.CUT_ID_GRAD

        ## additional node module
        self.node_nce_loss_flag = cfg.MODEL.TRACKER.GNN.NODE_MODEL.CLASSIFY.NCE.FLAG
        self.node_cls_loss_flag = cfg.MODEL.TRACKER.GNN.NODE_MODEL.CLASSIFY.CLS.FLAG
        self.node_cnn = NodeCNN(cfg, backbone_out_feat_dim) if cfg.MODEL.TRACKER.NODE_PRE_ENCODE.FLAG else None
        self.node_nce_head = NodeNCEHead(cfg) if self.node_nce_loss_flag else None
        self.node_cls_head = NodeClsHead(cfg.MODEL.TRACKER.GNN.NODE_MODEL) if self.node_cls_loss_flag else None

        ## helper class variables
        self.id_marker = 0 # record the last id recorded

    @property
    def device(self):
        return next(self.parameters()).device

    def compute_loss(self, new_t1_feats_list, new_t2_feats_list, t1_info, t2_info, gt_ids_t2, edge_dicts):
        loss_dict = {}

        ## Run node cls module (further classification of detection)
        if self.node_cls_loss_flag:
            # node_cls_loss_t1 = self.node_cls_head(new_t1_feats_list, gt_ids_t1)
            node_cls_loss_t2 = self.node_cls_head(new_t2_feats_list, gt_ids_t2)
            node_cls_loss = node_cls_loss_t2
            # node_cls_loss = node_cls_loss_t1 + node_cls_loss_t2
            loss_dict.update({'node__loss_cls': node_cls_loss})

        ## Run edge cls module
        match_loss, edge_cls_metrics, edge_info = self.edge_cls_head(edge_dicts, t1_info, t2_info)
        loss_dict.update({'edge__loss_match': match_loss})

        ## Run node cls module
        if self.node_nce_loss_flag:
            nce_loss = self.node_nce_head(new_t1_feats_list, new_t2_feats_list, edge_dicts, edge_info)
            loss_dict.update({'edge__loss_nce': nce_loss})
        return loss_dict, edge_cls_metrics

    def cut_det_grad(self, detector_outputs):
        new_detector_outputs = {}
        for k in detector_outputs.keys():
            new_detector_outputs[k] = detector_outputs[k]
            if k in ['id', 'fmap']:
                if self.cut_id_grad_flag:
                    new_detector_outputs[k] = detector_outputs[k].detach()
            elif k in ['hm', 'wh', 'reg']:
                if self.cut_det_grad_flag:
                    new_detector_outputs[k] = detector_outputs[k].detach()
        return new_detector_outputs

    def forward(self, detector_outs, batched_inputs=None):
        # vis_input(batched_inputs[0]) # visualize input training sample
        node_featmap = self.node_cnn(detector_outs['feat_map']) if self.node_cnn is not None else detector_outs['feat_map']
        detector_outs['outputs']['node'] = node_featmap
        det_outs = self.cut_det_grad(detector_outs['outputs'])
        det_outs['inputs'] = batched_inputs

        ## assign GT or pseudo labels
        decoded_detections, gt_ids, detection_batch_mask_list = self.detection_decoder.decode_training(det_outs, detector_outs)

        ## masking out pair of t1 and t2 if either t1 or t2 has no detections above threshold
        t1_batch_mask_list, t2_batch_mask_list = detection_batch_mask_list[::2], detection_batch_mask_list[1::2]
        t1_box_nums, t2_box_nums = [b.sum().item() for b in t1_batch_mask_list], [b.sum().item() for b in t2_batch_mask_list]
        t1_mask, t2_mask = (decoded_detections['batch_idx'] % 2) == 0, (decoded_detections['batch_idx'] % 2) == 1

        prev_idx = 0
        t1_box_nums_new, t2_box_nums_new = [], []
        for t1_box_num, t2_box_num in zip(t1_box_nums, t2_box_nums):
            if t1_box_num == 0 or t2_box_num == 0:
                t1_mask[prev_idx:prev_idx + t1_box_num] = False
                t2_mask[prev_idx + t1_box_num:prev_idx + t1_box_num + t2_box_num] = False
            else:
                t1_box_nums_new.append(t1_box_num)
                t2_box_nums_new.append(t2_box_num)
            prev_idx += t1_box_num + t2_box_num
        t1_box_nums, t2_box_nums = t1_box_nums_new, t2_box_nums_new

        ## if no boxes in all batches, return zero loss
        if sum(t1_box_nums) == 0 or sum(t2_box_nums) == 0:
            loss_dict = {}
            loss_dict['edge__loss_match'] = torch.tensor([0.], device=node_featmap.device)
            loss_dict['node__loss_cls'] = torch.tensor([0.], device=node_featmap.device)
            return loss_dict, None

        ## parsing each attribute of t1 and t2
        gt_ids_t1, gt_ids_t2 = gt_ids[t1_mask], gt_ids[t2_mask]
        scores_t1, scores_t2 = decoded_detections['scores'][t1_mask], decoded_detections['scores'][t2_mask]
        fmap_feats_t1, fmap_feats_t2 = decoded_detections['fmap_feat'][t1_mask], decoded_detections['fmap_feat'][t2_mask]
        node_feats_t1, node_feats_t2 = decoded_detections['node_feat'][t1_mask], decoded_detections['node_feat'][t2_mask]
        boxes_dict_t1, boxes_dict_t2 = masking_dict(decoded_detections['bboxes'], t1_mask, t2_mask)

        ## run GNN module
        t1_info = {'tensor': {'boxes_dict': boxes_dict_t1, 'fmap_feats': fmap_feats_t1, 'node_feats': node_feats_t1,
                              'scores': scores_t1, 'gt_ids': gt_ids_t1},
                   'box_nums': t1_box_nums}
        t2_info = {'tensor': {'boxes_dict': boxes_dict_t2, 'fmap_feats': fmap_feats_t2, 'node_feats': node_feats_t2,
                              'scores': scores_t2, 'gt_ids': gt_ids_t2},
                   'box_nums': t2_box_nums}
        new_t1_feats_list, new_t2_feats_list, edge_dicts = self.gnn(t1_info, t2_info)

        ## compute loss
        loss_dict, edge_cls_metrics = self.compute_loss(new_t1_feats_list, new_t2_feats_list, t1_info, t2_info, gt_ids_t2, edge_dicts)

        return loss_dict, edge_cls_metrics

    def inference(self, detector_outs, t1_info, batched_inputs):
        node_featmap = self.node_cnn(detector_outs['feat_map']) if self.node_cnn is not None else detector_outs['feat_map']
        detector_outs['outputs']['node'] = node_featmap

        decoded_detections = self.detection_decoder.decode_inference(batched_inputs, detector_outs)
        boxes_dict = decoded_detections['bboxes']
        scores = decoded_detections['scores'].unsqueeze(0)
        classes = decoded_detections['classes'].unsqueeze(0)
        fmap_feats = decoded_detections['fmap_feat']
        node_feats = decoded_detections['node_feat']
        pred_ids = self.assign_pred_ids(scores)
        pred_id_mask = (pred_ids != 0).squeeze()
        box_nums = fmap_feats.size(0)

        ## Run GNN
        t2_info = {'tensor': {'boxes_dict': boxes_dict, 'fmap_feats': fmap_feats, 'node_feats': node_feats,
                              'scores': scores, 'pred_ids': pred_ids},
                   'box_nums': [box_nums]}

        if 'img_meta' in detector_outs: # for debug
            t2_info['img_meta'] = detector_outs['img_meta']

        if t2_info['box_nums'][0] == 0:
            t2_info['tensor']['tids'] = torch.zeros_like(t2_info['tensor']['pred_ids'])
        elif len(t1_info.keys()) == 0 or t1_info['box_nums'][0] == 0:
            ## 1st frame or no detections at previous frame
            pred_ids[:, pred_id_mask] += self.id_marker
            t2_info['tensor']['tids'] = pred_ids
            self.id_marker += pred_id_mask.sum()

        else:
            ## run GNN to update node and edge features
            new_t1_feats, new_t2_feats, edge_dict = self.gnn(t1_info, t2_info, self.history_manager.tracklets)

            ## node classification branch
            self.run_node_cls(t2_info, new_t2_feats, pred_ids)

            ## edge classification branch
            edge_scores, edge_idxs = self.edge_cls_head.inference(edge_dict)

            ## match t1 and t2 nodes based on edge and node scores
            self.match_t1_t2(t1_info, t2_info, edge_scores, edge_idxs)

        self.history_manager.update_history(t1_info, t2_info)

        ## make result output format
        tlbr = t2_info['tensor']['boxes_dict']['tlbr']
        tids = t2_info['tensor']['tids']
        result = {'boxes': tlbr, 'scores': scores, 'classes': classes, 'tids': tids}

        if 'reid_feat' in decoded_detections:
            result.update({'reid_feat': decoded_detections['reid_feat']})
        if not self.prev_det_proposal_flag:
            mask = (t2_info['tensor']['tids'] != 0).squeeze(0)
            t2_info['box_nums'][0] = mask.sum().item()
            t2_info['tensor']['boxes_dict'] = mask_boxes_dict(t2_info['tensor']['boxes_dict'], mask)
            t2_info['tensor']['node_feats'] = t2_info['tensor']['node_feats'][mask]
            t2_info['tensor']['fmap_feats'] = t2_info['tensor']['fmap_feats'][mask]
            t2_info['tensor']['scores'] = t2_info['tensor']['scores'][:, mask]
            t2_info['tensor']['pred_ids'] = t2_info['tensor']['pred_ids'][:, mask]
            t2_info['tensor']['tids'] = t2_info['tensor']['tids'][:, mask]
            if 'pos_node_mask' in t2_info['tensor']:
                t2_info['tensor']['pos_node_mask'] = t2_info['tensor']['pos_node_mask'][mask]

        ## save memory - t2 now becomes t1
        update_dict(t2_info, t1_info)

        return result

    def run_node_cls(self, t2_info, new_t2_feats, pred_ids):
        if not self.node_cls_loss_flag:
            return

        pos_node_mask_t2, init_node_mask_t2 = self.node_cls_head.inference(new_t2_feats[-1])
        t2_info['tensor']['pos_node_mask'] = pos_node_mask_t2.reshape(-1)
        if init_node_mask_t2 is not None:
            pred_ids_by_node = torch.zeros_like(pred_ids)
            pred_ids_by_node[..., init_node_mask_t2] = torch.arange(1, init_node_mask_t2.sum() + 1, device=pred_ids.device)
            t2_info['tensor']['pred_ids'] = pred_ids_by_node

    def match_t1_t2(self, t1_info, t2_info, edge_scores, edge_idxs):
        curr_pred_ids, prev_pred_ids = t2_info['tensor']['pred_ids'], t1_info['tensor']['tids']
        curr_tlbr, prev_tlbr = t2_info['tensor']['boxes_dict']['tlbr'], t1_info['tensor']['boxes_dict']['tlbr']
        curr_feats, prev_feats = t2_info['tensor']['fmap_feats'], t1_info['tensor']['fmap_feats']
        curr_scores, prev_scores = t2_info['tensor']['scores'], t1_info['tensor']['scores']
        if self.history_manager.tracklets is not None:
            prev_pred_ids = torch.cat([prev_pred_ids, self.history_manager.tracklets['tids']], dim=1)
            prev_tlbr = torch.cat([prev_tlbr, self.history_manager.tracklets['boxes_dict']['tlbr']], dim=0)
            prev_feats = torch.cat([prev_feats, self.history_manager.tracklets['fmap_feats']], dim=0)
            prev_scores = torch.cat([prev_scores, self.history_manager.tracklets['scores']], dim=1)

        num_t2_det = len(t2_info['tensor']['fmap_feats'])
        mask_curr_pred_having_id = curr_pred_ids != 0
        mask_prev_pred_having_id = (prev_pred_ids != 0).squeeze()
        new_tids = torch.zeros_like(curr_pred_ids)
        mask_pos_edge = (edge_scores >= self.edge_cls_head.inf_thresh).squeeze(1)
        pos_edge_scores = edge_scores[mask_pos_edge]
        pos_edge_idxs = edge_idxs[mask_pos_edge]

        if pos_edge_scores.size(0) > 0:  ## Case 1. there are edges which score is above thresh
            matches, u_detection, u_unconfirmed, t1_matched_idx, t2_matched_idx, prev_pred_ids = self.match_by_edge_score(pos_edge_scores,
                               pos_edge_idxs, prev_pred_ids, curr_pred_ids, mask_prev_pred_having_id, new_tids, t1_info, t2_info)

            ## verify false positive recovery case
            if self.node_cls_loss_flag and len(matches) > 0:
                matches, t1_matched_idx, t2_matched_idx, u_detection, u_unconfirmed = self.verify_fp_recovery(matches, u_detection, u_unconfirmed,
                                                                            t1_matched_idx, t2_matched_idx, new_tids, curr_pred_ids, t2_info)

            ## second matching stage using iou
            if self.second_match_flag:
                iou_m = self.match_remain_t1_t2_by_iou(new_tids, prev_pred_ids, curr_pred_ids, prev_tlbr, curr_tlbr,
                                                mask_prev_pred_having_id, u_unconfirmed, u_detection)
                if iou_m is not None and len(iou_m) > 0:
                    t2_iou_m_idx = torch.tensor(iou_m[:, 0], device=self.device)
                    t1_iou_m_idx = torch.tensor(iou_m[:, 1], device=self.device)
                    if len(matches) > 0:
                        t1_matched_idx = torch.cat([t1_matched_idx, t1_iou_m_idx])
                        t2_matched_idx = torch.cat([t2_matched_idx, t2_iou_m_idx])
                        matches = np.concatenate([matches, iou_m])
                    else:
                        t1_matched_idx, t2_matched_idx = t1_iou_m_idx, t2_iou_m_idx
                        matches = iou_m

            ## update appearance features of tracklets as weighted sum of tracklet and its matched current detection
            if self.SF_flag and len(matches) > 0:
                prev_scores_having_id, prev_feats_having_id = prev_scores[:, mask_prev_pred_having_id], prev_feats[mask_prev_pred_having_id]
                updated_t2_fmap_feats = self.smoothing_features(matches, t1_matched_idx, t2_matched_idx, curr_scores, curr_feats,
                                                                prev_scores_having_id, prev_feats_having_id)
                t2_info['tensor']['fmap_feats'] = updated_t2_fmap_feats[0:num_t2_det]

            ## assign id to new tracklets
            mask_new_det = torch.logical_and(new_tids == 0, curr_pred_ids != 0)
            num_new_dets = mask_new_det.sum()
            if num_new_dets > 0:
                new_det_ids = torch.arange(self.id_marker + 1, self.id_marker + num_new_dets + 1, device=new_tids.device)
                new_tids[mask_new_det] = new_det_ids

            ## visualization for debug
            if 'img_meta' in t1_info:
                t1_info_vis = {'img_meta': t1_info['img_meta'], 'tlbr': prev_tlbr[mask_prev_pred_having_id], 'id': prev_pred_ids}
                t2_info_vis = {'img_meta': t2_info['img_meta'], 'tlbr': curr_tlbr, 'id': new_tids, 'mask': t2_info['tensor']['scores'] >= self.init_det_thresh}
                t1_info_vis.update({'unmatched': u_unconfirmed})
                t2_info_vis.update({'unmatched': u_detection})
                save_path = '/mnt/video_nfs4/jshyun/gst_output/vis_inf_by_pair/gst_dla34_17'
                # save_path = ''
                self.vis_pred_detections_by_pair(t1_info_vis, t2_info_vis, save_path)
        elif mask_curr_pred_having_id.sum() > 0:  ## Case 2. No edge > thresh & Yes detection > thresh
            new_tids[mask_curr_pred_having_id] = curr_pred_ids[mask_curr_pred_having_id] + self.id_marker
            num_new_dets = mask_curr_pred_having_id.sum()
        else:  ## Case 3. No edge > thresh & No detection > thresh
            t2_info['tensor']['tids'] = new_tids
            return

        self.id_marker += num_new_dets
        t2_info['tensor']['tids'] = new_tids

    def match_by_edge_score(self, pos_edge_scores, pos_edge_idxs, prev_pred_ids, curr_pred_ids, mask_prev_pred_having_id, new_tids, t1_info, t2_info):
        t1_matched_idx, t2_matched_idx = None, None

        ## Fill edge scores into matrix
        det_idxs = pos_edge_idxs[:, 0]
        trk_idxs = pos_edge_idxs[:, 1] - curr_pred_ids.size(1)
        edge_score_mat = torch.zeros(curr_pred_ids.size(1), prev_pred_ids.size(1))
        edge_score_mat[det_idxs, trk_idxs] = pos_edge_scores.squeeze(1).cpu()

        ## filtering out edge scores which t1 nodes do not have id
        edge_score_mat = edge_score_mat[:, mask_prev_pred_having_id]
        prev_pred_ids = prev_pred_ids[:, mask_prev_pred_having_id]
        cost_matrix = 1 - edge_score_mat

        matches, u_detection, u_unconfirmed = matching.linear_assignment(cost_matrix.cpu().numpy(), thresh=1 - self.edge_cls_head.inf_thresh)
        if len(matches) > 0:
            t2_matched_idx = torch.tensor(matches[:, 0], device=self.device)
            t1_matched_idx = torch.tensor(matches[:, 1], device=self.device)
            new_tids[:, t2_matched_idx] = prev_pred_ids[:, t1_matched_idx]
        return matches, u_detection, u_unconfirmed, t1_matched_idx, t2_matched_idx, prev_pred_ids

    def verify_fp_recovery(self, matches, u_detection, u_unconfirmed, t1_matched_idx, t2_matched_idx, new_tids, curr_pred_ids, t2_info):
        ## remove false positively recovered detections by node classification

        curr_recovered_mask = torch.logical_and(new_tids != 0, curr_pred_ids == 0).squeeze()
        invalid_recovered_mask = torch.logical_and(curr_recovered_mask, ~t2_info['tensor']['pos_node_mask'])
        new_tids[:, invalid_recovered_mask] = 0
        if invalid_recovered_mask.sum() > 0:
            invalid_recovered_idx = torch.nonzero(invalid_recovered_mask)  # [#invalid, 1]
            matched_valid_mask = ~(invalid_recovered_idx == t2_matched_idx.unsqueeze(0)).any(dim=0)

            ## append matches filtered out by node classifier
            invalid_matches = matches[~matched_valid_mask.cpu().numpy()]
            u_detection = np.concatenate([u_detection, invalid_matches[:, 0]])
            u_unconfirmed = np.concatenate([u_unconfirmed, invalid_matches[:, 1]])

            ## remove filtered out matches
            t2_matched_idx = t2_matched_idx[matched_valid_mask]
            t1_matched_idx = t1_matched_idx[matched_valid_mask]
            matches = matches[matched_valid_mask.cpu().numpy()]
        return matches, t1_matched_idx, t2_matched_idx, u_detection, u_unconfirmed

    def smoothing_features(self, matches, t1_matched_idx, t2_matched_idx, curr_scores, curr_feats, prev_scores, prev_feats):
        matched_curr_feats = curr_feats[t2_matched_idx]
        prev_feats = prev_feats[t1_matched_idx]
        if self.SF_weight_by_score_flag:
            curr_scores = curr_scores[:, t2_matched_idx]
            prev_scores = prev_scores[:, t1_matched_idx]
            sum_scores = prev_scores + curr_scores
            norm_prev_scores = prev_scores / sum_scores
            norm_curr_scores = curr_scores / sum_scores
            smoothed_curr_feats = matched_curr_feats * norm_curr_scores.reshape(-1, 1) + prev_feats * norm_prev_scores.reshape(-1, 1)
        else:
            smoothed_curr_feats = matched_curr_feats * self.SF_alpha + prev_feats * (1 - self.SF_alpha)
        smoothed_curr_feats = F.normalize(smoothed_curr_feats, dim=1)
        curr_feats[t2_matched_idx] = smoothed_curr_feats
        return curr_feats

    def match_remain_t1_t2_by_iou(self, new_tids, t1_pred_ids, t2_pred_ids, t1_tlbr, t2_tlbr, mask_prev_pred_having_id, u_t1_idx, u_t2_idx):
        if len(u_t1_idx) == 0:
            return None

        mask_new_det = torch.logical_and(new_tids == 0, t2_pred_ids != 0).squeeze()
        if not self.second_match_recovery:
            # use only t2 detections above threshold to avoid recovery
            if mask_new_det.sum() == 0:
                return None
            u_t2_idx = torch.nonzero(mask_new_det).squeeze(1).cpu().numpy()

        iou_cost_thresh = 1 - self.second_match_iou_thresh
        u_t1_tlbr = t1_tlbr[mask_prev_pred_having_id][u_t1_idx]
        u_t2_tlbr = t2_tlbr[u_t2_idx]
        iou_mat = ops.box_iou(u_t2_tlbr, u_t1_tlbr)  # [t1_box_num, t2_box_num]
        iou_cost = 1 - iou_mat
        iou_m, iou_u_t2, iou_u_t1 = matching.linear_assignment(iou_cost.cpu().numpy(), thresh=iou_cost_thresh)
        if len(iou_m > 0):
            new_tids[0, u_t2_idx[iou_m[:, 0]]] = t1_pred_ids[0, u_t1_idx[iou_m[:, 1]]]
            iou_m = np.stack([u_t2_idx[iou_m[:, 0]], u_t1_idx[iou_m[:, 1]]], axis=1)
        return iou_m

    @torch.no_grad()
    def assign_pred_ids(self, topk_scores):
        scores = topk_scores.reshape(topk_scores.shape[:-1])
        pred_ids = torch.zeros(scores.shape, device=scores.device, dtype=torch.long)
        pred_mask = scores >= self.init_det_thresh
        pred_query_idx = torch.nonzero(pred_mask, as_tuple=True)
        pred_tgt_ids = torch.arange(1, 1 + pred_mask.sum(), device=pred_mask.device)
        pred_ids[pred_query_idx] = pred_tgt_ids
        return pred_ids

    def reset_id_marker(self):
        self.id_marker = 0

    def reset(self, fps=30):
        self.reset_id_marker()
        self.history_manager.reset(fps)

    @staticmethod
    def vis_decoded_detections_by_pair(raw_detections, decoded_detections, gt_ids, remap_id=False, show_first_sample=True, save_path=''):
        ## TODO. FIX for flattened batch version
        gt_inputs = []
        for b_idx in range(len(raw_detections['inputs'])):
            img = raw_detections['inputs'][b_idx]['image']
            _, img_h, img_w = img.shape
            gt_output = dict(gt_boxes=decoded_detections['bboxes']['tlbr'][b_idx].cpu().numpy() * 4,
                             scores=decoded_detections['scores'][b_idx].cpu().numpy().reshape(-1),
                             gt_ids=gt_ids[b_idx].cpu().numpy().reshape(-1),
                             gt_classes=decoded_detections['classes'][b_idx].cpu().numpy().reshape(-1))
            gt_instances = Instances((int(img_h), int(img_w)), **gt_output)
            gt_inputs.append({'image': img,
                              'instances': gt_instances,
                              'sequence_name': raw_detections['inputs'][b_idx]['sequence_name'],
                              'file_name': raw_detections['inputs'][b_idx]['file_name']})
        if save_path != '':
            os.makedirs(save_path)
        if show_first_sample:
            vis_topk_input_pairs(gt_inputs[0:2], remap_id, save_path)
        else:
            vis_topk_input_pairs(gt_inputs, remap_id, save_path)

    @staticmethod
    def vis_pred_detections_by_pair(t1_info_vis, t2_info_vis, save_path=''):
        import matplotlib.pyplot as plt
        import cv2

        t1_img_name, t2_img_name = t1_info_vis['img_meta']['img_name'], t2_info_vis['img_meta']['img_name']
        t1_img_ori, t1_tlbr_ori, t1_id_ori = t1_info_vis['img_meta']['image'], t1_info_vis['tlbr'], t1_info_vis['id']
        t2_img_ori, t2_tlbr_ori, t2_id_ori = t2_info_vis['img_meta']['image'], t2_info_vis['tlbr'], t2_info_vis['id']
        t2_score_mask = t2_info_vis['mask'].reshape(-1)
        t1_unmatched_idx, t2_unmatched_idx = t1_info_vis['unmatched'], t2_info_vis['unmatched']

        t1_img, t2_img = np.ascontiguousarray(t1_img_ori), np.ascontiguousarray(t2_img_ori)
        t1_tlbr, t2_tlbr = t1_tlbr_ori.cpu().numpy(), t2_tlbr_ori.cpu().numpy()
        t1_id, t2_id = t1_id_ori.reshape(-1).cpu().numpy(), t2_id_ori.reshape(-1).cpu().numpy()

        H, W, _ = t1_img.shape
        t1_tlbr = (t1_tlbr * 4).astype(int)
        t2_tlbr = (t2_tlbr * 4).astype(int)
        t1_tlbr = np.clip(t1_tlbr, [0, 0, 0, 0], [W-1, H-1, W-1, H-1])
        t2_tlbr = np.clip(t2_tlbr, [0, 0, 0, 0], [W-1, H-1, W-1, H-1])

        font_face = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.9
        font_thickness = 2

        ## visualize t1 with bbox and id
        t1_unmatched_num, t1_matched_num = 0, 0
        for idx, t1_box in enumerate(t1_tlbr):
            if idx in t1_unmatched_idx:
                font_color = (1, 0, 0) # unmatched box id in RED
                t1_unmatched_num += 1
            else:
                font_color = (0, 1, 0) # matched box id in green
                t1_matched_num += 1
            t1_img = cv2.rectangle(t1_img, (t1_box[0], t1_box[1]), (t1_box[2], t1_box[3]), (0, 1, 0), 2)
            cv2.putText(t1_img, str(t1_id[idx]), (t1_box[0], t1_box[1]), font_face, font_scale, font_color, font_thickness)

        ## visualize t2 with bbox and id
        t2_new_num, t2_rec_num, t2_matched_num = 0, 0, 0
        for idx, t2_box in enumerate(t2_tlbr):
            if t2_id[idx] != 0:
                box_thickness = 2
                if t2_score_mask[idx]: # pred box with score higher than threshold
                    if idx in t2_unmatched_idx:
                        box_color = (0, 0, 1)
                        font_color = (0, 0, 1)
                        t2_new_num += 1
                    else:
                        box_color = (0, 1, 0)
                        font_color = (0, 1, 0)
                        t2_matched_num += 1
                else: # recovered detection which score lower than threshold
                    box_color = (1, 0, 0)
                    font_color = (1, 0, 0)
                    t2_rec_num += 1
            else: # low score unmatched boxes
                box_thickness = 1
                box_color = (1, 1, 1)
                # box_color = (0.5, 0.5, 0.5)

            t2_img = cv2.rectangle(t2_img, (t2_box[0], t2_box[1]), (t2_box[2], t2_box[3]), box_color, box_thickness)
            if t2_id[idx] != 0:
                cv2.putText(t2_img, str(t2_id[idx]), (t2_box[0], t2_box[1]), font_face, font_scale, font_color, font_thickness)

        plt.subplot(2, 1, 1)
        plt.text(-400, 50, 'T1 unmatched in red ID\nT2 recovered in red ID\nT2 new det in blue ID')

        plt.imshow(t1_img)
        plt.title(f'T1 - {t1_img_name}')
        plt.axis('off')
        plt.tight_layout(pad=0)

        plt.subplot(2, 1, 2)
        plt.text(-400, 50, f'T1 unmatched: {t1_unmatched_num}\nT1 matched: {t1_matched_num}\
                        \nT2 recovered: {t2_rec_num}\nT2 new: {t2_rec_num}\nT2 matched: {t2_matched_num}')
        plt.imshow(t2_img)
        plt.title(f'T2 - {t2_img_name}')
        plt.axis('off')
        plt.tight_layout(pad=0)

        if save_path != '':
            save_filename = os.path.join(save_path, t2_img_name + '.png')
            if os.path.exists(save_filename):
                os.remove(save_filename)
            plt.savefig(save_filename)
        else:
            plt.show()
        plt.clf()


def plot_edge_t1_t2(tgt_t1_idx, batched_inputs, detector_outs, edge_scores, edge_idxs, t1_info, t2_info, savename=None):
    import cv2
    import os
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpl_patches
    from matplotlib.collections import PatchCollection
    from projects.Datasets.Transforms.augmentation import CenterAffine
    from projects.Datasets.MOT.vis.colormap import id2color

    def transform_boxes(boxes, img_info, scale=1):
        '''
        transform predicted boxes to target boxes

        Args:
            boxes: Tensor
                torch Tensor with (Batch, N, 4) shape
            img_info: Dict
                dict contains all information of original image
            scale: float
                used for multiscale testing
        '''
        boxes = boxes.cpu().numpy().reshape(-1, 4)

        center = img_info['center']
        size = img_info['size']
        output_size = (img_info['width'], img_info['height'])
        src, dst = CenterAffine.generate_src_and_dst(center, size, output_size)
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))

        coords = boxes.reshape(-1, 2)
        aug_coords = np.column_stack((coords, np.ones(coords.shape[0])))
        target_boxes = np.dot(aug_coords, trans.T).reshape(-1, 4)
        return target_boxes

    curr_frame_idx = batched_inputs[0]['frame_idx']

    tgt_t1_idx_wo_offset = tgt_t1_idx - t2_info['box_nums'][0]
    tgt_edge_mask = edge_idxs[:, 1] == tgt_t1_idx
    t1_tlbrs = transform_boxes(t1_info['tensor']['boxes_dict']['tlbr'], detector_outs['img_info'])
    t2_tlbrs = transform_boxes(t2_info['tensor']['boxes_dict']['tlbr'], detector_outs['img_info'])
    tgt_t2_idxs = edge_idxs[tgt_edge_mask][:, 0].cpu()
    tgt_t1_tlbrs = t1_tlbrs[tgt_t1_idx_wo_offset]
    tgt_t2_tlbrs = t2_tlbrs[tgt_t2_idxs]
    tgt_t2_scores = t2_info['tensor']['scores'][0][tgt_t2_idxs]
    tgt_edge_scores = edge_scores[tgt_edge_mask]
    tgt_t1_tids = t1_info['tensor']['tids'][0, tgt_t1_idx - t2_info['box_nums'][0]]
    tgt_t2_tids = t2_info['tensor']['tids'][0, tgt_t2_idxs]

    img_wh = detector_outs['img_info']['center'] * 2
    curr_img_filename = batched_inputs[0]['file_name']
    prev_img_filename = curr_img_filename.replace('{}.jpg'.format(curr_frame_idx + 1), '{}.jpg'.format(curr_frame_idx))
    curr_img = cv2.imread(curr_img_filename)[:, :, ::-1]
    prev_img = cv2.imread(prev_img_filename)[:, :, ::-1]
    prev_patches, curr_patches = [], []
    patch_kwargs = {'color': 'red', 'linewidth': 1, 'fill': False}

    fig, axs = plt.subplots(2, 1)
    axs[0].imshow(prev_img)
    axs[1].imshow(curr_img)
    axs[0].axis('off')
    axs[1].axis('off')

    ## draw previous frame bbox
    patch_kwargs['color'] = 'blue' if tgt_t1_tids == 0 else 'green'
    prev_patches.append(mpl_patches.Rectangle((tgt_t1_tlbrs[0:2]), *(tgt_t1_tlbrs[2:] - tgt_t1_tlbrs[0:2]), **patch_kwargs))
    tgt_t1_cxcy = (tgt_t1_tlbrs[0:2] + tgt_t1_tlbrs[2:]) / 2
    text = "{}".format(tgt_t1_tids.item())
    text_color = bright_text_label(id2color(tgt_t1_tids.item(), rgb=True, maximum=1))
    text_kwargs = {"bbox": {"facecolor": "black", "alpha": 0.8, "pad": 0.7, "edgecolor": "none"}, "fontsize": 10.0, "verticalalignment": "top", "zorder": 10}
    axs[0].text(tgt_t1_tlbrs[0], tgt_t1_tlbrs[1], text, color=text_color, **text_kwargs)

    cs = mpl_patches.ConnectionStyle.Arc3(rad=-0.2)

    for tgt_t2_idx in range(len(tgt_t2_tlbrs)):
        tgt_t2_tlbr = tgt_t2_tlbrs[tgt_t2_idx]
        tgt_t2_score = tgt_t2_scores[tgt_t2_idx]
        tgt_t2_tid = tgt_t2_tids[tgt_t2_idx]
        tgt_edge_score = tgt_edge_scores[tgt_t2_idx]
        edge_color = 'r'
        text_color = bright_text_label(id2color(tgt_t2_tid.item(), rgb=True, maximum=1))

        patch_kwargs['color'] = 'red'
        if tgt_t2_score < 0.5:
            patch_kwargs['color'] = 'yellow'

        if tgt_t1_tids != 0 and tgt_t1_tids == tgt_t2_tids[tgt_t2_idx]:
            edge_color = 'g'
            # patch_kwargs['color'] = 'green'
            # if tgt_t2_score < 0.5:
            #     patch_kwargs['color'] = 'blue'

        tgt_t2_cxcy = (tgt_t2_tlbr[0:2] + tgt_t2_tlbr[2:4]) / 2
        tgt_t2_cx_in_img = tgt_t2_cxcy[0] >= 0 and tgt_t2_cxcy[0] <= img_wh[0]
        tgt_t2_cy_in_img = tgt_t2_cxcy[1] >= 0 and tgt_t2_cxcy[1] <= img_wh[1]
        if tgt_t2_cx_in_img and tgt_t2_cy_in_img:
            ## do not show objects whose center point out of image
            ## draw current detection box
            curr_patches.append(mpl_patches.Rectangle((tgt_t2_tlbr[0:2]), *(tgt_t2_tlbr[2:] - tgt_t2_tlbr[0:2]), **patch_kwargs))
            text = "{}-{:.2f}".format(tgt_t2_tid.item(), tgt_t2_score.item())
            axs[1].text(tgt_t1_tlbrs[0], tgt_t1_tlbrs[1], text, color=text_color, **text_kwargs)

            if tgt_edge_score > 0.0:
                con = mpl_patches.ConnectionPatch(
                    color=edge_color, alpha=max(tgt_edge_score.item(), 0.1),
                    xyA=(tgt_t1_cxcy[0], tgt_t1_cxcy[1]), coordsA=axs[0].transData,
                    xyB=(tgt_t2_cxcy[0], tgt_t2_cxcy[1]), coordsB=axs[1].transData,
                    connectionstyle=cs)
                axs[1].add_patch(con)

    axs[0].add_collection(PatchCollection(prev_patches, match_original=True))
    axs[1].add_collection(PatchCollection(curr_patches, match_original=True))

    fig.tight_layout(pad=1.0)
    if savename is not None:
        save_dir = os.path.join(savename, f"{batched_inputs[0]['sequence_name']}-#{curr_frame_idx + 1}")
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, f"{tgt_t1_tids.item()}.png"))
        plt.close()
    else:
        plt.show()
        plt.close()

def change_color_brightness(color, brightness_factor):
    '''
    code borrowed from detectron2/utils/visualizer
    '''
    import colorsys
    import matplotlib.colors as mplc

    assert brightness_factor >= -1.0 and brightness_factor <= 1.0
    color = mplc.to_rgb(color)
    polygon_color = colorsys.rgb_to_hls(*mplc.to_rgb(color))
    modified_lightness = polygon_color[1] + (brightness_factor * polygon_color[1])
    modified_lightness = 0.0 if modified_lightness < 0.0 else modified_lightness
    modified_lightness = 1.0 if modified_lightness > 1.0 else modified_lightness
    modified_color = colorsys.hls_to_rgb(polygon_color[0], modified_lightness, polygon_color[2])
    return modified_color

def bright_text_label(color):
    import matplotlib.colors as mplc

    color = change_color_brightness(color, brightness_factor=0.7)
    color = np.maximum(list(mplc.to_rgb(color)), 0.2)
    color[np.argmax(color)] = max(0.8, np.max(color))
    return color

# savename = "/mnt/video_nfs4/jshyun/gst_output/mot20_sgt56_edges/"
# for tgt_t1_idx in torch.nonzero(t1_info['tensor']['tids'].squeeze() != 0):
#     plot_edge_t1_t2(t2_info['box_nums'][0] + tgt_t1_idx, batched_inputs, detector_outs, edge_scores, edge_idxs, t1_info, t2_info, savename)