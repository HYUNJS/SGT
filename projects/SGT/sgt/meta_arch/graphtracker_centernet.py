import torch, os, logging, shutil
import numpy as np
from torch import nn
import torch.nn.functional as F
import torchvision.ops as ops

from projects.FairMOT.fairmot.tracker import matching
from projects.SGT.sgt.utils import *


class SGT_CenterNet_Decoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.logger = logging.getLogger(__name__)

        ## config for feature map size
        self.resolution_down_ratio = cfg.MODEL.CENTERNET.DOWN_RATIO
        self.hm_height, self.hm_width = cfg.MODEL.CENTERNET.HM_SIZE

        ## config for selecting tracking candidates
        self.topk_det_flag = cfg.MODEL.TRACKER.GNN.TOPK_DET_FLAG
        self.topk_det = cfg.MODEL.TRACKER.GNN.TOPK_DET
        self.det_low_thresh = cfg.MODEL.TRACKER.GNN.DET_LOW_THRESH
        self.inference_gt = cfg.MODEL.SGT.INFERENCE_GT

        ## config for label assignment
        self.train_by_gt = cfg.MODEL.TRACKER.LABEL_ASSIGNMENT.TRAIN_BY_GT
        self.label_assignment_metric = cfg.MODEL.TRACKER.LABEL_ASSIGNMENT.METRIC
        self.min_wh_ratio = cfg.MODEL.TRACKER.LABEL_ASSIGNMENT.MIN_WH_RATIO
        self.min_iou = cfg.MODEL.TRACKER.LABEL_ASSIGNMENT.MIN_IOU
        self.fill_gt_flag = cfg.MODEL.TRACKER.LABEL_ASSIGNMENT.FILL_GT_FLAG
        self.replace_unmatched_pred = cfg.MODEL.TRACKER.LABEL_ASSIGNMENT.REPLACE_POLICY

    def decode_training(self, det_outs, detector_outs):
        if self.train_by_gt:
            decoded_detections, gt_ids, detection_batch_mask_list = self.decode_gt_training(detector_outs['targets'], det_outs)
        else:
            decoded_detections = self.decode_predictions(det_outs)
            gt_ids, detection_batch_mask_list = self.assign_gt_ids(detector_outs['targets'], det_outs, decoded_detections)

        return decoded_detections, gt_ids, detection_batch_mask_list

    def decode_inference(self, batched_inputs, detector_outs):
        if self.inference_gt:
            decoded_detections = self.decode_gt_inference(detector_outs['outputs'], batched_inputs)
        else:
            decoded_detections = self.decode_predictions(detector_outs['outputs'])

        return decoded_detections

    def decode_predictions(self, det_outs):
        if self.topk_det_flag:
            decoded_detections = self.decode_topk(det_outs)
        else:
            decoded_detections = self.decode_thresh(det_outs)

        return decoded_detections

    def decode_gt_inference(self, det_outs, batched_inputs):
        device = det_outs['outputs']['fmap'].device
        tlbr = (batched_inputs[0]['instances'].gt_boxes.tensor / self.resolution_down_ratio).to(device)
        cxcywh = tlbr2cxcywh(tlbr)
        boxes_dict = {'xs': cxcywh[:, 0:1], 'ys': cxcywh[:, 1:2], 'wh': cxcywh[:, 2:4], 'tlbr': tlbr}
        scores = torch.ones(len(tlbr), 1, device=device)
        classes = torch.zeros(len(tlbr), 1, device=device, dtype=torch.long)

        ## indexing node/fmap/reid features
        x_idx = cxcywh[:, 0].clamp(min=0, max=self.hm_width - 1).type(torch.long)
        y_idx = cxcywh[:, 1].clamp(min=0, max=self.hm_height - 1).type(torch.long)
        node_feature = det_outs["node"]
        fmap = det_outs["fmap"]
        reid_feature = det_outs['id'] if 'id' in det_outs else None
        if node_feature is not None:
            node_feature = F.normalize(node_feature, dim=1)
            node_feature = node_feature.permute(0, 2, 3, 1)[0, y_idx, x_idx]
        if fmap is not None:
            fmap = F.normalize(fmap, dim=1)
            fmap = fmap.permute(0, 2, 3, 1)[0, y_idx, x_idx]
        if reid_feature is not None:
            reid_feature = F.normalize(reid_feature, dim=1)
            reid_feature = reid_feature.permute(0, 2, 3, 1)[0, y_idx, x_idx]

        ## collecting output
        detections = {}
        detections['bboxes'] = boxes_dict
        detections['scores'] = scores
        detections['classes'] = classes
        if fmap is not None:
            detections['fmap_feat'] = fmap
        if node_feature is not None:
            detections['node_feat'] = node_feature
        if reid_feature is not None:
            detections['reid_feat'] = reid_feature
        detections['batch_idx'] = torch.zeros(len(tlbr), device=device, dtype=torch.long)

        return detections

    def decode_gt_training(self, targets, det_outs):
        B, _, H, W = targets['hm'].shape

        gt_mask = targets['reg_mask'] > 0
        gt_ids = targets['id'][gt_mask]

        batch_idx = torch.zeros(int(targets['reg_mask'].sum()), dtype=torch.long, device=gt_ids.device)
        gt_idx_saver = 0
        for b_idx, num_gt in enumerate(targets['reg_mask'].sum(dim=1).type(torch.long)):
            batch_idx[gt_idx_saver:gt_idx_saver + num_gt] = b_idx
            gt_idx_saver += num_gt

        gt_whs = targets['wh'][gt_mask]
        gt_regs = targets['reg'][gt_mask]
        gt_inds = targets['ind'][gt_mask]  # in format of y*W + x
        gt_xs_int = gt_inds % W
        gt_ys_int = gt_inds // W
        gt_tlbrs = targets['tlbr'][gt_mask]
        gt_xs_tgt = (gt_xs_int + gt_regs[:, 0]).unsqueeze(1)
        gt_ys_tgt = (gt_ys_int + gt_regs[:, 1]).unsqueeze(1)
        scores = torch.ones_like(batch_idx).unsqueeze(1)
        clses = torch.zeros_like(batch_idx).unsqueeze(1)
        gt_whs = gt_whs.reshape(-1, 2, 2).sum(dim=1)

        bboxes = {'xs': gt_xs_tgt, 'ys': gt_ys_tgt, 'wh': gt_whs, 'tlbr': gt_tlbrs}

        fmap_feature_list, node_feature_list = [], []
        detection_batch_mask_list = []
        for b_idx in range(B):
            detection_batch_mask = batch_idx == b_idx
            detection_batch_mask_list.append(detection_batch_mask)
            gt_x_int = gt_xs_int[detection_batch_mask]
            gt_y_int = gt_ys_int[detection_batch_mask]
            fmap_feature = det_outs['fmap'][b_idx].permute(1, 2, 0)[gt_y_int, gt_x_int]
            node_feature = det_outs['node'][b_idx].permute(1, 2, 0)[gt_y_int, gt_x_int]

            ## normalize magnitude of features
            fmap_feature = F.normalize(fmap_feature, dim=1)
            node_feature = F.normalize(node_feature, dim=1)

            fmap_feature_list.append(fmap_feature)
            node_feature_list.append(node_feature)
        fmap_features = torch.cat(fmap_feature_list)
        node_features = torch.cat(node_feature_list)

        decoded_detections = {}
        decoded_detections['batch_idx'] = batch_idx
        decoded_detections['bboxes'] = bboxes
        decoded_detections['scores'] = scores
        decoded_detections['classes'] = clses
        decoded_detections['fmap_feat'] = fmap_features
        decoded_detections['node_feat'] = node_features

        return decoded_detections, gt_ids, detection_batch_mask_list


    def decode_topk(self, results, tlbr_flag=True):
        K = self.topk_det
        assert results['reg'] is not None and tlbr_flag  # Implementation based on this assumption & only one class

        hm = results["hm"].sigmoid()
        reg = results["reg"]
        wh = results["wh"]
        node_feature = results["node"]
        fmap = results["fmap"]
        reid_feature = results['id'] if 'id' in results else None
        detections = {}

        batch, channel, height, width = hm.shape
        hm = pseudo_nms(hm) # suppress neighboring peaks

        scores, index, clses, ys, xs = topk_score(hm, K=K)
        clses = clses.reshape(batch, K, 1).long()
        scores = scores.reshape(batch, K, 1)

        ## x and y coordinates
        reg = gather_feature(reg, index, use_transform=True)
        reg = reg.reshape(batch, K, 2)
        xs = xs.view(batch, K, 1) + reg[:, :, 0:1]
        ys = ys.view(batch, K, 1) + reg[:, :, 1:2]

        ## width and height
        wh = gather_feature(wh, index, use_transform=True)
        wh = wh.reshape(batch, K, 4)

        ## collect box info as dict
        tlbr = torch.cat([xs - wh[..., 0:1],
                          ys - wh[..., 1:2],
                          xs + wh[..., 2:3],
                          ys + wh[..., 3:4]], dim=2)
        if wh.size(-1) == 4:
            wh = wh.reshape(-1, 2, 2).sum(dim=1).reshape(*wh.shape[0:-1], 2)
        bboxes = {'xs': xs, 'ys': ys, 'wh': wh, 'tlbr': tlbr}

        ## indexing node/fmap/reid features
        if node_feature is not None:
            node_feature = F.normalize(node_feature, dim=1)
            node_feature = gather_feature(node_feature, index, use_transform=True)
        if fmap is not None:
            fmap = F.normalize(fmap, dim=1)
            fmap = gather_feature(fmap, index, use_transform=True)
        if reid_feature is not None:
            reid_feature = F.normalize(reid_feature, dim=1)
            reid_feature = gather_feature(reid_feature, index, use_transform=True)

        detections['bboxes'] = flatten_boxes_dict(bboxes)
        detections['scores'] = scores.flatten(0, 1)
        detections['classes'] = clses.flatten(0, 1)
        if fmap is not None:
            detections['fmap_feat'] = fmap.flatten(0, 1)
        if node_feature is not None:
            detections['node_feat'] = node_feature.flatten(0, 1)
        if reid_feature is not None:
            detections['reid_feat'] = reid_feature.flatten(0, 1)
        detections['batch_idx'] = torch.stack([torch.arange(batch)] * K, dim=1).reshape(-1)

        return detections

    def decode_thresh(self, results, tlbr_flag=True):
        hm = results["hm"].sigmoid()
        reg = results["reg"]
        wh = results["wh"]
        node_feature = results["node"]
        fmap = results["fmap"]
        reid_feature = results['id'] if 'id' in results else None
        detections = {}
        assert results['reg'] is not None and tlbr_flag  # Implementation based on this assumption & only one class

        batch, channel, height, width = hm.shape
        hm = pseudo_nms(hm)

        score_mask = hm >= self.det_low_thresh
        det_index = torch.nonzero(score_mask)  # [#dets, 4] where #dets in total batch, [batch_idx, cls index, y_index, x_index]
        scores = hm[score_mask].unsqueeze(1)
        batch_idx = det_index[:, 0]
        clses = det_index[:, 1].unsqueeze(1)
        ys_tgt = det_index[:, 2]
        xs_tgt = det_index[:, 3]

        ## x and y coordinates
        xs_offset_tgt = reg[:, 0:1][score_mask]
        ys_offset_tgt = reg[:, 1:2][score_mask]
        xs_tgt = xs_tgt + xs_offset_tgt
        ys_tgt = ys_tgt + ys_offset_tgt

        ## width and height
        w1 = wh[:, 0:1][score_mask]
        w2 = wh[:, 2:3][score_mask]
        h1 = wh[:, 1:2][score_mask]
        h2 = wh[:, 3:4][score_mask]

        wh_tgt = torch.stack([w1 + w2, h1 + h2], dim=1)

        ## collect box info as dict
        tlbr_tgt = torch.stack([xs_tgt - w1, ys_tgt - h1, xs_tgt + w2, ys_tgt + h2], dim=1)
        bboxes = {'xs': xs_tgt.unsqueeze(1), 'ys': ys_tgt.unsqueeze(1), 'wh': wh_tgt, 'tlbr': tlbr_tgt}

        ## indexing node/fmap/reid features
        if node_feature is not None:
            node_feature = F.normalize(node_feature, dim=1)
            node_feature = node_feature.permute(0, 2, 3, 1)[score_mask.squeeze(1)]
        if fmap is not None:
            fmap = F.normalize(fmap, dim=1)
            fmap = fmap.permute(0, 2, 3, 1)[score_mask.squeeze(1)]
        if reid_feature is not None:
            reid_feature = F.normalize(reid_feature, dim=1)
            reid_feature = reid_feature.permute(0, 2, 3, 1)[score_mask.squeeze(1)]

        detections['bboxes'] = bboxes
        detections['scores'] = scores
        detections['classes'] = clses
        detections['batch_idx'] = batch_idx
        if fmap is not None:
            detections['fmap_feat'] = fmap
        if node_feature is not None:
            detections['node_feat'] = node_feature
        if reid_feature is not None:
            detections['reid_feat'] = reid_feature

        return detections

    @torch.no_grad()
    def assign_gt_ids(self, targets, raw_detections, decoded_detections):
        if self.label_assignment_metric == 'dist':
            gt_ids, detection_batch_mask_list = self.label_gt_ids_by_distance(targets, raw_detections, decoded_detections)
        elif self.label_assignment_metric == 'iou':
            gt_ids, detection_batch_mask_list = self.label_gt_ids_by_iou(targets, raw_detections, decoded_detections)
        else:
            raise Exception(f'{self.label_assignment_metric} is not supported for assigning GT ids')

        return gt_ids, detection_batch_mask_list

    def label_gt_ids_by_distance(self, targets, raw_detections, decoded_detections):
        reg_mask = targets['reg_mask'] > 0
        topk_boxes_dict = decoded_detections['bboxes']
        matched_pred_nums, detection_batch_mask_list = [], []
        gt_matched_ids = torch.zeros(topk_boxes_dict['tlbr'].shape[0:-1], dtype=torch.long, device=reg_mask.device)

        ## assign GT ids to the prediction boxes
        for b_idx in range(reg_mask.size(0)):
            reg_mask_b = reg_mask[b_idx]
            gt_num_b = reg_mask_b.sum().item()
            detection_batch_mask = decoded_detections['batch_idx'] == b_idx
            detection_batch_mask_list.append(detection_batch_mask)
            if gt_num_b == 0:
                matched_pred_nums.append(0)
                continue

            gt_tlbr = targets['tlbr'][b_idx][reg_mask_b]
            gt_x, gt_y = (gt_tlbr[:, 0] + gt_tlbr[:, 2]) / 2, (gt_tlbr[:, 1] + gt_tlbr[:, 3]) / 2
            gt_xy = torch.stack([gt_x, gt_y], dim=1)
            gt_wh = targets['wh'][b_idx][reg_mask_b]
            gt_id = targets['id'][b_idx][reg_mask_b]

            pred_xy = torch.cat([topk_boxes_dict['xs'][detection_batch_mask], topk_boxes_dict['ys'][detection_batch_mask]], dim=1)
            pred_wh = topk_boxes_dict['wh'][detection_batch_mask]

            ## filter GT by center pairwise L1 distance
            xy_diff = (gt_xy.unsqueeze(1) - pred_xy.unsqueeze(0)).abs()
            gt_wh_thresh = gt_wh
            if gt_wh_thresh.size(1) == 4:
                gt_wh_thresh = gt_wh_thresh[:, :2] + gt_wh_thresh[:, 2:]
            gt_wh_thresh = gt_wh_thresh * self.min_wh_ratio # [#GT, C_wh] where C_wh is either 2 or 4
            # gt_wh_thresh = gt_wh_thresh.clamp(min=1) # if threshold is below than 1 pixel, clamp to 1?
            gt_mask = (xy_diff < gt_wh_thresh.unsqueeze(1)).all(dim=-1).any(dim=-1)
            unmatched_gt = np.arange(gt_num_b)[~gt_mask.cpu().numpy()]
            unmatched_pred = np.arange(pred_xy.size(0))

            if gt_mask.sum() > 0:
                # dist_mat = (xy_diff[gt_mask].pow(2)).sum(dim=-1).sqrt()
                xy_diff_norm_by_wh = xy_diff[gt_mask] / gt_wh_thresh[gt_mask].unsqueeze(1)
                dist_mat = xy_diff_norm_by_wh.pow(2).sum(dim=-1).sqrt()
                matches, unmatched_gt_masked, unmatched_pred = matching.linear_assignment(dist_mat.cpu().numpy(), thresh=np.inf)
                matched_pred_nums.append(len(matches))

                batch_matched_idx = torch.nonzero(detection_batch_mask).squeeze(1)[matches[:, 1]]
                gt_matched_ids[batch_matched_idx] = gt_id[gt_mask][matches[:, 0]]
                decoded_detections['classes'][batch_matched_idx] = 1
            ## fill unmatched GT boxes
            if self.fill_gt_flag:
                self.fill_unmatched_gt_label(b_idx, unmatched_gt, unmatched_pred, gt_tlbr, gt_wh, gt_id, decoded_detections, raw_detections, gt_matched_ids)

        # matched_pred_num = sum(matched_pred_nums)
        # gt_num = reg_mask.sum().item()
        # print(f"DIST Assignment (#matched/#GT) - {matched_pred_num}/{gt_num} - {100*matched_pred_num/gt_num:.2f}%")
        return gt_matched_ids, detection_batch_mask_list

    def label_gt_ids_by_iou(self, targets, raw_detections, decoded_detections):
        reg_mask = targets['reg_mask'] > 0
        topk_boxes_dict = decoded_detections['bboxes']
        iou_cost_thresh = 1 - self.min_iou
        matched_pred_nums, detection_batch_mask_list = [], []
        gt_matched_ids = torch.zeros(topk_boxes_dict['tlbr'].shape[0:-1], dtype=torch.long, device=reg_mask.device)

        ## assign GT ids to the prediction boxes
        for b_idx in range(reg_mask.size(0)):
            reg_mask_b = reg_mask[b_idx]
            gt_num_b = reg_mask_b.sum().item()
            detection_batch_mask = decoded_detections['batch_idx'] == b_idx
            detection_batch_mask_list.append(detection_batch_mask)
            if gt_num_b == 0:
                matched_pred_nums.append(0)
                continue

            gt_tlbr = targets['tlbr'][b_idx][reg_mask_b]
            gt_wh = targets['wh'][b_idx][reg_mask_b]
            gt_id = targets['id'][b_idx][reg_mask_b]

            pred_tlbr = topk_boxes_dict['tlbr'][detection_batch_mask]

            iou_mat = ops.box_iou(gt_tlbr, pred_tlbr) # Tensor(#gt, #pred)
            iou_cost_mat = 1 - iou_mat
            matches, unmatched_gt, unmatched_pred = matching.linear_assignment(iou_cost_mat.cpu().numpy(), thresh=iou_cost_thresh)
            matched_pred_num_b = len(matches)
            if matched_pred_num_b > 0:  # when all predicted boxes have IoU lower than min_iou, skip assigning GT id
                batch_matched_idx = torch.nonzero(detection_batch_mask).squeeze(1)[matches[:, 1]]
                gt_matched_ids[batch_matched_idx] = gt_id[matches[:, 0]]
                decoded_detections['classes'][batch_matched_idx] = 1

            matched_pred_nums.append(matched_pred_num_b)

            ## fill unmatched GT boxes
            if self.fill_gt_flag:
                self.fill_unmatched_gt_label(b_idx, unmatched_gt, unmatched_pred, gt_tlbr, gt_wh, gt_id, decoded_detections, raw_detections, gt_matched_ids)

        # matched_pred_num = sum(matched_pred_nums)
        # gt_num = reg_mask.sum().item()
        # print(f"IOU Assignment (#matched/#GT) - {matched_pred_num}/{gt_num} - {100*matched_pred_num/gt_num:.2f}%")
        return gt_matched_ids, detection_batch_mask_list

    def fill_unmatched_gt_label(self, b_idx, unmatched_gt, unmatched_pred, gt_tlbr, gt_wh, gt_id, decoded_detections, raw_detections, gt_matched_ids):
        unmatched_gt_num_b, unmatched_pred_num_b = len(unmatched_gt), len(unmatched_pred)
        fill_gt_num = min(unmatched_gt_num_b, unmatched_pred_num_b)
        if fill_gt_num > 0:
            ## slice GT labels by the remaining detection slots
            unmatched_gt = unmatched_gt[:fill_gt_num]
            unmatched_gt_wh = gt_wh[unmatched_gt]
            unmatched_gt_tlbr = gt_tlbr[unmatched_gt]

            ## Convert GT tlbr into cxcy format and discretize
            unmatched_gt_cx = ((unmatched_gt_tlbr[:, 0] + unmatched_gt_tlbr[:, 2]) / 2)
            unmatched_gt_cy = ((unmatched_gt_tlbr[:, 1] + unmatched_gt_tlbr[:, 3]) / 2)
            ## TODO. should we just filter out GT boxes if they are out from image too much? (e.g. 80% in-image)
            unmatched_gt_cx = unmatched_gt_cx.clamp(min=0, max=self.hm_width - 1)
            unmatched_gt_cy = unmatched_gt_cy.clamp(min=0, max=self.hm_height - 1)
            unmatched_gt_cx_int = unmatched_gt_cx.type(torch.long)
            unmatched_gt_cy_int = unmatched_gt_cy.type(torch.long)

            ## replace topk pred by GT
            if self.replace_unmatched_pred == 'reverse': ## option1. reversely order
                unmatched_pred = unmatched_pred[::-1].copy()
            elif self.replace_unmatched_pred == 'random': ## option2. randomly mix
                np.random.shuffle(unmatched_pred)

            detection_batch_mask = decoded_detections['batch_idx'] == b_idx
            u_pred_idx = unmatched_pred[:fill_gt_num]
            batch_u_pred_idx = torch.nonzero(detection_batch_mask).squeeze(1)[u_pred_idx]
            ## gradient is detached from this
            decoded_detections['fmap_feat'][batch_u_pred_idx, :] = raw_detections['fmap'][b_idx][:, unmatched_gt_cy_int, unmatched_gt_cx_int].permute(1, 0)
            decoded_detections['node_feat'][batch_u_pred_idx, :] = raw_detections['node'][b_idx][:, unmatched_gt_cy_int, unmatched_gt_cx_int].permute(1, 0)
            decoded_detections['bboxes']['xs'][batch_u_pred_idx, :] = unmatched_gt_cx.reshape(-1, 1)
            decoded_detections['bboxes']['ys'][batch_u_pred_idx, :] = unmatched_gt_cy.reshape(-1, 1)
            decoded_detections['bboxes']['wh'][batch_u_pred_idx, :] = unmatched_gt_wh
            decoded_detections['bboxes']['tlbr'][batch_u_pred_idx, :] = unmatched_gt_tlbr
            decoded_detections['scores'][batch_u_pred_idx, :] = 1.0
            decoded_detections['classes'][batch_u_pred_idx, :] = 2

            ## fill GT matched id
            gt_matched_ids[batch_u_pred_idx] = gt_id[unmatched_gt]
