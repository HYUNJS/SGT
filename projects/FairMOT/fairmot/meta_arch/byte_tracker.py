import logging
import torch
import numpy as np
from scipy.optimize import linear_sum_assignment

from torch import nn
import torch.nn.functional as F
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from projects.FairMOT.fairmot.tracker import matching
from projects.FairMOT.fairmot.tracker.basetrack import TrackState
from projects.FairMOT.fairmot.tracker.multitracker import STrack, joint_stracks, sub_stracks, remove_duplicate_stracks
from projects.FairMOT.fairmot.tracker.kalman_filter import KalmanFilter

__all__ = ["BYTETracker"]


@META_ARCH_REGISTRY.register()
class BYTETracker(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        tracker_cfg = cfg.MODEL.TRACKER.BYTETRACK
        self.tracker_cfg = tracker_cfg

        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]
        self.buffer_size = None
        self.max_time_lost = None
        self.frame_id = 0

        ## configs for thesholding values
        self.det_thres = tracker_cfg.DET_THRES
        self.low_det_thres = tracker_cfg.LOW_DET_THRES
        self.init_track_det_thres = tracker_cfg.INIT_TRACK_DET_THRES
        self.first_match_thres = tracker_cfg.FIRST_MATCH_THRES
        self.second_match_thresh = tracker_cfg.SECOND_MATCH_THRES
        self.low_match_thresh = tracker_cfg.LOW_MATCH_THRES

        ## configs for extra tracking components
        self.kalman_filter_flag = tracker_cfg.KALMAN_FILTER_FLAG
        self.feature_smooth_flag = tracker_cfg.FEATURE_SMOOTH_FLAG
        self.feature_smooth_by_score = tracker_cfg.FEATURE_SMOOTH_BY_SCORE
        self.fuse_motion_flag = tracker_cfg.FUSE_MOTION_FLAG
        self.kalman_filter = KalmanFilter() if self.kalman_filter_flag or self.fuse_motion_flag else None

    def init(self, frame_rate=30):
        self.tracked_stracks = []
        self.lost_stracks = []
        self.removed_stracks = []
        self.frame_id = 0
        self.buffer_size = int(frame_rate / 30.0 * self.tracker_cfg.TRACK_BUFFER)
        self.max_time_lost = self.buffer_size

    def forward(self, predictions):
        results = []
        for prediction in predictions:
            results.append(self.update(prediction))
        return results

    def _postprocess(self, output):
        remain_inds = output["instances"].scores >= self.det_thres
        boxes = output["instances"].pred_boxes.tensor
        boxes = boxes[remain_inds.cpu().numpy()]
        classes = output["instances"].pred_classes[remain_inds].detach().cpu().numpy()
        scores = output["instances"].scores[remain_inds].detach().cpu().numpy()
        id_feature = output["instances"].id_feature[remain_inds].detach().cpu().numpy()
        query_indices = torch.nonzero(remain_inds).view(-1).cpu().numpy()
        return boxes, classes, scores, id_feature, query_indices

    def _get_low_dets(self, output):
        scores = output["instances"].scores
        remain_inds = torch.logical_and(scores < self.det_thres, scores >= self.low_det_thres)
        boxes = output["instances"].pred_boxes.tensor
        boxes = boxes[remain_inds.cpu().numpy()]
        classes = output["instances"].pred_classes[remain_inds].detach().cpu().numpy()
        scores = output["instances"].scores[remain_inds].detach().cpu().numpy()
        id_feature = output["instances"].id_feature[remain_inds].detach().cpu().numpy()
        query_indices = torch.nonzero(remain_inds).view(-1).cpu().numpy()
        return boxes, classes, scores, id_feature, query_indices

    def update(self, prediction):
        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        boxes, classes, scores, id_feature, query_indices = self._postprocess(prediction)
        l_boxes, l_classes, l_scores, l_id_feature, l_query_indices = self._get_low_dets(prediction)
        if len(boxes) > 0:
            '''Detections'''
            detections = [STrack(STrack.tlbr_to_tlwh(box), cls, score, embedding, 30, query_idx, self.kalman_filter_flag,
                        self.feature_smooth_flag, self.feature_smooth_by_score) for (box, cls, score, embedding, query_idx)
                        in zip(boxes, classes, scores, id_feature, query_indices)]
        else:
            detections = []

        ''' Add newly detected tracklets to tracked_stracks'''
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        ''' Step 2: First association, with embedding'''
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        # Predict the current location with KF
        if self.kalman_filter_flag:
            STrack.multi_predict(strack_pool)
        dists = matching.embedding_distance(strack_pool, detections)
        if self.fuse_motion_flag:
            dists = matching.fuse_motion(self.kalman_filter, dists, strack_pool, detections) ## this is critical step - most cost becomes np.inf
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.first_match_thres)

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(detections[idet], self.frame_id, self.feature_smooth_flag)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        ''' Step 3: Second association, with IOU'''
        detections = [detections[i] for i in u_detection]
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        dists = matching.iou_distance(r_tracked_stracks, detections)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.second_match_thresh)

        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        ''' ByteTrack part '''
        # association the untrack to the low score detections
        if len(l_boxes) > 0:
            detections_second = [STrack(STrack.tlbr_to_tlwh(l_box), l_cls, l_score, l_emb, 30, l_query_idx, self.kalman_filter_flag,
                                self.feature_smooth_flag, self.feature_smooth_by_score) for (l_box, l_cls, l_score, l_emb, l_query_idx)
                                 in zip(l_boxes, l_classes, l_scores, l_id_feature, l_query_indices)]
        else:
            detections_second = []

        second_tracked_stracks = [r_tracked_stracks[i] for i in u_track if r_tracked_stracks[i].state == TrackState.Tracked]
        dists = matching.iou_distance(second_tracked_stracks, detections_second)
        matches, u_track, u_detection_second = matching.linear_assignment(dists, thresh=self.low_match_thresh)
        for itracked, idet in matches:
            track = second_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)


        for it in u_track:
            # track = r_tracked_stracks[it] ## ERROR CODE
            track = second_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        detections = [detections[i] for i in u_detection]
        dists = matching.iou_distance(unconfirmed, detections)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_starcks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        """ Step 4: Init new stracks"""
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.init_track_det_thres:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated_starcks.append(track)
        """ Step 5: Update state"""
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        # print('Ramained match {} s'.format(t4-t3))

        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)
        # get scores of lost tracks
        output_stracks = [track for track in self.tracked_stracks if track.is_activated]

        logger = logging.getLogger(__name__)
        logger.debug('===========Frame {}=========='.format(self.frame_id))
        logger.debug('Activated: {}'.format([track.track_id for track in activated_starcks]))
        logger.debug('Refind: {}'.format([track.track_id for track in refind_stracks]))
        logger.debug('Lost: {}'.format([track.track_id for track in lost_stracks]))
        logger.debug('Removed: {}'.format([track.track_id for track in removed_stracks]))

        online_tlwhs = []
        online_ids = []
        valid_mask = []
        for i, t in enumerate(output_stracks):
            tlwh = t.tlwh
            tid = t.track_id
            online_tlwhs.append(tlwh)
            online_ids.append(tid)
            valid_mask.append(i)
        # for i, t in enumerate(output_stracks):
        #     tlwh = t.tlwh
        #     tid = t.track_id
        #     vertical = tlwh[2] / tlwh[3] > 1.6
        #     if tlwh[2] * tlwh[3] > self.tracker_cfg.MIN_BOX_AREA and not vertical:
        #         online_tlwhs.append(tlwh)
        #         online_ids.append(tid)
        #         valid_mask.append(i)

        prediction = self.set_ids(prediction, output_stracks, valid_mask)

        results = {
            "frame_id": self.frame_id,
            "online_tlwhs": online_tlwhs,
            "online_ids": online_ids,
            "instances": prediction["instances"],
        }
        return results

    def set_ids(self, prediction, output_stracks, valid_mask):
        boxes = prediction["instances"].pred_boxes.tensor
        n, _ = boxes.shape
        ids = torch.zeros(n, dtype=torch.int32)
        for i, t in enumerate(output_stracks):
            if i in valid_mask:
                ## TODO. ERROR here - query_idx not matched - reproduced by oracle analysis & BYTE & fairmot_DLA_34_1x_66
                # if len(ids) <= t.query_idx:
                #     t.query_idx
                ids[t.query_idx] = t.track_id
        prediction["instances"].pred_ids = ids
        return prediction
