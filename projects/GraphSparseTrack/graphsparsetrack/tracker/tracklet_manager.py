import torch


class TrackletManager(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.topk_det = cfg.MODEL.TRACKER.GNN.TOPK_DET
        self.history_by_fps = cfg.MODEL.TRACKER.MANAGER.HISTORY_BY_FPS
        self.min_len_by_fps = cfg.MODEL.TRACKER.MANAGER.MIN_LEN_BY_FPS
        self.history_len = cfg.MODEL.TRACKER.MANAGER.HISTORY_LEN
        self.min_tracklet_len = cfg.MODEL.TRACKER.MANAGER.MIN_TRACKLET_LEN
        self.min_tracklet_sec = cfg.MODEL.TRACKER.MANAGER.MIN_TRACKLET_SEC
        self.tracklets = None
        self.ages = torch.tensor([])
        self.tgt_keys = ['tids', 'boxes_dict', 'fmap_feats', 'node_feats', 'scores']
        self.update_flag = self.history_len > 0
        self.tracklet_lens = {}

    def len(self):
        return self.ages.size(0)

    def reset(self, fps):
        self.ages = torch.tensor([])
        self.tracklets = None
        self.tracklet_lens = {}

        if self.history_by_fps:
            self.set_history_len(fps)
        if self.min_len_by_fps:
            self.set_min_tracklet_len(round(fps * self.min_tracklet_sec))

    def set_history_len(self, max_age):
        self.history_len = max_age

    def set_min_tracklet_len(self, min_age):
        self.min_tracklet_len = min_age

    def add_tracklets(self, t1_info, not_found_mask_t1):
        num_tracklets = not_found_mask_t1.sum()
        tracklets = {}
        for k in self.tgt_keys:
            if k == 'boxes_dict':
                tracklets[k] = {}
                for sub_k in t1_info['tensor'][k].keys():
                    ndim = t1_info['tensor'][k][sub_k].dim()
                    if ndim == 2:
                        tracklets[k][sub_k] = t1_info['tensor'][k][sub_k][not_found_mask_t1]
                    elif ndim == 3:
                        tracklets[k][sub_k] = t1_info['tensor'][k][sub_k][:, not_found_mask_t1]
                    else:
                        raise NotImplementedError("Add tracklets with attribute which dim {} is not supported".format(ndim))
            else:
                if k == 'tids' or k == 'scores':
                    tracklets[k] = t1_info['tensor'][k][:, not_found_mask_t1]
                else:
                    tracklets[k] = t1_info['tensor'][k][not_found_mask_t1]

        if self.tracklets is None:
            self.tracklets = tracklets
        else:
            for k in self.tgt_keys:
                if k == 'boxes_dict':
                    for sub_k in t1_info['tensor'][k].keys():
                        n_dim = tracklets[k][sub_k].dim()
                        if n_dim == 2:
                            self.tracklets[k][sub_k] = torch.cat([self.tracklets[k][sub_k], tracklets[k][sub_k]], dim=0)
                        elif n_dim == 3:
                            self.tracklets[k][sub_k] = torch.cat([self.tracklets[k][sub_k], tracklets[k][sub_k]], dim=1)
                        else:
                            raise NotImplementedError("[add_tracklets] unmatched dimension of tracklets {}".format(k))
                else:
                    if k == 'tids' or k == 'scores':
                        self.tracklets[k] = torch.cat([self.tracklets[k], tracklets[k]], dim=1)
                    else:
                        self.tracklets[k] = torch.cat([self.tracklets[k], tracklets[k]], dim=0)
        self.ages = torch.cat([self.ages, torch.ones(num_tracklets)])

    def update_tracklets(self, not_found_mask_history):
        ## update age of tracklets which are not found in T2
        self.ages[not_found_mask_history] += 1

        ## remove tracklets which are found in T2
        self.ages = self.ages[not_found_mask_history]
        for k in self.tgt_keys:
            if k == 'boxes_dict':
                for sub_k in self.tracklets[k].keys():
                    n_dim = self.tracklets[k][sub_k].dim()
                    if n_dim == 2:
                        self.tracklets[k][sub_k] = self.tracklets[k][sub_k][not_found_mask_history]
                    elif n_dim == 3:
                        self.tracklets[k][sub_k] = self.tracklets[k][sub_k][:, not_found_mask_history]
                    else:
                        raise NotImplementedError("[update_tracklets] self.tracklets[{}][{}] dim : {} not supported".format(k, sub_k, n_dim))
            else:
                if k == 'tids' or k == 'scores':
                    self.tracklets[k] = self.tracklets[k][:, not_found_mask_history]
                else:
                    self.tracklets[k] = self.tracklets[k][not_found_mask_history]

    def remove_inactive_tracklets(self):
        inactive_mask = self.ages > self.history_len
        if inactive_mask.sum() > 0:
            active_mask = self.ages <= self.history_len
            self.ages = self.ages[active_mask]
            for k in self.tgt_keys:
                if k == 'boxes_dict':
                    for sub_k in self.tracklets[k].keys():
                        n_dim = self.tracklets[k][sub_k].dim()
                        if n_dim == 2:
                            self.tracklets[k][sub_k] = self.tracklets[k][sub_k][active_mask]
                        elif n_dim == 3:
                            self.tracklets[k][sub_k] = self.tracklets[k][sub_k][:, active_mask]
                        else:
                            raise NotImplementedError("[remove_inactive_tracklets] self.tracklets[{}][{}] dim : {} not supported".format(k, sub_k, n_dim))
                else:
                    if k == 'tids' or k == 'scores':
                        self.tracklets[k] = self.tracklets[k][:, active_mask]
                    else:
                        self.tracklets[k] = self.tracklets[k][active_mask]

    def record_tids(self, tids):
        for t in tids[(tids != 0)].cpu().numpy():
            if t not in self.tracklet_lens:
                self.tracklet_lens[t] = 0
            self.tracklet_lens[t] += 1

    def filter_noisy_tracklets(self, t1_info, not_found_mask_t1):
        '''
            Do not add tracklets which length is shorter than thresh
            Because they might be False Positive tracklets
            By doing so, it can reduce the FP association that FP detection are included in long-term matching
        '''
        non_zero_idxs = torch.nonzero(not_found_mask_t1)
        for idx, not_found_tid in enumerate(t1_info['tensor']['tids'][0, not_found_mask_t1].cpu().numpy()):
            if self.tracklet_lens[not_found_tid] <= self.min_tracklet_len:
                not_found_mask_t1[non_zero_idxs[idx]] = False
        return not_found_mask_t1

    def update_history(self, t1_info, t2_info):
        if not self.update_flag or len(t1_info.keys()) == 0:
            return
        n_box = t1_info['tensor']['tids'].size(1)
        t1_tids = t1_info['tensor']['tids']  # [1, k or n_box]
        t2_tids = t2_info['tensor']['tids'] if 'tids' in t2_info['tensor'] else torch.zeros(1, 0) # [1, k]
        self.record_tids(t1_tids)

        if t2_tids.size(1) == 0:
            not_found_mask_t1 = torch.ones(t1_tids.size(1), dtype=torch.bool)
            if self.tracklets is not None:
                not_found_mask_history = torch.ones(self.tracklets['tids'].size(1), dtype=torch.bool)
            else:
                not_found_mask_history = torch.ones(0, dtype=torch.bool)
        else:
            ## Find [t1_tids; history] not in t2 tids
            if self.tracklets is not None:
                t1_tids = torch.cat([t1_tids, self.tracklets['tids']], dim=1)  # [1, k+N]
            t1_mat = t1_tids.repeat(t2_tids.size(1), 1)  # [k, k+N or n_box+N]
            t2_mat = t2_tids.t().repeat(1, t1_tids.size(1)) # [k, k+N or n_box+N]
            t1_t2_eq_mat = t1_mat == t2_mat
            not_found_mask = (t1_t2_eq_mat.sum(dim=0) == 0)  # [k+N or n_box+N]
            not_found_mask = torch.logical_and(not_found_mask, (t1_tids != 0).squeeze())
            not_found_mask_t1 = not_found_mask[0:n_box]
            not_found_mask_history = not_found_mask[n_box:]

        ## update tracklets
        if len(not_found_mask_history) > 0:
            self.update_tracklets(not_found_mask_history)

        ## terminate inactive tracklets
        if len(self.ages) > 0:
            self.remove_inactive_tracklets()

        ## add new tracklets
        if not_found_mask_t1.sum() > 0:
            not_found_mask_t1_filtered = self.filter_noisy_tracklets(t1_info, not_found_mask_t1)
            if not_found_mask_t1_filtered.sum() > 0:
                self.add_tracklets(t1_info, not_found_mask_t1_filtered)