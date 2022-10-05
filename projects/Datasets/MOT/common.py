import logging
import random
import numpy as np
import torch.utils.data as data

from detectron2.utils.serialize import PicklableWrapper


class MapDataset(data.Dataset):

    def __init__(self, cfg, dataset, map_func, is_train):
        self._cfg = cfg
        self._dataset = dataset
        self._map_func = PicklableWrapper(map_func)  # wrap so that a lambda will work

        self._rng = random.Random(42)
        self._fallback_candidates = set(range(len(dataset)))

        self.num_samples = cfg.DATALOADER.NUM_SAMPLES_TRAIN if is_train else cfg.DATALOADER.NUM_SAMPLES_TEST
        self.max_frame_dist = cfg.DATALOADER.MAX_FRAME_DIST_TRAIN if is_train else cfg.DATALOADER.MAX_FRAME_DIST_TEST
        self.min_frame_dist = cfg.DATALOADER.MIN_FRAME_DIST_TRAIN if is_train else cfg.DATALOADER.MIN_FRAME_DIST_TEST
        self.pre_sample_flag = cfg.DATALOADER.PRE_SAMPLE_FLAG
        if not is_train and self.pre_sample_flag: ## pre-sampling for only test-phase (sparse video testing environment)
            self.max_frame_dist, self.min_frame_dist = 1, 1
        assert self.max_frame_dist >= self.min_frame_dist, "Max frame dist ({}) must be >= Min frame dist ({})".format(self.max_frame_dist, self.min_frame_dist)

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, idx):
        retry_count = 0
        cur_idx = int(idx)

        while True:
            data = self._map_func(self.get_data(cur_idx))
            if data is not None:
                self._fallback_candidates.add(cur_idx)
                return data

            # _map_func fails for this idx, use a random new index from the pool
            retry_count += 1
            self._fallback_candidates.discard(cur_idx)
            cur_idx = self._rng.sample(self._fallback_candidates, k=1)[0]

            if retry_count >= 3:
                logger = logging.getLogger(__name__)
                logger.warning(
                    "Failed to apply `_map_func` for idx: {}, retry count: {}".format(
                        idx, retry_count
                    )
                )

    def get_data(self, cur_idx):
        data = []
        if self.num_samples == 1:
            data.append(self._dataset[cur_idx])
        else:
            sequence_name = self._dataset[cur_idx]['sequence_name']
            frame_offsets = self.get_frame_offsets()
            for i in frame_offsets:
                next_idx = cur_idx + i
                if next_idx < len(self._dataset) and self._dataset[next_idx]['sequence_name'] == sequence_name:
                    data.append(self._dataset[next_idx])
                else:
                    data.append(data[-1])
        return data

    def get_frame_offsets(self):
        if self.max_frame_dist == 0:
            frame_offsets = [0] * self.num_samples
        else:
            # frame_idxs = list(range(1, self.max_frame_dist)) if self.max_frame_dist != 1 else [1]
            frame_idxs = list(range(self.min_frame_dist, self.max_frame_dist+1))
            random_idxs = random.sample(frame_idxs, k=(self.num_samples - 1))
            random_idxs.sort()
            frame_offsets = [0, *random_idxs]
        return frame_offsets
