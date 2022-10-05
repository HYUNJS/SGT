from torch.utils.data.sampler import Sampler
from detectron2.utils import comm
from typing import List


class SeqInferenceSampler(Sampler):
    def __init__(self, seq_name_list: List, size: int):
        """
        Args:
            size (int): the total number of data of the underlying dataset to sample from
        """
        self._size = size
        assert size > 0
        self._rank = comm.get_rank()
        self._world_size = comm.get_world_size()
        self.idx_per_seq = self.build_idx_per_seq(seq_name_list)
        self.sequence_names = list(self.idx_per_seq.keys())
        self.pad_idx_per_rank = self.build_idx_per_rank()
        self._local_indices = self.pad_idx_per_rank[self._rank]

    def build_idx_per_seq(self, seq_name_list):
        idx_per_seq = {}
        for idx, seq_name in enumerate(seq_name_list):
            if seq_name not in idx_per_seq:
                idx_per_seq[seq_name] = []
            idx_per_seq[seq_name].append(idx)
        return idx_per_seq

    def build_idx_per_rank(self):
        total_num_sequence = len(self.sequence_names)
        num_seq_per_rank = (total_num_sequence - 1) // self._world_size + 1
        idx_per_rank = {}
        for rank_id in range(self._world_size):
            begin_seq_idx = rank_id * num_seq_per_rank
            end_seq_idx = (rank_id + 1) * num_seq_per_rank
            idx_list = []
            for seq_name in self.sequence_names[begin_seq_idx:end_seq_idx]:
                idx_list = [*idx_list, *self.idx_per_seq[seq_name]]
            idx_per_rank[rank_id] = idx_list

        pad_idx_per_rank = self.pad_idx(idx_per_rank)
        return pad_idx_per_rank

    def pad_idx(self, idx_per_rank):
        len_per_rank = [len(x) for x in idx_per_rank.values()]
        max_len = max(len_per_rank)
        new_idx_per_rank = {}
        for rank, indicies in idx_per_rank.items():
            pad_size = max_len - len(indicies)
            if pad_size > 0:
                pad_indices = indicies + indicies[:pad_size]
            else:
                pad_indices = indicies
            new_idx_per_rank[rank] = pad_indices
        return new_idx_per_rank

    def __iter__(self):
        yield from self._local_indices

    def __len__(self):
        return len(self._local_indices)
