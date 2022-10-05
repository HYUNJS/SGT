import torch
from torch import nn
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from projects.FairMOT.fairmot.build import build_detector, build_tracker

__all__ = ["FairMOT"]


@META_ARCH_REGISTRY.register()
class FairMOT(nn.Module):
    """
    Generalized Tracker. Any models that contains the following two components:
    1. Per-image object detector
    2. Causal or online object tracker
    """

    def __init__(self, cfg):
        super().__init__()
        self.detector = build_detector(cfg)
        self.tracker = build_tracker(cfg)
        self.cur_seq_name = ""
        self.seq_name_list = []

    @property
    def device(self):
        return self.pixel_mean.device

    def postprocess(self, results):
        inst = results[0]['instances']
        results[0]['instances'] = inst[inst.pred_ids != 0]

    def forward(self, batched_inputs):
        if not self.training:
            return self.inference(batched_inputs)
        else:
            losses = self.detector(batched_inputs)
            return losses

    def inference(self, batched_inputs, detected_instances=None, do_postprocess=True):
        assert not self.training

        if self.cur_seq_name != batched_inputs[0]["sequence_name"]:
            self.cur_seq_name = batched_inputs[0]["sequence_name"]
            self.seq_name_list.append(self.cur_seq_name)
            self.tracker.init(batched_inputs[0]['seq_fps'])

        detections = self.detector(batched_inputs)
        results = self.tracker(detections)
        self.postprocess(results)
        return results
