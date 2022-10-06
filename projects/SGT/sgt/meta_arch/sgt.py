import itertools, os, cv2, torch
from torch import nn
import numpy as np
import os.path as osp

from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.structures import Boxes, Instances

from projects.Datasets.Transforms.augmentation import CenterAffine
from projects.SGT.sgt.build import build_detector, build_tracker
from projects.SGT.sgt.meta_arch.loss import MultiLossNet


__all__ = ["SparseGraphTracker"]


@META_ARCH_REGISTRY.register()
class SparseGraphTracker(nn.Module):
    """
    Generalized Tracker. Any models that contains the following two components:
    1. Per-image object detector
    2. Causal or online object tracker
    """

    def __init__(self, cfg):
        super().__init__()
        self.detector = build_detector(cfg)
        self.tracker = build_tracker(cfg)
        self.loss_net = MultiLossNet(cfg)
        self.cur_seq_name = ""
        self.seq_name_list = []
        self.parallel_batch = cfg.DATALOADER.PARALLEL_BATCH
        assert self.parallel_batch
        self.prev_seq_name = ''
        self.prev_frame_idx = -1
        self.mems = {}

        self.debug = False
        self.clip_by_image = cfg.INPUT.CLIP_BY_IMAGE

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, batched_inputs):
        if not self.training:
            return self.inference(batched_inputs)
        else:
            batched_inputs = list(itertools.chain(*batched_inputs))
            detector_outs = self.detector(batched_inputs, return_all=True)
            det_loss_dict = detector_outs['losses']
            trk_loss_dict, edge_cls_metrics = self.tracker(detector_outs, batched_inputs=batched_inputs)
            final_loss_dict = self.loss_net(det_loss_dict, trk_loss_dict)

            return final_loss_dict

    @torch.no_grad()
    def inference(self, batched_inputs):
        new_sequence = self.prev_seq_name != batched_inputs[0]['sequence_name']
        new_sequence = new_sequence or self.prev_frame_idx > batched_inputs[0]['frame_idx']
        self.prev_seq_name = batched_inputs[0]['sequence_name']
        self.prev_frame_idx = batched_inputs[0]['frame_idx']
        if new_sequence:
            fps = batched_inputs[0]['seq_fps'] if 'seq_fps' in batched_inputs[0] else 30
            self.tracker.reset(fps)
            self.mems = {}

        ## Inference detector
        images = self.detector.preprocess_image(batched_inputs)
        detector_outs = self.detector.inference(images, batched_inputs, do_postprocess=False, return_all=True)
        img_info = detector_outs['img_info']

        ## Inference gnn tracker
        if self.debug: # for debugging
            detector_outs['img_meta'] = {
                'image': self.detector.denormalizer(images.tensor[0]).permute(1, 2, 0).cpu().numpy()[:,:,::-1],
                'filename': batched_inputs[0]['file_name'],
                'img_name': batched_inputs[0]['sequence_name'] + '-' + str(batched_inputs[0]['frame_idx']+1)
            }

        results = self.tracker.inference(detector_outs, self.mems, batched_inputs)
        results = self._postprocess(results, img_info) # transform bboxes to the original image resolution

        return results

    def _postprocess(self, results, img_info):
        boxes, scores, classes, tids = results['boxes'], results['scores'], results['classes'], results['tids']

        mask = tids != 0
        scores = scores[mask].reshape(-1)
        classes = classes[mask].reshape(-1).to(torch.int64)
        tids = tids[mask].reshape(-1)
        boxes = boxes[mask] if boxes.dim() == 3 else boxes[mask.squeeze(0)]

        scores = scores.reshape(-1)
        classes = classes.reshape(-1).to(torch.int64)
        tids = tids.reshape(-1)

        boxes = self.transform_boxes(boxes, img_info)
        if self.clip_by_image:
            boxes, valid_size_mask = self.fit_bbox_to_img(boxes, img_info)
            if np.logical_not(valid_size_mask).sum() > 0:
                boxes = boxes[valid_size_mask]
                tids = tids[valid_size_mask]
                scores = scores[valid_size_mask]
                classes = classes[valid_size_mask]
        boxes = Boxes(boxes)

        output = dict(pred_boxes=boxes, scores=scores, pred_classes=classes, pred_ids=tids)
        ori_w, ori_h = img_info['center'] * 2
        det_instance = Instances((int(ori_h), int(ori_w)), **output)

        return [{"instances": det_instance}]

    @staticmethod
    def fit_bbox_to_img(boxes, img_info):
        '''
        Compared to mot17, mot20 bbox GT format is limited to the image size
        min(tlx or tly): 1
        max(tlx+w or tly+h): 1 + img_size
        '''
        img_w, img_h = img_info['center'] * 2
        boxes = boxes.clip(min=0)
        boxes[:, 0::2] = boxes[:, 0::2].clip(max=img_w)
        boxes[:, 1::2] = boxes[:, 1::2].clip(max=img_h)

        ## remove if width and height <= 1
        h = (boxes[:, 2] - boxes[:, 0]) > 1
        w = (boxes[:, 3] - boxes[:, 1]) > 1
        valid_size_mask = np.logical_and(h, w)

        return boxes, valid_size_mask

    @staticmethod
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