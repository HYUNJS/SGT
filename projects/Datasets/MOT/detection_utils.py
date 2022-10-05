import torch
import numpy as np

from detectron2.data import transforms as T
from detectron2.structures import Boxes, BoxMode, Instances, Keypoints
from detectron2.data.detection_utils import transform_keypoint_annotations


def transform_instance_annotations(
    annotation, transforms, image_size, keypoint_hflip_indices=None, clip_by_image=False, filter_out_image=False):
    if isinstance(transforms, (tuple, list)):
        transforms = T.TransformList(transforms)
    # bbox is 1d (per-instance bounding box)
    bbox = BoxMode.convert(annotation["bbox"], annotation["bbox_mode"], BoxMode.XYXY_ABS)
    bbox = transforms.apply_box(np.array([bbox]))
    if len(bbox) == 0:
        return None
    bbox = bbox[0]
    if clip_by_image:
        bbox = bbox.clip(min=(0, 0, 0, 0), max=list(image_size + image_size)[::-1])
    if not clip_by_image and filter_out_image:
        min_size = 1.0
        transformed_bbox = bbox.clip(min=(0, 0, 0, 0), max=list(image_size + image_size)[::-1])
        if (transformed_bbox[2] - transformed_bbox[0] < min_size) or (transformed_bbox[3] - transformed_bbox[1] < min_size):
            transformed_bbox
            return None

    annotation["bbox"] = bbox
    annotation["bbox_mode"] = BoxMode.XYXY_ABS
    if "keypoints" in annotation:
        keypoints = transform_keypoint_annotations(
            annotation["keypoints"], transforms, image_size, keypoint_hflip_indices
        )
        annotation["keypoints"] = keypoints
    return annotation


def annotations_to_instances(annos, image_size):
    boxes = [BoxMode.convert(obj["bbox"], obj["bbox_mode"], BoxMode.XYXY_ABS) for obj in annos]
    target = Instances(image_size)
    target.gt_boxes = Boxes(boxes)
    classes = [obj["category_id"] for obj in annos]
    classes = torch.tensor(classes, dtype=torch.int64)
    target.gt_classes = classes

    ids = [obj["track_id"] for obj in annos]
    # ids = torch.tensor(ids, dtype=torch.int16)
    ids = torch.tensor(ids, dtype=torch.long)
    target.gt_ids = ids
    if len(annos) and "keypoints" in annos[0]:
        kpts = [obj.get("keypoints", []) for obj in annos]
        target.gt_keypoints = Keypoints(kpts)
    return target
