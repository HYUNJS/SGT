import os

from detectron2.data.datasets.builtin_meta import _get_builtin_metadata
from detectron2.data.datasets.register_coco import register_coco_instances

_PREDEFINED_SPLITS_COCO = {}
_PREDEFINED_SPLITS_COCO["coco"] = {
    "nfs_coco_2017_train": ("coco", "train2017", "annotations/instances_train2017.json"),
    "nfs_coco_2017_val": ("coco", "val2017", "annotations/instances_val2017.json"),
    "nfs_coco_2017_test": ("coco", "test2017", "annotations/image_info_test2017.json"),
}

_PREDEFINED_SPLITS_COCO["coco_person"] = {
    "nfs_keypoints_coco_2017_train": ("coco", "train2017", "annotations/person_keypoints_train2017.json"),
    "nfs_keypoints_coco_2017_val": ("coco", "val2017", "annotations/person_keypoints_val2017.json"),
}


def register_all_coco(root):
    for dataset_name, splits_per_dataset in _PREDEFINED_SPLITS_COCO.items():
        for key, (dataset_dir, image_root, json_file) in splits_per_dataset.items():
            register_coco_instances(
                key,
                _get_builtin_metadata(dataset_name),
                os.path.join(root, dataset_dir, json_file) if "://" not in json_file else json_file,
                os.path.join(root, dataset_dir, image_root),
            )
