import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from projects.Datasets.MOT.mot_det import load_mot_dataset_dicts


CLASS_NAMES = [
    "pedestrian",
]

CLASS_COLORS = [
    (0,255,0),
]

caching = False

SPLITS = [
    ## MOT16
    ("mot16_train", "MOT16", "train", None, caching),
    ("mot16_test", "MOT16", "test", None, caching),
    ## MOT17
    ("mot17_train", "MOT17", "train", "SDP", caching),
    ("mot17_test", "MOT17", "test", "SDP", caching),
    ("mot17_sub_train", "MOT17_sub", "train", "SDP", caching),
    ("mot17_sub_val", "MOT17_sub", "val", "SDP", caching),
    ## MOT20
    ("mot20_train", "MOT20", "train", None, caching),
    ("mot20_test", "MOT20", "test", None, caching),
    ("mot20_sub_train", "MOT20_sub", "train", None, caching),
    ("mot20_sub_val", "MOT20_sub", "val", None, caching),
    ## HIEVE
    ('hieve_sub_train', 'hieve_sub', 'train', None, caching),
    ('hieve_sub_val', 'hieve_sub', 'val', None, caching),
    ('hieve_train', 'hieve', 'train', None, caching),
    ('hieve_test', 'hieve', 'test', None, caching),
]

def register_all_mot(root, cfg=None):
    min_vis = 0.0
    for name, dataset_dir, split, detector, caching in SPLITS:
        if cfg is not None:
            min_vis = cfg.INPUT.TRAIN_MIN_VISIBILITY if split == 'train' else cfg.TEST.EVAL_MIN_VISIBILITY
        register_mot(name, os.path.join(root, dataset_dir), split, detector, min_vis, caching)


def register_mot(name, root_dir, split, detector, min_vis, caching):
    DatasetCatalog.register(name, lambda: load_mot_dataset_dicts(name, root_dir, split, detector, min_vis, caching))
    MetadataCatalog.get(name).set(
        thing_classes=CLASS_NAMES, thing_colors=CLASS_COLORS, root_dir=root_dir, split=split, detector=detector,
        evaluator_type="mot",
    )
