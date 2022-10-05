from detectron2.config import CfgNode as CN


def add_random_affine_config(cfg: CN):
    """
    Add config for MOT dataset.
    """
    _C = cfg

    # RandomAffine
    cfg.INPUT.RANDOM_AFFINE = False
    cfg.INPUT.SCALE = 0.05
    cfg.INPUT.SHIFT = 0.05
    cfg.INPUT.RANDOM_HUE = False
    cfg.INPUT.DEGREE = 0.0
