from detectron2.config import CfgNode as CN


def add_epoch_trainer_config(cfg: CN):
    _C = cfg
    _C.SOLVER.MAX_EPOCH = 0

    # ---------------------------------------------------------------------------- #
    # OPTIMIZER
    # ---------------------------------------------------------------------------- #
    # options: "SGD, "Adam"
    _C.SOLVER.OPTIMIZER = "SGD"
