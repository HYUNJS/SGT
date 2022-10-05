from detectron2.config import CfgNode as CN


def add_mix_dataset_config(cfg: CN):
    """
    Add config for mix dataset.
    """
    cfg.DATASETS.FILL_ID_FLAG = False
    cfg.DATASETS.DENSE_SAMPLING_DATASETS = ['mot15', 'mot17', 'mot20', 'hieve']
    cfg.DATASETS.SPARSE_SAMPLING_DATASETS = ['caltech'] # if none, input [] in config file
    cfg.INPUT.FILTER_OUT_IMAGE = False

    cfg.DATALOADER.MAX_FRAME_DIST_TRAIN_SPARSE = 3
    cfg.DATALOADER.MIN_FRAME_DIST_TRAIN_SPARSE = 1