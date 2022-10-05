from detectron2.config import CfgNode as CN
from projects.Datasets.Transforms.config import add_random_affine_config


def add_mot_dataset_config(cfg: CN):
    """
    Add config for MOT dataset.
    """
    add_random_affine_config(cfg)

    _C = cfg

    # DataLoader
    cfg.DATALOADER.PARALLEL_BATCH = False
    # Options: InferenceSampler, OrderedSampler
    _C.DATALOADER.SAMPLER_TEST = "InferenceSampler"
    # In case of True, the total number of tracking objects is calculated,
    # and a unique ID is assigned to each object again.
    _C.DATALOADER.REASSIGN_ID = False

    # Total number of tracking objects
    _C.DATALOADER.NUM_IDS = 559

    # number of frames in one segment
    cfg.DATALOADER.NUM_SAMPLES_TRAIN = 1
    cfg.DATALOADER.NUM_SAMPLES_TEST = 1

    cfg.DATALOADER.PRE_SAMPLE_FLAG = False # Pre-sample the frames from the sequence instead of randomly sample in the dataloader dynamically
    cfg.DATALOADER.PRE_SAMPLE_INTERVAL = 10

    cfg.DATALOADER.MAX_FRAME_DIST_TRAIN = 1
    cfg.DATALOADER.MAX_FRAME_DIST_TEST = 1
    cfg.DATALOADER.MIN_FRAME_DIST_TRAIN = 1
    cfg.DATALOADER.MIN_FRAME_DIST_TEST = 1
    # if True, the dataloader will filter out images that have no pair in next frame at train time.
    cfg.DATALOADER.FILTER_PAIRLESS_ANNOTATIONS = False

    cfg.DATALOADER.SEED = -1

    cfg.DATALOADER.STREAM_SAMPLER = CN()
    cfg.DATALOADER.STREAM_SAMPLER.CLIP_LEN = 1
    # CLIP_INTERVAL = [CLIP_LEN - RANGE: CLIP_LEN + RANGE]
    cfg.DATALOADER.STREAM_SAMPLER.CLIP_RANGE = 0

    # TEST
    cfg.TEST.EVAL_DETECTION = False
    cfg.TEST.EVAL_TRACKING = True
    cfg.TEST.VIS_FLAG = False
    cfg.TEST.VIS_OUT_DIR = './output/vis'
    cfg.TEST.EVAL_FILTER_DET_FLAG = False
    cfg.TEST.EVAL_DET_BY_SEQ_FLAG = False
    cfg.TEST.EVAL_FILTER_DET_SCORE_THRESH = 0.5
    cfg.TEST.EVAL_FILTER_DET_SIZE_THRESH = 100
    cfg.TEST.EVAL_FILTER_DET_RATIO_THRESH = 1.6
    cfg.TEST.EVAL_MIN_VISIBILITY = 0.0

    cfg.INPUT.TRAIN_MIN_VISIBILITY = 0.0
    cfg.INPUT.CROP.MIN_SIZE_TRAIN = (400, 500, 600)
    cfg.INPUT.CLIP_BY_IMAGE = True
    cfg.INPUT.MIN_SIDE = 1e-5
