from detectron2.config import CfgNode as CN


def add_centernet_config(cfg: CN):
    _C = cfg

    _C.MODEL.IDENTITY_ON = False

    # ---------------------------------------------------------------------------- #
    # ResNet
    # ---------------------------------------------------------------------------- #
    cfg.MODEL.RESNETS.PRETRAINED = False

    # ---------------------------------------------------------------------------- #
    # DLA
    # ---------------------------------------------------------------------------- #
    _C.MODEL.DLA = CN()
    _C.MODEL.DLA.LEVELS = [1, 1, 1, 2, 2, 1]
    _C.MODEL.DLA.CHANNELS = [16, 32, 64, 128, 256, 512]
    _C.MODEL.DLA.DEPTH = 34
    _C.MODEL.DLA.LAST_LEVEL = 5
    _C.MODEL.DLA.OUT_CHANNEL = 0
    _C.MODEL.DLA.PRETRAINED = False

    # ---------------------------------------------------------------------------- #
    # CenterNet
    # ---------------------------------------------------------------------------- #
    _C.MODEL.CENTERNET = CN()

    _C.MODEL.CENTERNET.SCORE_THRESH_TEST = 0.01

    _C.MODEL.CENTERNET.UPSAMPLE_LAYER = CN()
    _C.MODEL.CENTERNET.UPSAMPLE_LAYER.NAME = "DLASegUp"

    # output stride. Currently only supports 4
    _C.MODEL.CENTERNET.DOWN_RATIO = 4
    # This is the number of foreground classes.
    _C.MODEL.CENTERNET.NUM_CLASSES = 80
    # max number of output objects
    _C.MODEL.CENTERNET.MAX_PER_IMAGE = 128
    _C.MODEL.CENTERNET.INPUT_ALIGN = True

    _C.MODEL.CENTERNET.HEAD = CN()
    _C.MODEL.CENTERNET.HEAD.NAME = "DLASegUp"
    _C.MODEL.CENTERNET.HEAD.INTER_CHANNELS = 256
    _C.MODEL.CENTERNET.HEAD.FINAL_KERNEL = 1
    _C.MODEL.CENTERNET.HEAD.REID_DIM = 512
    _C.MODEL.CENTERNET.HEAD.REID_ASPP_ON = False

    _C.MODEL.CENTERNET.NUM_STACKS = 1
    _C.MODEL.CENTERNET.WH_WEIGHT = 0.1
    _C.MODEL.CENTERNET.REG_OFFSET = True
    _C.MODEL.CENTERNET.OFF_WEIGHT = 1
    _C.MODEL.CENTERNET.ID_WEIGHT = 1
    _C.MODEL.CENTERNET.HM_WEIGHT = 1

    _C.MODEL.CENTERNET.OUTPUT_SIZE = (512, 512)
    _C.MODEL.CENTERNET.AFFINE = CN()
    _C.MODEL.CENTERNET.AFFINE.BORDER = 128
    _C.MODEL.CENTERNET.AFFINE.RANDOM_AUG = True
    _C.MODEL.CENTERNET.BRIGHTNESS = CN()
    _C.MODEL.CENTERNET.BRIGHTNESS.INTENSITY_MIN = 0.6
    _C.MODEL.CENTERNET.BRIGHTNESS.INTENSITY_MAX = 1.4
    _C.MODEL.CENTERNET.CONTRAST = CN()
    _C.MODEL.CENTERNET.CONTRAST.INTENSITY_MIN = 0.6
    _C.MODEL.CENTERNET.CONTRAST.INTENSITY_MAX = 1.4
    _C.MODEL.CENTERNET.SATURATION = CN()
    _C.MODEL.CENTERNET.SATURATION.INTENSITY_MIN = 0.6
    _C.MODEL.CENTERNET.SATURATION.INTENSITY_MAX = 1.4
    _C.MODEL.CENTERNET.LIGHTING = CN()
    _C.MODEL.CENTERNET.LIGHTING.SCALE = 1
    _C.MODEL.CENTERNET.MIN_OVERLAP = 0.7
    _C.MODEL.CENTERNET.HM_SIZE = (128, 128)

    _C.MODEL.CENTERNET.TLBR_FLAG = False
    _C.MODEL.CENTERNET.ID_LOSS = True

    # Default setting of deformable conv is based on ResNet 50
    _C.MODEL.CENTERNET.DECONV_CHANNEL = [2048, 256, 128, 64]
    _C.MODEL.CENTERNET.DECONV_KERNEL = [4, 4, 4]
    _C.MODEL.CENTERNET.MODULATE_DEFORM = True
    _C.MODEL.CENTERNET.BIAS_VALUE = -2.19

    _C.INPUT.NORM_BY_MEAN_STD_FLAG = True

    ## Oracle Analysis
    _C.MODEL.CENTERNET.INFERENCE_GT = False

    ## det & id loss weight
    _C.MODEL.LOSS = CN()
    _C.MODEL.LOSS.AUTO_WEIGHT_FLAG = True
    _C.MODEL.LOSS.DET_WEIGHT = 1.0
    _C.MODEL.LOSS.ID_WEIGHT = 1.0