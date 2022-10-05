import copy
import logging
import numpy as np
import torch

from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from projects.Datasets.Transforms.augmentation import LetterBox, CenterAffine
from projects.Datasets.MOT.detection_utils import transform_instance_annotations, annotations_to_instances


def build_COCO_transform_gen(cfg, is_train):
    logger = logging.getLogger(__name__)
    tfm_gens = []
    param = cfg.MODEL.CENTERNET
    if is_train:
        tfm_gens.append(CenterAffine(param.AFFINE.BORDER, param.OUTPUT_SIZE, param.AFFINE.RANDOM_AUG))
        tfm_gens.append(T.RandomFlip())
        tfm_gens.append(T.RandomBrightness(param.BRIGHTNESS.INTENSITY_MIN, param.BRIGHTNESS.INTENSITY_MAX))
        tfm_gens.append(T.RandomContrast(param.CONTRAST.INTENSITY_MIN, param.CONTRAST.INTENSITY_MAX))
        tfm_gens.append(T.RandomSaturation(param.SATURATION.INTENSITY_MIN, param.SATURATION.INTENSITY_MAX))
        tfm_gens.append(T.RandomLighting(param.LIGHTING.SCALE))
        logger.info("TransformGens used in training: " + str(tfm_gens))
    return tfm_gens


def build_MOT_transform_gen(cfg, is_train):
    logger = logging.getLogger(__name__)
    tfm_gens = []
    if is_train:
        # tfm_gens.append(RandomHue())
        tfm_gens.append(LetterBox(cfg.MODEL.CENTERNET.OUTPUT_SIZE))
        # tfm_gens.append(T.RandomFlip())
        logger.info("TransformGens used in training: " + str(tfm_gens))
    else:
        tfm_gens.append(LetterBox(cfg.MODEL.CENTERNET.OUTPUT_SIZE))
        logger.info("TransformGens used in inference: " + str(tfm_gens))
    return tfm_gens


class COCOCenterNetDatasetMapper:

    def __init__(self, cfg, is_train=True):
        if cfg.INPUT.CROP.ENABLED and is_train:
            self.crop_gen = T.RandomCrop(cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE)
            logging.getLogger(__name__).info("CropGen used in training: " + str(self.crop_gen))
        else:
            self.crop_gen = None

        self.tfm_gens = build_COCO_transform_gen(cfg, is_train)

        # fmt: off
        self.img_format     = cfg.INPUT.FORMAT
        self.mask_on        = cfg.MODEL.MASK_ON
        self.mask_format    = cfg.INPUT.MASK_FORMAT
        self.keypoint_on    = cfg.MODEL.KEYPOINT_ON
        self.down_ratio     = cfg.MODEL.CENTERNET.DOWN_RATIO
        self.num_classes    = cfg.MODEL.CENTERNET.NUM_CLASSES
        self.load_proposals = cfg.MODEL.LOAD_PROPOSALS
        # fmt: on
        if self.keypoint_on and is_train:
            # Flip only makes sense in training
            self.keypoint_hflip_indices = utils.create_keypoint_hflip_indices(cfg.DATASETS.TRAIN)
        else:
            self.keypoint_hflip_indices = None

        if self.load_proposals:
            self.min_box_side_len = cfg.MODEL.PROPOSAL_GENERATOR.MIN_SIZE
            self.proposal_topk = (
                cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TRAIN
                if is_train
                else cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TEST
            )
        self.is_train = is_train

    def __call__(self, dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        utils.check_image_size(dataset_dict, image)

        image, transforms = T.apply_transform_gens(self.tfm_gens, image)

        image_shape = image.shape[:2]  # h, w
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))

        if not self.is_train:
            # USER: Modify this if you want to keep them for some reason.
            dataset_dict.pop("annotations", None)
            dataset_dict.pop("sem_seg_file_name", None)
            return dataset_dict

        if "annotations" in dataset_dict:
            # USER: Modify this if you want to keep them for some reason.
            for anno in dataset_dict["annotations"]:
                if not (self.mask_on or self.keypoint_on):
                    anno.pop("segmentation", None)
                if not self.keypoint_on:
                    anno.pop("keypoints", None)

            # USER: Implement additional transformations if you have other types of data
            annos = []
            for obj in dataset_dict.pop("annotations"):
                if obj.get("iscrowd", 0) == 0:
                    annos.append(utils.transform_instance_annotations(obj, transforms, image_shape))

            instances = utils.annotations_to_instances(
                annos, image_shape, mask_format=self.mask_format
            )
            # Create a tight bounding box from masks, useful when image is cropped
            dataset_dict["instances"] = utils.filter_empty_instances(instances)

        return dataset_dict


class MOTCenterNetDatasetMapper:

    def __init__(self, cfg, is_train=True):
        if cfg.INPUT.CROP.ENABLED and is_train:
            self.crop_gen = T.RandomCrop(cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE)
        else:
            self.crop_gen = None

        self.tfm_gens = build_MOT_transform_gen(cfg, is_train)
        logging.getLogger(__name__).info(
            "Full TransformGens used: {}, crop: {}".format(str(self.tfm_gens), str(self.crop_gen))
        )

        self.img_format = cfg.INPUT.FORMAT
        self.is_train = is_train

    def __call__(self, dataset_dicts):
        if type(dataset_dicts).__name__ == 'tuple' and len(dataset_dicts) == 2:
            dataset_dicts, gen_fake_img_flag = dataset_dicts[0], dataset_dicts[1]
        dataset_dicts = copy.deepcopy(dataset_dicts)
        crop_gen_on = True if self.crop_gen else False
        for dataset_dict in dataset_dicts:
            image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
            utils.check_image_size(dataset_dict, image)

            if "annotations" not in dataset_dict:
                image, transforms = T.apply_transform_gens(
                    ([self.crop_gen] if self.crop_gen else []) + self.tfm_gens, image
                )
            else:
                if crop_gen_on:
                    crop_tfm = utils.gen_crop_transform_with_instance(
                        self.crop_gen.get_crop_size(image.shape[:2]),
                        image.shape[:2],
                        np.random.choice(dataset_dict["annotations"]),
                    )
                    image = crop_tfm.apply_image(image)
                image, transforms = T.apply_transform_gens(self.tfm_gens, image)
                if crop_gen_on:
                    transforms = crop_tfm + transforms

            image_shape = image.shape[:2]  # h, w
            dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))

            if not self.is_train:
                dataset_dict.pop("annotations", None)
            else:
                if "annotations" in dataset_dict:
                    annos = []
                    for obj in dataset_dict.pop("annotations"):
                        annos.append(transform_instance_annotations(obj, transforms, image_shape))
                    instances = annotations_to_instances(annos, image_shape)
                    dataset_dict["instances"] = utils.filter_empty_instances(instances)

            if len(dataset_dicts) == 1:
                return dataset_dicts[0]
            else:
                return dataset_dicts
