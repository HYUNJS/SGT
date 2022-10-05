import torch
import copy
import logging
import numpy as np
import torch

from fvcore.transforms import HFlipTransform
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from projects.Datasets.Transforms.augmentation import RandomAffine, RandomHue
from projects.Datasets.MOT.detection_utils import transform_instance_annotations, annotations_to_instances


def build_transform_gen(cfg, is_train):
    if is_train:
        min_size = cfg.INPUT.MIN_SIZE_TRAIN
        max_size = cfg.INPUT.MAX_SIZE_TRAIN
        sample_style = cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING
    else:
        min_size = cfg.INPUT.MIN_SIZE_TEST
        max_size = cfg.INPUT.MAX_SIZE_TEST
        sample_style = "choice"
    if sample_style == "range":
        assert len(min_size) == 2, "more than 2 ({}) min_size(s) are provided for ranges".format(len(min_size))

    logger = logging.getLogger(__name__)
    tfm_gens = []
    if is_train and cfg.INPUT.RANDOM_FLIP != "none":
        tfm_gens.append(T.RandomFlip())
    if is_train and cfg.INPUT.RANDOM_AFFINE:
        tfm_gens.append(RandomAffine(cfg.INPUT.SCALE, cfg.INPUT.SHIFT, cfg.INPUT.DEGREE))
    if is_train and cfg.INPUT.RANDOM_HUE:
        tfm_gens.append(RandomHue(cfg.INPUT.FORMAT))
    tfm_gens.append(T.ResizeShortestEdge(min_size, max_size, sample_style))
    if is_train:
        logger.info("TransformGens used in training: " + str(tfm_gens))
    return tfm_gens


class DefaultMOTDatasetMapper:

    def __init__(self, cfg, is_train=True):
        if cfg.INPUT.CROP.ENABLED and is_train:
            self.crop_gen = [
                T.ResizeShortestEdge(cfg.INPUT.CROP.MIN_SIZE_TRAIN, sample_style="choice"),
                T.RandomCrop(cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE),
            ]
        else:
            self.crop_gen = None

        self.tfm_gens = build_transform_gen(cfg, is_train)
        logging.getLogger(__name__).info(
            "Full TransformGens used: {}, crop: {}".format(str(self.tfm_gens), str(self.crop_gen))
        )

        self.img_format = cfg.INPUT.FORMAT
        self.is_train = is_train

    def __call__(self, dataset_dicts):
        dataset_dicts = copy.deepcopy(dataset_dicts)

        transforms = None
        crop_on = True if (self.crop_gen is not None) and (np.random.rand() > 0.5) else False

        for dataset_dict in dataset_dicts:
            image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
            utils.check_image_size(dataset_dict, image)

            if transforms is None:
                image, transforms = T.apply_transform_gens((self.crop_gen if crop_on else []) + self.tfm_gens, image)
            else:
                for t in transforms.transforms:
                    image = t.apply_image(image)

            image_shape = image.shape[:2]  # h, w
            dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
            dataset_dict["transforms"] = transforms

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
