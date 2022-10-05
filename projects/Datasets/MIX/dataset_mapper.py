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
from projects.Datasets.MOT.dataset_mapper import build_transform_gen


class DefaultMixDatasetMapper:

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

        self.gen_fake_img = RandomAffine(cfg.INPUT.SCALE, cfg.INPUT.SHIFT, cfg.INPUT.DEGREE, prob=1.0)
        self.img_format = cfg.INPUT.FORMAT
        self.is_train = is_train

    def __call__(self, dataset_dicts):
        if type(dataset_dicts).__name__ == 'tuple' and len(dataset_dicts) == 2:
            dataset_dicts, gen_fake_img_flag = dataset_dicts[0], dataset_dicts[1]
        dataset_dicts = copy.deepcopy(dataset_dicts)

        transforms = None
        crop_on = True if (self.crop_gen is not None) and (np.random.rand() > 0.5) else False
        for dataset_dict in dataset_dicts:
            image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
            utils.check_image_size(dataset_dict, image)

            if transforms is None:
                image, transforms = T.apply_transform_gens((self.crop_gen if crop_on else []) + self.tfm_gens, image)
            else:
                if gen_fake_img_flag:
                    transforms = transforms.transforms + [self.gen_fake_img]
                image, transforms = T.apply_transform_gens(transforms, image)

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
