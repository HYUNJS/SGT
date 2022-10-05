import cv2
import copy
import logging
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection

from detectron2.data import transforms as T
from detectron2.data import detection_utils as utils
from projects.Datasets.Transforms.augmentation import LetterBox, RandomHue, FairAffine
from projects.Datasets.MOT.detection_utils import transform_instance_annotations, annotations_to_instances


def build_transform_gen(cfg, is_train):
    logger = logging.getLogger(__name__)
    tfm_gens = []
    if is_train:
        tfm_gens.append(RandomHue(cfg.INPUT.FORMAT, prob=1.))
        tfm_gens.append(LetterBox(cfg.MODEL.CENTERNET.OUTPUT_SIZE))
        tfm_gens.append(FairAffine(degrees=(-5, 5), translate=(.1, .1), scale=(.5, 1.2), shear=(-2, 2), border_value=(127.5, 127.5, 127.5)))
        tfm_gens.append(T.RandomFlip())
        logger.info("TransformGens used in training: " + str(tfm_gens))
    else:
        tfm_gens.append(LetterBox(cfg.MODEL.CENTERNET.OUTPUT_SIZE))
        logger.info("TransformGens used in inference: " + str(tfm_gens))
    return tfm_gens

class MOTFairMOTDatasetMapper:

    def __init__(self, cfg, is_train=True):
        if cfg.INPUT.CROP.ENABLED and is_train:
            self.crop_gen = T.RandomCrop(cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE)
        else:
            self.crop_gen = None

        self.tfm_gens = build_transform_gen(cfg, is_train)
        logging.getLogger(__name__).info(
            "Full TransformGens used: {}, crop: {}".format(str(self.tfm_gens), str(self.crop_gen))
        )

        self.img_format = cfg.INPUT.FORMAT
        self.clip_by_image = cfg.INPUT.CLIP_BY_IMAGE
        self.filter_out_image = cfg.INPUT.FILTER_OUT_IMAGE
        self.vis_input_frame_flag = False
        self.min_side = cfg.INPUT.MIN_SIDE
        self.is_train = is_train
        # self.gen_fake_img = FairAffine(degrees=(-5, 5), translate=(.1, .1), scale=(.8, 1.2), shear=(-2, 2), border_value=(127.5, 127.5, 127.5))
        self.gen_fake_img = FairAffine(degrees=(-2, 2), translate=(.05, .05), scale=(.9, 1.1), shear=(-1, 1), border_value=(127.5, 127.5, 127.5))

    def __call__(self, dataset_dicts):
        gen_fake_img_flag = False
        if type(dataset_dicts).__name__ == 'tuple' and len(dataset_dicts) == 2:
            dataset_dicts, gen_fake_img_flag = dataset_dicts[0], dataset_dicts[1]
        dataset_dicts = copy.deepcopy(dataset_dicts)
        crop_on = True if self.crop_gen else False
        transforms = None

        for dataset_dict in dataset_dicts: # iterate number of frames
            image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
            utils.check_image_size(dataset_dict, image)

            if transforms is None: # first frame
                image, transforms = T.apply_transform_gens((self.crop_gen if crop_on else []) + self.tfm_gens, image)
            else: # next frames
                if gen_fake_img_flag:
                    transforms = transforms.transforms + [self.gen_fake_img]
                image, transforms = T.apply_transform_gens(transforms, image)

            image_shape = image.shape[:2]  # h, w
            dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
            dataset_dict["transforms"] = transforms

            if "annotations" in dataset_dict:
                annos = []
                if self.vis_input_frame_flag:
                    ori_annos = copy.deepcopy(dataset_dict["annotations"])
                    self.plot_ori_anno_vs_filtered_anno(image, annos, ori_annos, transforms)
                for obj in dataset_dict.pop("annotations"):
                    transformed_obj = transform_instance_annotations(obj, transforms, image_shape,
                                     clip_by_image=self.clip_by_image and self.is_train, filter_out_image=self.filter_out_image and self.is_train)
                    if transformed_obj is not None:
                        annos.append(transformed_obj)
                instances = annotations_to_instances(annos, image_shape)
                dataset_dict["instances"] = utils.filter_empty_instances(instances, box_threshold=self.min_side)

        if len(dataset_dicts) == 1:
            return dataset_dicts[0]
        else:
            return dataset_dicts

    def plot_ori_anno_vs_filtered_anno(self, image, annos, ori_annos, transforms):
        image_shape = image.shape[:2]  # h, w
        ori_transformed_annos = []
        ori_annos = copy.deepcopy(ori_annos)
        for obj in ori_annos:
            transformed_obj = transform_instance_annotations(obj, transforms, image_shape,
                                             clip_by_image=False, filter_out_image=False)
            if transformed_obj is not None:
                ori_transformed_annos.append(transformed_obj)
        patches, patches_ori = [], []
        for a in annos:
            tlx, tly, brx, bry = a['bbox']
            patches.append(Rectangle((tlx, tly), brx - tlx, bry - tly, color='red', linewidth=1, fill=False))
        for a in ori_transformed_annos:
            tlx, tly, brx, bry = a['bbox']
            patches_ori.append(Rectangle((tlx, tly), brx - tlx, bry - tly, color='blue', linewidth=1, fill=False))
        fig, axes = plt.subplots(2, 1)
        axes[0].imshow(image[:, :, ::-1])
        axes[1].imshow(image[:, :, ::-1])
        axes[0].add_collection(PatchCollection(patches, match_original=True))
        axes[1].add_collection(PatchCollection(patches_ori, match_original=True))
        axes[0].text(0, 40, '{}'.format(len(annos)), color='red', fontsize=8)
        axes[1].text(0, 40, '{}'.format(len(ori_annos)), color='blue', fontsize=8)
        axes[0].set_title('Real loader')
        axes[1].set_title('W/O clip and filter')
        plt.show()
        plt.close()