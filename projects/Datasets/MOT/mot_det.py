import glob
import json
import logging
import os
import pickle

import numpy as np

from detectron2.data import MetadataCatalog
from detectron2.structures import BoxMode
from detectron2.utils.file_io import PathManager
from detectron2.utils.comm import is_main_process

VALID_CLASSES = [
    1, 2, 7,
]


def load_mot_dataset_dicts(dataset_name, root_dir: str, split: str, detector: str = None, min_vis: float = 0.0, caching: bool = False):
    root_dir_name = root_dir.split('/')[-1]
    split_dir = os.path.join(root_dir, split)
    # seq_dirs = sorted(glob.glob(os.path.join(split_dir, "*")))
    seq_dirs = sorted(glob.glob(f'{split_dir}/*/'))
    logger = logging.getLogger(__name__)
    if detector is not None:
        seq_dirs = [sd for sd in seq_dirs if detector in sd]

    cache_file = os.path.join(root_dir, "{}_{}.pkl".format(root_dir_name, split))
    if caching and os.path.isfile(cache_file):
        with open(cache_file, 'rb') as f:
            cache_dataset = pickle.load(f)
            dataset_dicts = cache_dataset['dataset_dicts']
        return dataset_dicts

    dataset_dicts, ratio_invisible_dict = list(), dict()
    image_id = 0
    anno_id = 0
    training = False
    for seq_dir in seq_dirs:
        seq_file = os.path.join(seq_dir, "seqinfo.ini")
        
        seq_meta = open(seq_file).read()

        seq_name = str(seq_meta[seq_meta.find('name=') + 5:seq_meta.find('\nimDir')])
        seq_img_dir_name = str(seq_meta[seq_meta.find('imDir=') + 6:seq_meta.find('\nframeRate')])
        seq_fps = int(seq_meta[seq_meta.find('frameRate=') + 10:seq_meta.find('\nseqLength')])
        seq_length = int(seq_meta[seq_meta.find('seqLength=') + 10:seq_meta.find('\nimWidth')])
        seq_img_w = int(seq_meta[seq_meta.find('imWidth=') + 8:seq_meta.find('\nimHeight')])
        seq_img_h = int(seq_meta[seq_meta.find('imHeight=') + 9:seq_meta.find('\nimExt')])
        seq_img_file_ext = str(seq_meta[seq_meta.find('imExt=') + 6:seq_meta.find('\n\n')])

        training = split == "train" or split == "val"
        seq_range = range(0, seq_length)
        if training:
            ann_file = os.path.join(seq_dir, "gt", "gt.txt")
            seq_ann = np.loadtxt(ann_file, dtype=np.float64, delimiter=',')
            seq_range = range(int(min(seq_ann[:, 0])) - 1, int(max(seq_ann[:, 0])))

        num_total_obj, num_invisible_obj = 0, 0
        for img_idx in seq_range:
            img_ann_std = dict()
            img_ann_std["file_name"] = os.path.join(root_dir, split, seq_name , seq_img_dir_name, \
                                        "{:6d}".format(img_idx+1).replace(" ","0",6) + seq_img_file_ext)
            img_ann_std["image_id"] = image_id # None for now. Fill in this field according to the usage later
            img_ann_std["width"] = seq_img_w
            img_ann_std["height"] = seq_img_h
            img_ann_std["sequence_name"] = seq_name
            img_ann_std["seq_fps"] = seq_fps
            img_ann_std['frame_idx'] = img_idx

            image_id += 1

            if training:
                obj_index = (seq_ann[:,0] == img_idx+1)
                img_ann_mot = seq_ann[obj_index]

                instances = list()
                for obj_ann in img_ann_mot:
                    if '15' in root_dir_name or 'hieve' in root_dir_name:
                        frm_idx, obj_id, x, y, w, h, confidence, _, _, _ = obj_ann
                        visibility = -1
                        if confidence != 1:
                            continue
                    else:
                        frm_idx, obj_id, x, y, w, h, confidence, category, visibility = obj_ann
                        if confidence != 1 or category != 1:
                            continue
                    if visibility >= 0 and visibility < min_vis:
                        num_invisible_obj += 1
                        num_total_obj += 1
                        continue
                    obj_id = int(obj_id)
                    x -= 1
                    y -= 1
                    bbox = [x, y, w, h]

                    obj_dict = {
                        "id": anno_id,
                        "category_id": 0, # constant 0 since MOT dataset uses single category (i.e., pedestrian)
                        "bbox": bbox,
                        "bbox_mode": BoxMode.XYWH_ABS,
                        "track_id": obj_id,
                        "confidence": confidence,
                        "visibility": visibility,
                    }
                    anno_id += 1
                    instances.append(obj_dict)
                    num_total_obj += 1
                img_ann_std["annotations"] = instances

            dataset_dicts.append(img_ann_std)
        if training:
            ratio_invisible_dict.update({seq_name: 100*num_invisible_obj/num_total_obj})
        logger.info("Dataset '{} / {}' is registered".format(seq_name, root_dir_name))

    if training:
        if is_main_process():
            for seq_name, ratio in ratio_invisible_dict.items():
                logger.info(f"Ratio of invisible (thresh = {min_vis}) objs {seq_name} - {ratio:.2f}")
        json_path = os.path.join(root_dir, "{}.json".format(split))

        if not os.path.exists(json_path):
            results = convert_mot2coco(dataset_name, dataset_dicts)
            logger.info("Saving {} annotations of coco format to {}".format(dataset_name, json_path))
            with PathManager.open(json_path, "w") as f:
                f.write(json.dumps(results))
                f.flush()
        else:
            logger.info("{} annotations of coco format already exist".format(dataset_name, json_path))

        if caching:
            with open(cache_file, 'wb') as f:
                cache_dataset = {'dataset_dicts': dataset_dicts}
                pickle.dump(cache_dataset, f, pickle.HIGHEST_PROTOCOL)

    return dataset_dicts


def convert_mot2coco(dataset_name, dataset_dicts):
    meta = MetadataCatalog.get(dataset_name)
    thing_classes = meta.thing_classes
    has_instances = 'annotations' in dataset_dicts[0]

    images = []
    annotations = []
    for i, data in enumerate(dataset_dicts):
        images.append({
            'file_name': data['file_name'],
            'id': data['image_id'],
            'height': int(data['height']),
            'width': int(data['width']),
        })
        if has_instances:
            for anno in data['annotations']:
                xywh = [int(coord) for coord in anno['bbox']]
                annotations.append({
                    'id': anno['id'],
                    'category_id': anno['category_id'],
                    'image_id': i,
                    'bbox': xywh,
                    'area': xywh[2] * xywh[3],
                    'iscrowd': 0,
                })

    categories = []
    for i, thing_class in enumerate(thing_classes):
        categories.append({
            "supercategory": thing_class,
            "id": i,
            "name": thing_classes,
        })

    results = {'images': images, 'categories': categories}
    if has_instances:
        results.update({'annotations': annotations})
    return results
