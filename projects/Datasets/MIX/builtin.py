import sys

import logging
import os
import os.path as osp
from PIL import Image
import numpy as np
import pickle
from detectron2.utils.logger import log_first_n

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode, Instances, Boxes
from projects.Datasets.MOT.vis.simple_visualzation_demo import SimpleVisualizationDemo

import matplotlib.pyplot as plt
from detectron2.utils.comm import is_main_process


CLASS_NAMES = [
    "pedestrian",
]

CLASS_COLORS = [
    (0, 255, 0),
]

HAS_IDS_FLAG_DICT = {
    'mot15': True,
    'mot16': True,
    'mot17': True,
    'mot17_sub': True,
    'mot20': True,
    'mot20_sub': True,
    'caltech': True,
    'citypersons': False,
    'cuhksysu': True,
    'prw': True,
    'eth': False,
    # 'crowdhuman': False,
    'crowdhuman': True, # FairMOT provided dataset already filled id
    'hieve': True,
    'hieve_sub': True,
}

IS_CONSECUTIVE_FLAG_DICT = {
    'mot15': True,
    'mot16': True,
    'mot17': True,
    'mot17_sub': True,
    'mot20': True,
    'mot20_sub': True,
    'caltech': True,
    'citypersons': False,
    'cuhksysu': False,
    # 'prw': True, # it is too sparse to say it is consecutive
    'prw': False,
    'eth': True,
    'crowdhuman': False,
    'hieve': True,
    'hieve_sub': True,
}

DENSE_VIDEO_FLAG_DICT = {
    'mot15': True,
    'mot16': True,
    'mot17': True,
    'mot17_sub': True,
    'mot20': True,
    'mot20_sub': True,
    # 'caltech': True, # motion is very high - do we consider False
    'caltech': False, # motion is very high - do we consider False
    'citypersons': False,
    'cuhksysu': False,
    'prw': False,
    'eth': True,
    'crowdhuman': False,
    'hieve': True,
    'hieve_sub': True,
}

caching = False

SPLITS = [
    ('mix_mot17_train', 'mot17', 'train', caching),
    ('mix_mot17_sub_train', 'mot17_sub', 'train', caching),
    ('mix_mot20_train', 'mot20', 'train', caching),
    ('mix_mot20_sub_train', 'mot20_sub', 'train', caching),
    ('mix_crowdhuman_train', 'crowdhuman', 'train', caching),
    ('mix_hieve_train', 'hieve', 'train', caching),
    ('mix_hieve_sub_train', 'hieve_sub', 'train', caching),
]


def vis_result(file_name, instances):
    demo = SimpleVisualizationDemo()
    vis_output = demo.run_on_image(file_name, instances)
    return vis_output.get_image()

def order_tid(tid_mapper, tid_list):
    return [tid_mapper[tid] for tid in tid_list]

def get_seq_name_from_mix_dataset(filename):
    filename = filename.lower()
    if 'mot' in filename:
        seq_name = filename.split('/')[-3]
    elif 'cuhksysu' in filename:
        seq_name = 'cuhksysu_' + filename.split('/')[-1].split('.')[0]
    elif 'eth' in filename:
        seq_name = filename.split('/')[-3]
    elif 'prw' in filename:
        seq_name = 'prw_' + filename.split('/')[-1].split('_')[0]
    elif 'caltech' in filename:
        seq_name = 'caltech_' + '_'.join(filename.split('/')[-1].split('_')[0:2])
    elif 'cityscapes' in filename:
        seq_name = 'citypersons_' + filename.split('/')[-1].split('.')[0]
    elif 'crowdhuman' in filename:
        seq_name = 'crowdhuman_' + filename.split('/')[-1].split('.')[0]
    elif 'hieve' in filename:
        seq_name = filename.split('/')[-3]
    return seq_name

def get_frame_idx_from_mix_dataset(dataset_name, filename):
    is_consecutive = IS_CONSECUTIVE_FLAG_DICT[dataset_name]
    if not is_consecutive:
        return 0

    filename = filename.lower()
    if 'mot' in filename or 'hieve' in filename:
        frame_idx = int(filename.split('/')[-1].split('.')[0]) - 1
    elif 'eth' in filename:
        frame_idx = int(filename.split('_')[-2])
    elif 'prw' in filename:
        frame_idx = int(filename.split('_')[-1].split('.')[0])
    elif 'caltech' in filename:
        frame_idx = int(filename.split('_')[-1].split('.')[0])
    else:
        raise Exception("{} is supposed to be not consecutive dataset. Therefore, no logic to get frame_idx".format(dataset_name))
    return frame_idx

def register_mix_tgt(root_dir, cfg):
    train_tgt_mix = cfg.DATASETS.TRAIN
    test_tgt_mix = tuple(t for t in cfg.DATASETS.TEST if 'mix' in t)
    dataset_names = train_tgt_mix + test_tgt_mix

    fill_id_flag = cfg.DATASETS.FILL_ID_FLAG
    dense_sampling_datasets = cfg.DATASETS.DENSE_SAMPLING_DATASETS
    sparse_sampling_datasets = cfg.DATASETS.SPARSE_SAMPLING_DATASETS
    train_min_vis = cfg.INPUT.TRAIN_MIN_VISIBILITY
    eval_min_vis = cfg.TEST.EVAL_MIN_VISIBILITY

    for reg_dataset_name, dataset_name, split, caching in SPLITS:
        if reg_dataset_name in dataset_names:
            min_vis = train_min_vis if split == 'train' else eval_min_vis
            register_mix(reg_dataset_name, dataset_name, split, root_dir, fill_id_flag, dense_sampling_datasets, sparse_sampling_datasets, min_vis, caching)

def register_mix(reg_dataset_name, dataset_name, split, root_dir, fill_id_flag, dense_sampling_datasets, sparse_sampling_datasets, min_vis, caching):
    has_ids_flag = HAS_IDS_FLAG_DICT[dataset_name]
    is_consecutive_flag = IS_CONSECUTIVE_FLAG_DICT[dataset_name]
    dense_video_flag = DENSE_VIDEO_FLAG_DICT[dataset_name]
    DatasetCatalog.register(reg_dataset_name, lambda: load_mix_dataset_dicts(dataset_name, split, root_dir, fill_id_flag,
                                                   dense_sampling_datasets, sparse_sampling_datasets, min_vis, caching))
    MetadataCatalog.get(reg_dataset_name).set(
        thing_classes=CLASS_NAMES, thing_colors=CLASS_COLORS, split=split, evaluator_type="mot",
        root_dir=osp.join(root_dir, dataset_name.upper()),
        has_ids_flag=has_ids_flag, is_consecutive_flag=is_consecutive_flag, dense_video_flag=dense_video_flag)

def load_mix_dataset_dicts(dataset_name, split, root_dir, fill_id_flag, dense_sampling_datasets, sparse_sampling_datasets, min_vis, caching):
    dataset_root = osp.join(root_dir, 'MIX')
    datalist_file_name = dataset_name + '.' + split
    datalist_file_path = osp.join(dataset_root, 'mix_data_list', datalist_file_name)
    has_ids_flag = HAS_IDS_FLAG_DICT[dataset_name]
    is_consecutive_flag = IS_CONSECUTIVE_FLAG_DICT[dataset_name]
    dense_video_flag = DENSE_VIDEO_FLAG_DICT[dataset_name]
    # fill_id_condition = not (dataset_name.replace('sub', '') in dense_sampling_datasets or dataset_name in sparse_sampling_datasets)
    fill_id_condition = not (any([d in dataset_name for d in [*dense_sampling_datasets, *sparse_sampling_datasets]]))

    fill_id_flag = fill_id_flag and fill_id_condition

    cache_file = os.path.join(dataset_root, 'MIX', "{}_{}.pkl".format(dataset_name, split))
    if caching and os.path.isfile(cache_file):
        with open(cache_file, 'rb') as f:
            cache_dataset = pickle.load(f)
            dataset_dicts = cache_dataset['dataset_dicts']
            obj_id_set_per_seq = cache_dataset['obj_id_set_per_seq']
            bbox_stat_per_seq = cache_dataset['bbox_stat_per_seq']
        return dataset_dicts, obj_id_set_per_seq, bbox_stat_per_seq

    with open(datalist_file_path, 'r') as fp:
        img_file_list = fp.readlines()

    img_file_path_list = [osp.join(dataset_root, x.strip()) for x in img_file_list]
    img_file_path_list = list(filter(lambda x: len(x) > 0, img_file_path_list))
    label_file_path_list = [x.replace('images', 'labels_with_ids').replace('.png', '.txt').replace('.jpg', '.txt') for x in img_file_path_list]

    dataset_dicts = list()
    image_id, anno_id, invalid_num = 0, 0, 0
    logger = logging.getLogger(__name__)
    logger.info("Build dataset dict from {}".format(datalist_file_name))
    ## suppose only train mode is supported in mix dataset
    obj_id_set, obj_id_set_per_seq, num_bbox_per_seq, num_bbox_below_vis_per_seq_list = set(), dict(), dict(), dict()
    for idx, img_file_path in enumerate(img_file_path_list):
        label_file_path = label_file_path_list[idx]
        W, H = Image.open(img_file_path).size
        seq_name = get_seq_name_from_mix_dataset(img_file_list[idx])
        if seq_name not in obj_id_set_per_seq:
            obj_id_set_per_seq[seq_name] = set()
        if seq_name not in num_bbox_per_seq:
            num_bbox_per_seq[seq_name] = []
            num_bbox_below_vis_per_seq_list[seq_name] = []
        img_ann_std = dict()
        img_ann_std["file_name"] = img_file_path
        img_ann_std["image_id"] = image_id
        img_ann_std["width"] = W
        img_ann_std["height"] = H
        img_ann_std["sequence_name"] = seq_name
        img_ann_std["seq_fps"] = None
        img_ann_std['frame_idx'] = get_frame_idx_from_mix_dataset(dataset_name, img_file_list[idx])
        image_id += 1

        if os.path.isfile(label_file_path):
            if 'mot17' in dataset_name or 'crowdhuman' in dataset_name:
                labels = np.loadtxt(label_file_path, dtype=np.float32).reshape(-1, 7)  # in shape [#objs, 6 or 7]
            else:
                labels = np.loadtxt(label_file_path, dtype=np.float32).reshape(-1, 6)
            if len(labels) == 0:
                labels = np.array([])
            else:
                ## normalize relative [ctx cty w h] to absolute [tlx tly w h]
                ## skip x -= 1 and y -= 1
                labels[:, [2, 4]] *= W
                labels[:, [3, 5]] *= H
                labels[:, 2] -= labels[:, 4] / 2
                labels[:, 3] -= labels[:, 5] / 2
        else:
            labels = np.array([])

        instances = list()
        label_num = 0
        label_num_below_vis = 0
        if len(labels) > 0:
            for label in labels:
                bbox = label[2:6].tolist()
                obj_id = int(label[1]) if not fill_id_flag else anno_id
                visibility = label[6] if len(label) == 7 else 1.0
                if visibility < min_vis:
                    label_num_below_vis += 1
                    continue
                obj_dict = {
                    "id": anno_id,
                    "category_id": 0,  # constant 0 since MOT dataset uses single category (i.e., pedestrian)
                    "bbox": bbox,
                    "bbox_mode": BoxMode.XYWH_ABS,
                    "track_id": obj_id,
                    "confidence": 1.0,
                    "visibility": visibility,
                }
                anno_id += 1
                instances.append(obj_dict)
                if obj_id != -1:
                    obj_id_set.add(obj_id)
                    obj_id_set_per_seq[seq_name].add(obj_id)
                else:
                    invalid_num += 1
                label_num += 1
        num_bbox_per_seq[seq_name].append(label_num)
        num_bbox_below_vis_per_seq_list[seq_name].append(label_num_below_vis)
        img_ann_std["annotations"] = instances
        dataset_dicts.append(img_ann_std)
    obj_id_set_per_seq['total'] = obj_id_set
    obj_id_set_per_seq['num_invalid'] = invalid_num
    max_bbox_per_seq = {k: max(v) for k, v in num_bbox_per_seq.items()}
    sum_bbox_per_seq = {k: sum(v) for k, v in num_bbox_per_seq.items()}
    num_img_per_seq = {k: len(v) for k, v in num_bbox_per_seq.items()}
    num_bbox_below_vis_per_seq = {k: sum(v) for k, v in num_bbox_below_vis_per_seq_list.items()}
    ratio_bbox_below_vis_per_seq = {
        k: 100 * num_bbox_below_vis_per_seq[k] / (num_bbox_below_vis_per_seq[k] + sum_bbox_per_seq[k]) for k in
        num_bbox_below_vis_per_seq.keys()}
    avg_bbox_per_seq = {k: v / num_img_per_seq[k] for k, v in sum_bbox_per_seq.items()}
    bbox_stat_per_seq = {'max': max_bbox_per_seq, 'avg': avg_bbox_per_seq, 'sum': sum_bbox_per_seq,
                         'num_img': num_img_per_seq,
                         'ratio_boxes_invisible': ratio_bbox_below_vis_per_seq, 'visibility_thresh': min_vis}
    if not fill_id_flag:
        ## mapping object id in-order if it is loaded from GT ID which may not be in-ordered
        tid_mapper = {tid: idx for idx, tid in enumerate(list(obj_id_set_per_seq['total']))}

        ## update object id set
        new_obj_id_set_per_seq = {}
        for seq_name in obj_id_set_per_seq.keys():
            new_obj_id_set = obj_id_set_per_seq[seq_name]
            if seq_name != 'num_invalid':
                new_obj_id_set = set(order_tid(tid_mapper, new_obj_id_set))
            new_obj_id_set_per_seq[seq_name] = new_obj_id_set
        obj_id_set_per_seq = new_obj_id_set_per_seq

        ## update dataset annotation id
        for d in dataset_dicts:
            for anno in d['annotations']:
                anno['track_id'] = tid_mapper[anno['track_id']]

    if is_main_process():
        logger.info(f"% bboxes below visibility threshold {min_vis}")
        for k, v in ratio_bbox_below_vis_per_seq.items():
            logger.info(f"{k} : {v:.2f}")
    if caching:
        with open(cache_file, 'wb') as f:
            cache_dataset = {'dataset_dicts': dataset_dicts, 'obj_id_set_per_seq': obj_id_set_per_seq, 'bbox_stat_per_seq': bbox_stat_per_seq}
            pickle.dump(cache_dataset, f, pickle.HIGHEST_PROTOCOL)

    return dataset_dicts, obj_id_set_per_seq, bbox_stat_per_seq
