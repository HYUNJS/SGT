import sys

import itertools
import logging, torch
import numpy as np
import torch.utils.data as data
from multiprocessing import Pool

from tabulate import tabulate
from termcolor import colored
from multiprocessing import Manager
from detectron2.utils.env import seed_all_rng
from detectron2.utils.comm import get_world_size, is_main_process
from detectron2.utils.logger import log_first_n
from detectron2.data import DatasetCatalog, MetadataCatalog, samplers
from projects.Datasets.MIX.dataset_mapper import DefaultMixDatasetMapper
from projects.Datasets.MIX.common import MixDataset
from projects.Datasets.MOT.build import build_batch_data_loader, filter_images_without_annotations, filter_images_without_next_tracking_boxes, trivial_batch_collator
from projects.Datasets.MOT.distributed_sampler import InfInferenceSampler
from projects.Datasets.MOT.sequence_distributed_sampler import SeqInferenceSampler

__all__ = [
    "build_mix_train_loader",
]


def id_offset_adder(dataset_dicts, tid_offset, id_offset):
    for i in range(len(dataset_dicts)):
        for j in range(len(dataset_dicts[i]['annotations'])):
            dataset_dicts[i]['annotations'][j]['id'] += id_offset
            tid = dataset_dicts[i]['annotations'][j]['track_id']
            if tid != -1:
                dataset_dicts[i]['annotations'][j]['track_id'] += tid_offset
    return dataset_dicts


def print_mix_dataset_info(dataset_dicts, dataset_names, obj_id_set_list, bbox_stat_list):
    num_datasets = len(dataset_names)
    num_images, num_annos, num_bbox_per_img_list, num_id_list, num_invalid_list = [], [], [], [], []
    id_flag_list, consec_flag_list, dense_flag_list, max_bboxs, visibility_thresh_list = [], [], [], [], []
    ratio_boxes_invisible_list = []
    for i in range(num_datasets):
        num_image = len(dataset_dicts[i])
        assert num_image, "Dataset '{}' is empty!".format(dataset_names[i])
        # num_anno = dataset_dicts[i][-1]['annotations'][-1]['id'] + 1
        meta_data = MetadataCatalog.get(dataset_names[i])
        id_flag, consec_flag, dense_flag = meta_data.has_ids_flag, meta_data.is_consecutive_flag, meta_data.dense_video_flag
        id_flag, consec_flag, dense_flag = ["Y" if x else "N" for x in (id_flag, consec_flag, dense_flag)]
        num_anno = sum(bbox_stat_list[i]['sum'].values())
        max_bbox = max(bbox_stat_list[i]['max'].values())
        num_images.append(num_image)
        num_annos.append(num_anno)
        num_bbox_per_img_list.append("{:.2f}".format(num_anno / num_image))
        max_bboxs.append(max_bbox)
        num_id_list.append(len(obj_id_set_list[i]['total']))
        num_invalid_list.append(obj_id_set_list[i]['num_invalid'])
        id_flag_list.append(id_flag)
        consec_flag_list.append(consec_flag)
        dense_flag_list.append(dense_flag)
        ratio_boxes_invisible_list.append("{:.2f}".format(sum(bbox_stat_list[i]['ratio_boxes_invisible'].values()) / len(bbox_stat_list[i]['ratio_boxes_invisible'])))
        visibility_thresh_list.append(bbox_stat_list[i]['visibility_thresh'])
    print_data = [list(dataset_names), ["# image", *num_images], ["# bbox", *num_annos], ["max bbox", *max_bboxs],
                  ["# bbox/img", *num_bbox_per_img_list], ["# ids", *num_id_list], ["# invalid ids", *num_invalid_list],
                  ["has id?", *id_flag_list], ["consecutive frames?", *consec_flag_list], ["Dense video?", *dense_flag_list],
                  ["visibility threshold", *visibility_thresh_list], ["% invisible boxes", *ratio_boxes_invisible_list]]
    table = tabulate(print_data, headers='firstrow')
    txt = "[Mix Dataset Info]\n" + colored(table, "cyan")
    if is_main_process():
        log_first_n(logging.INFO, txt, key='message')


def reassign_id_mix(dataset_dicts_list, obj_id_set_list):
    num_dataset = len(obj_id_set_list)
    tid_nums = [max(obj_id_set_list[i]['total']) + 1 if len(obj_id_set_list[i]['total']) != 0 else 0 for i in range(num_dataset)]
    tid_offsets = np.array([0, *tid_nums]).cumsum()
    id_offsets = np.array([0, *[dataset_dicts_list[i][-1]['annotations'][-1]['id'] for i in range(num_dataset)][:-1]]).cumsum()
    with Pool(processes=num_dataset) as pool:
        dataset_dicts_list_updated = pool.starmap(id_offset_adder, zip(dataset_dicts_list, tid_offsets, id_offsets))
    return dataset_dicts_list_updated


def get_mix_dataset_dicts(dataset_names, dense_sampling_datasets=['mot15', 'mot17', 'mot20'], sparse_sampling_datasets=['caltech'], filter_empty=False, filter_pairless=False):
    assert len(dataset_names)

    dataset_dicts_list, obj_id_set_list, gen_fake_img_flag_list = [], [], []
    bbox_stat_list, dense_video_flag_list, tid_num_list = [], [], []
    for dataset_name in dataset_names:
        dataset_dicts, obj_id_set, bbox_stat = DatasetCatalog.get(dataset_name)
        meta_data = MetadataCatalog.get(dataset_name)
        gen_fake_img_flag = not (any([d in dataset_name for d in [*dense_sampling_datasets, *sparse_sampling_datasets]]))
        dense_video_flag = meta_data.dense_video_flag
        if "annotations" in dataset_dicts[0]:
            if filter_empty:
                dataset_dicts = filter_images_without_annotations(dataset_dicts)
            if filter_pairless:
                dataset_dicts = filter_images_without_next_tracking_boxes(dataset_dicts)
        dataset_dicts_list.append(dataset_dicts)
        obj_id_set_list.append(obj_id_set)
        bbox_stat_list.append(bbox_stat)
        gen_fake_img_flag_list.append(gen_fake_img_flag)
        dense_video_flag_list.append(dense_video_flag)
    has_instances = "annotations" in dataset_dicts_list[0][0]
    if has_instances:
        tid_num_list = [max(obj_id_set['total']) + 1 if len(obj_id_set['total']) != 0 else 0 for obj_id_set in obj_id_set_list]
        print_mix_dataset_info(dataset_dicts_list, dataset_names, obj_id_set_list, bbox_stat_list)
        dataset_dicts_list = reassign_id_mix(dataset_dicts_list, obj_id_set_list)

    dataset_dicts = dataset_dicts_list
    # required_meta_data = {'obj_id_set_list': obj_id_set_list, 'gen_fake_img_flag_list':gen_fake_img_flag_list, 'dense_video_flag_list': dense_video_flag_list}
    required_meta_data = {'gen_fake_img_flag_list': gen_fake_img_flag_list, 'dense_video_flag_list': dense_video_flag_list, 'tid_num_list': tid_num_list}
    return dataset_dicts, required_meta_data


def build_mix_train_loader(cfg, mapper=None):
    dataset_dicts, required_meta_data = get_mix_dataset_dicts(cfg.DATASETS.TRAIN,
                                        dense_sampling_datasets=cfg.DATASETS.DENSE_SAMPLING_DATASETS, sparse_sampling_datasets=cfg.DATASETS.SPARSE_SAMPLING_DATASETS,
                                        filter_empty=cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS, filter_pairless=cfg.DATALOADER.FILTER_PAIRLESS_ANNOTATIONS)
    cfg.DATALOADER.defrost()
    cfg.DATALOADER.NUM_IDS = sum(required_meta_data['tid_num_list']) + 1
    cfg.DATALOADER.freeze()
    sampler_name = cfg.DATALOADER.SAMPLER_TRAIN
    logger = logging.getLogger(__name__)
    logger.info("Using training sampler {}".format(sampler_name))
    assert sampler_name == "TrainingSampler", "Only TrainingSampler is supported in mix dataset"

    if mapper is None:
        mapper = DefaultMixDatasetMapper(cfg, is_train=True)

    seed = None if cfg.DATALOADER.SEED == -1 else cfg.DATALOADER.SEED
    dataset = MixDataset(cfg, dataset_dicts, required_meta_data, mapper, is_train=True)
    sampler = samplers.TrainingSampler(len(dataset), seed=seed)

    return build_batch_data_loader(
        dataset,
        sampler,
        cfg.SOLVER.IMS_PER_BATCH,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
    )

def build_mix_test_loader(cfg, dataset_name, mapper=None):
    dataset_dicts, required_meta_data = get_mix_dataset_dicts([dataset_name], filter_empty=cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS,
                                                            filter_pairless=cfg.DATALOADER.FILTER_PAIRLESS_ANNOTATIONS)
    if cfg.DATALOADER.PRE_SAMPLE_FLAG:
        raise Exception("Pre sample mode is not yet suppported in mix test dataloader")
    if mapper is None:
        mapper = DefaultMixDatasetMapper(cfg, is_train=False)
    dataset = MixDataset(cfg, dataset_dicts, required_meta_data, mapper, is_train=False)

    sampler_name = cfg.DATALOADER.SAMPLER_TEST
    if sampler_name == "InferenceSampler":
        sampler = samplers.InferenceSampler(len(dataset))
    elif sampler_name == "InfInferenceSampler":
        sampler = InfInferenceSampler(len(dataset))
    elif sampler_name == "SeqInferenceSampler":
        seq_name_list = [dataset_dicts[0][i]['sequence_name'] for i in range(len(dataset_dicts[0]))]
        sampler = SeqInferenceSampler(seq_name_list, len(seq_name_list))
    else:
        raise ValueError("Unknown test sampler: {}".format(sampler_name))

    # Always use 1 image per worker during inference since this is the
    # standard when reporting inference time in papers.
    batch_sampler = torch.utils.data.sampler.BatchSampler(sampler, 1, drop_last=False)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        batch_sampler=batch_sampler,
        collate_fn=trivial_batch_collator,
    )
    return data_loader
