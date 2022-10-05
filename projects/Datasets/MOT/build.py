import itertools
import logging
import numpy as np
import torch.utils.data

from tabulate import tabulate
from termcolor import colored
from multiprocessing import Manager
from detectron2.utils.env import seed_all_rng
from detectron2.utils.comm import get_world_size
from detectron2.utils.logger import log_first_n
from detectron2.data import DatasetCatalog, MetadataCatalog, samplers
from detectron2.data.common import DatasetFromList
from detectron2.data.detection_utils import check_metadata_consistency
from projects.Datasets.MOT.dataset_mapper import DefaultMOTDatasetMapper
from projects.Datasets.MOT.common import MapDataset
from projects.Datasets.MOT.distributed_sampler import InfInferenceSampler
from projects.Datasets.MOT.sequence_distributed_sampler import SeqInferenceSampler

__all__ = [
    "build_mot_train_loader",
    "build_mot_test_loader",
    "get_mot_dataset_dicts",
    "print_instances_class_histogram",
]


def print_instances_class_histogram(dataset_dicts, class_names, filter_ignore=False):
    """
    Args:
        dataset_dicts (list[dict]): list of dataset dicts.
        class_names (list[str]): list of class names (zero-indexed).
    """
    num_classes = len(class_names)
    hist_bins = np.arange(num_classes + 1)
    histogram = np.zeros((num_classes,), dtype=np.int)
    for entry in dataset_dicts:
        annos = entry["annotations"]
        if filter_ignore:
            classes = [x["category_id"] for x in annos if (not x.get("iscrowd", 0)) and (not x.get("ignore", 0))]
        else:
            classes = [x["category_id"] for x in annos if not x.get("iscrowd", 0)]
        histogram += np.histogram(classes, bins=hist_bins)[0]

    N_COLS = min(6, len(class_names) * 2)

    def short_name(x):
        # make long class names shorter. useful for lvis
        if len(x) > 13:
            return x[:11] + ".."
        return x

    data = list(
        itertools.chain(*[[short_name(class_names[i]), int(v)] for i, v in enumerate(histogram)])
    )
    total_num_instances = sum(data[1::2])
    data.extend([None] * (N_COLS - (len(data) % N_COLS)))
    if num_classes > 1:
        data.extend(["total", total_num_instances])
    data = itertools.zip_longest(*[data[i::N_COLS] for i in range(N_COLS)])
    table = tabulate(
        data,
        headers=["category", "#instances"] * (N_COLS // 2),
        tablefmt="pipe",
        numalign="left",
        stralign="center",
    )

    if filter_ignore:
        txt = "Distribution of instances among all {} categories with ignore of {}:\n".format(num_classes, False) + colored(table, "cyan")
    else:
        txt = "Distribution of instances among all {} categories:\n".format(num_classes) + colored(table, "cyan")
    log_first_n(
        logging.INFO,
        txt,
        key="message",
    )
    return txt


def print_instances_track_id_histogram(dataset_dicts):
    seq_info = dict()
    for annotations_per_frame in dataset_dicts:
        seq_name = annotations_per_frame['sequence_name']
        if seq_name not in seq_info:
            seq_info[seq_name] = {"num_frames": 0, "tid_set": set(), "max_obj": 0, "total_obj": 0}
        tids = [annotation['track_id'] for annotation in annotations_per_frame['annotations']]
        seq_info[seq_name]["tid_set"].update(tids)
        seq_info[seq_name]["max_obj"] = max(len(annotations_per_frame['annotations']), seq_info[seq_name]["max_obj"])
        seq_info[seq_name]["total_obj"] += len(annotations_per_frame['annotations'])
        seq_info[seq_name]["num_frames"] += 1

    seq_names = seq_info.keys()
    num_frames = [seq_info[k]["num_frames"] for k in seq_info.keys()]
    nds = [len(seq_info[k]["tid_set"]) for k in seq_info.keys()]
    max_objs = [seq_info[k]["max_obj"] for k in seq_info.keys()]
    avg_objs = [seq_info[k]["total_obj"] / seq_info[k]["num_frames"] for k in seq_info.keys()]

    N_COLS = 5
    data = list(
        itertools.chain(
            *[[seq_name, int(num_frame), int(max_obj), f"{avg_obj:.1f}", int(num_ids)] for
              seq_name, num_frame, max_obj, avg_obj, num_ids in
              zip(seq_names, num_frames, max_objs, avg_objs, nds)]
        ))
    data = itertools.zip_longest(*[data[i::N_COLS] for i in range(N_COLS)])
    table = tabulate(
        data,
        headers=["sequence name", "#frame", "#max obj", "#avg obj", "#id"],
        tablefmt="pipe",
        numalign="left",
        stralign="center",
    )
    txt = "Distribution of instances among all sequences:\n" + colored(table, "cyan")
    log_first_n(logging.INFO, txt, key="message")


def get_mot_dataset_dicts(dataset_names, reassign_id=False, num_ids=None,
                          filter_empty=False, filter_pairless=False):
    assert len(dataset_names)
    
    dataset_dicts_list = list()
    for dataset_name in dataset_names:
        dataset_dicts = DatasetCatalog.get(dataset_name)
        dataset_dicts_list.append(dataset_dicts)
    dataset_dicts = dataset_dicts_list

    # check amount of data samples for each dataset listed in cfg.DATASETS.TRAIN or TEST
    for dataset_name, dicts in zip(dataset_names, dataset_dicts):
        assert len(dicts), "Dataset '{}' is empty!".format(dataset_name)

    dataset_dicts = list(itertools.chain.from_iterable(dataset_dicts))

    has_instances = "annotations" in dataset_dicts[0]
    total_ids = None
    if has_instances:
        if reassign_id:
            dataset_dicts, total_ids = reassign_tracking_id(dataset_dicts, num_ids)
        if filter_empty:
            dataset_dicts = filter_images_without_annotations(dataset_dicts)
        if filter_pairless:
            dataset_dicts = filter_images_without_next_tracking_boxes(dataset_dicts)
        class_names = MetadataCatalog.get(dataset_names[0]).thing_classes
        check_metadata_consistency("thing_classes", dataset_names)
        print_instances_class_histogram(dataset_dicts, class_names)
        print_instances_track_id_histogram(dataset_dicts)
    if total_ids is None:
        return dataset_dicts
    else:
        return dataset_dicts, total_ids


def build_mot_train_loader(cfg, mapper=None):
    dataset_dicts, total_ids = get_mot_dataset_dicts(
        cfg.DATASETS.TRAIN,
        reassign_id=cfg.DATALOADER.REASSIGN_ID,
        num_ids=cfg.DATALOADER.NUM_IDS if hasattr(cfg.DATALOADER, "NUM_IDS") else None,
        filter_empty=cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS,
        filter_pairless=cfg.DATALOADER.FILTER_PAIRLESS_ANNOTATIONS,
    )
    cfg.DATALOADER.defrost()
    cfg.DATALOADER.NUM_IDS = total_ids
    cfg.DATALOADER.freeze()
    dataset = DatasetFromList(dataset_dicts, copy=False)

    sampler_name = cfg.DATALOADER.SAMPLER_TRAIN
    logger = logging.getLogger(__name__)
    logger.info("Using training sampler {}".format(sampler_name))

    if mapper is None:
        mapper = DefaultMOTDatasetMapper(cfg, is_train=True)

    if sampler_name == "TrainingSampler":
        seed = None if cfg.DATALOADER.SEED == -1 else cfg.DATALOADER.SEED
        dataset = MapDataset(cfg, dataset, mapper, is_train=True)
        sampler = samplers.TrainingSampler(len(dataset), seed=seed)
    elif sampler_name == "RepeatFactorTrainingSampler":
        dataset = MapDataset(cfg, dataset, mapper, is_train=True)
        sampler = samplers.RepeatFactorTrainingSampler(
            dataset_dicts, cfg.DATALOADER.REPEAT_THRESHOLD
        )
    else:
        raise ValueError("Unknown training sampler: {}".format(sampler_name))

    return build_batch_data_loader(
        dataset,
        sampler,
        cfg.SOLVER.IMS_PER_BATCH,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
    )

def build_mot_test_loader(cfg, dataset_name, mapper=None):
    dataset_dicts = get_mot_dataset_dicts(
        [dataset_name],
        filter_empty=cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS
    )
    if cfg.DATALOADER.PRE_SAMPLE_FLAG:
        seq_listed_dataset_dicts = listing_by_sequence(dataset_dicts)
        intvs = [cfg.DATALOADER.PRE_SAMPLE_INTERVAL] * len(seq_listed_dataset_dicts)
        dataset_dicts = list(itertools.chain.from_iterable(map(lambda x, intv: sampled_by_interval(x, intv), seq_listed_dataset_dicts, intvs)))
    dataset = DatasetFromList(dataset_dicts)

    if mapper is None:
        mapper = DefaultMOTDatasetMapper(cfg, is_train=False)
    dataset = MapDataset(cfg, dataset, mapper, is_train=False)
    # bbox_scale_stat_ori(dataset_dicts)
    # bbox_scale_stat_transformed(dataset)

    sampler_name = cfg.DATALOADER.SAMPLER_TEST
    if sampler_name == "InferenceSampler":
        sampler = samplers.InferenceSampler(len(dataset))
    elif sampler_name == "InfInferenceSampler":
        sampler = InfInferenceSampler(len(dataset))
    elif sampler_name == "SeqInferenceSampler":
        seq_name_list = [dataset_dicts[i]['sequence_name'] for i in range(len(dataset_dicts))]
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


def build_batch_data_loader(dataset, sampler, total_batch_size, *, num_workers=0):
    world_size = get_world_size()
    assert (
        total_batch_size > 0 and total_batch_size % world_size == 0
    ), "Total batch size ({}) must be divisible by the number of gpus ({}).".format(
        total_batch_size, world_size
    )

    batch_size = total_batch_size // world_size
    batch_sampler = torch.utils.data.sampler.BatchSampler(
        sampler, batch_size, drop_last=True
    )  # drop_last so the batch always have the same size
    return torch.utils.data.DataLoader(
        dataset,
        num_workers=num_workers,
        batch_sampler=batch_sampler,
        collate_fn=trivial_batch_collator,
        worker_init_fn=worker_init_reset_seed,
    )


def trivial_batch_collator(batch):
    """
    A batch collator that does nothing.
    """
    return batch


def filter_images_without_annotations(dataset_dicts):
    num_before = len(dataset_dicts)

    def valid(anns):
        if len(anns) == 0:
            return False
        else:
            return True

    dataset_dicts = [x for x in dataset_dicts if valid(x["annotations"])]
    num_after = len(dataset_dicts)
    logger = logging.getLogger(__name__)
    logger.info(
        "Removed {} images with no usable annotations. {} images left.".format(
            num_before - num_after, num_after
        )
    )
    return dataset_dicts


def filter_images_without_next_tracking_boxes(dataset_dicts):
    num_before = len(dataset_dicts)

    def has_next_tracking_boxes(t1, t2):
        if t1['sequence_name'] != t2['sequence_name']:
            return False
        t1_annos = t1['annotations']
        t2_annos = t2['annotations']

        matching = []
        for i, t1_anno in enumerate(t1_annos):
            for j, t2_anno in enumerate(t2_annos):
                if t1_anno['track_id'] == t2_anno['track_id']:
                    matching.append((i, j))
                    break

        # remove non matching boxes
        # cur_frame['annotations'] = [cur_annos[pair[0]] for pair in matching]
        # next_frame['annotations'] = [next_annos[pair[1]] for pair in matching]

        if len(matching) == 0:
            return False

        return True

    filtered_dataset_dicts = []
    for i in range(len(dataset_dicts) - 1):
        if has_next_tracking_boxes(dataset_dicts[i], dataset_dicts[i + 1]):
            filtered_dataset_dicts.append(dataset_dicts[i])

    num_after = len(filtered_dataset_dicts)
    logger = logging.getLogger(__name__)
    logger.info(
        "Removed {} images without next tracking boxes. {} images left.".format(num_before - num_after, num_after)
    )
    return filtered_dataset_dicts


def filter_images_in_last_interval(dataset_dicts):
    prev_seq_name = ""
    slicing_idx_per_seq, fps_per_seq = dict(), dict()
    for idx, d in enumerate(dataset_dicts):
        if prev_seq_name is not d['sequence_name']:
            fps_per_seq[d['sequence_name']] = d['seq_fps']
            slicing_idx_per_seq[d['sequence_name']] = [idx]
            if idx != 0:
                slicing_idx_per_seq[prev_seq_name].append(idx - fps_per_seq[prev_seq_name])
            prev_seq_name = d['sequence_name']
    slicing_idx_per_seq[prev_seq_name].append(idx - fps_per_seq[prev_seq_name])
    fps_per_seq[d['sequence_name']] = d['seq_fps']
    new_dataset_dicts = [dataset_dicts[i:j] for i, j in slicing_idx_per_seq.values()]
    new_dataset_dicts = [d for dd in new_dataset_dicts for d in dd]
    logger = logging.getLogger(__name__)
    logger.info("\nInterval sampling mode - slicing idxs per sequence\n{}".format(str(slicing_idx_per_seq)))
    logger.info("\nFPS per sequence\n{}".format(str(fps_per_seq)))
    logger.info("Removed {} images in last interval. {} images left.".format(
        len(dataset_dicts) - len(new_dataset_dicts), len(new_dataset_dicts)))
    return new_dataset_dicts


def reassign_tracking_id(dataset_dicts, config_num_ids):
    # find all tracking ids per sequence
    seq_name_dict = []
    tid_set = dict()
    for annotations_per_frame in dataset_dicts:
        seq_name = annotations_per_frame['sequence_name']
        if seq_name not in tid_set:
            tid_set[seq_name] = set()
            seq_name_dict.append(seq_name)
        tids = [annotation['track_id'] for annotation in annotations_per_frame['annotations']]
        tid_set[seq_name].update(tids)

    nds = [len(x) for x in tid_set.values()]
    cds = [sum(nds[:i]) for i in range(len(nds))]

    # generate mapping dict from original tracking id to new id
    tid_map, offset = dict(), 0
    for i, seq_name in enumerate(seq_name_dict):
        tid_map[seq_name] = dict()
        offset = cds[i]
        for tid, new_tid in zip(tid_set[seq_name], range(0, nds[i] + 1)):
            tid_map[seq_name][tid] = new_tid + offset

    # remap tracking ids into dataset_dicts
    for annotations_per_frame in dataset_dicts:
        seq_name = annotations_per_frame['sequence_name']
        seq_tid_map = tid_map[seq_name]
        for annotation in annotations_per_frame['annotations']:
            tid = annotation['track_id']
            new_tid = seq_tid_map[tid]
            annotation['track_id'] = new_tid

    num_ids = sum(nds)
    logger = logging.getLogger(__name__)
    if config_num_ids is not None:
        logger.info("Number of tracking IDs in config: {}".format(config_num_ids))
    logger.info("Number of tracking IDs in dataset: {}".format(num_ids))

    return dataset_dicts, num_ids


def worker_init_reset_seed(worker_id):
    seed_all_rng(np.random.randint(2 ** 31) + worker_id)

def listing_by_sequence(dataset_dicts):
    ## separate dataset_dicts by sequence
    sampled_dataset_dicts = [[]]
    prev_seq_name = dataset_dicts[0]['sequence_name']
    for d in dataset_dicts:
        curr_seq_name = d['sequence_name']
        if prev_seq_name != curr_seq_name:
            sampled_dataset_dicts.append([])
        sampled_dataset_dicts[-1].append(d)
        prev_seq_name = curr_seq_name
    return sampled_dataset_dicts

def sampled_by_interval(sequence_dataset_dicts, interval):
    sampled_dataset_dicts = []
    for i in range(0,len(sequence_dataset_dicts), interval):
        sampled_dataset_dicts.append(sequence_dataset_dicts[i])
    return sampled_dataset_dicts

def bbox_scale_stat_transformed(dataset):
    areaRng = {'small': [0 ** 2, 32 ** 2], 'medium': [32 ** 2, 96 ** 2], 'large': [96 ** 2, 1e5 ** 2]}
    areaCnt = {'small': 0, 'medium': 0, 'large': 0}
    areaRatio = {'small': 0, 'medium': 0, 'large': 0}
    bbox_cnt = 0
    for data in dataset:
        for area in data['instances'].gt_boxes.area():
            bbox_cnt += 1
            for area_k in areaRng.keys():
                if area > areaRng[area_k][0] and area < areaRng[area_k][1]:
                    areaCnt[area_k] += 1
    for area_k in areaRng.keys():
        areaRatio[area_k] = areaCnt[area_k] / bbox_cnt
    txt = f"[bbox_scale_stat_transformed]\ncount per area: {areaCnt}\nratio per area:{areaRatio}"
    log_first_n(logging.INFO, txt, key="message")

def bbox_scale_stat_ori(dataset_dicts):
    areaRng = {'small':[0**2, 32**2], 'medium':[32**2, 96**2], 'large':[96**2, 1e5**2]}
    areaCnt = {'small':0, 'medium':0, 'large':0}
    areaRatio = {'small':0, 'medium':0, 'large':0}
    bbox_cnt = 0
    for data in dataset_dicts:
        for anno in data['annotations']:
            w, h = anno['bbox'][2:]
            area = w*h
            bbox_cnt += 1
            for area_k in areaRng.keys():
                if area > areaRng[area_k][0] and area < areaRng[area_k][1]:
                    areaCnt[area_k] += 1
    for area_k in areaRng.keys():
        areaRatio[area_k] = areaCnt[area_k] / bbox_cnt
    txt = f"[bbox_scale_stat_ori]\ncount per area: {areaCnt}\nratio per area:{areaRatio}"
    log_first_n(logging.INFO, txt, key="message")
