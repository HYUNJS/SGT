import contextlib
import copy
import glob
import itertools
import os.path as osp
import logging
import os
from collections import OrderedDict, defaultdict
import motmetrics as mm
import numpy as np
import torch
from fvcore.common.file_io import PathManager
from tabulate import tabulate
import io

from detectron2.data import MetadataCatalog
from detectron2.evaluation.coco_evaluation import _evaluate_predictions_on_coco, instances_to_coco_json
from detectron2.evaluation.evaluator import DatasetEvaluator
from detectron2.utils.comm import all_gather, is_main_process, synchronize
from detectron2.utils.logger import create_small_table

from projects.Datasets.MOT.vis.simple_visualzation_demo import SimpleVisualizationDemo
from projects.Datasets.MOT.mot_det import load_mot_dataset_dicts, convert_mot2coco
from projects.Datasets.MOT.evaluation.io import read_results, unzip_objs, write_results
from projects.Datasets.MOT.evaluation.mot2coco_evaluation import COCOJSON


class MotEvaluator(DatasetEvaluator):

    def __init__(self, dataset_name, cfg, distributed, output_dir=None):
        self._distributed = distributed
        self._output_dir = output_dir

        self._logger = logging.getLogger(__name__)
        self.dataset_name = dataset_name
        self.data_type = 'mot'
        self.gt_frame_set = {}
        self.gt_ignore_frame_set = {}
        self.gt_frame_dict = {}
        self.gt_ignore_frame_dict = {}
        self.seq_name_list = []
        self.demo = SimpleVisualizationDemo()
        self.eval_detection = cfg.TEST.EVAL_DETECTION
        self.eval_tracking = cfg.TEST.EVAL_TRACKING
        self.cpu_device = torch.device("cpu")
        self.vis_flag = cfg.TEST.VIS_FLAG
        self.vis_out_dir = cfg.TEST.VIS_OUT_DIR
        self.eval_img_ids = []
        self.eval_frame_idxs = {}
        self.pre_sample_flag = cfg.DATALOADER.PRE_SAMPLE_FLAG
        self.eval_filter_det_flag = cfg.TEST.EVAL_FILTER_DET_FLAG
        self.eval_det_by_seq_flag = cfg.TEST.EVAL_DET_BY_SEQ_FLAG
        self.eval_filter_det_score_thresh = cfg.TEST.EVAL_FILTER_DET_SCORE_THRESH
        self.eval_filter_det_size_thresh = cfg.TEST.EVAL_FILTER_DET_SIZE_THRESH
        self.eval_filter_det_ratio_thresh = cfg.TEST.EVAL_FILTER_DET_RATIO_THRESH
        self.eval_min_vis = cfg.TEST.EVAL_MIN_VISIBILITY

        if self.eval_detection:
            self.load_mot_annotations_coco_format(dataset_name)
        self.load_mot_annotations(dataset_name)
        self.reset()

    def add_eval_img_ids(self, eval_img_ids):
        self._logger.info(f"[MOT_evaluator] add eval img_ids # {len(eval_img_ids)}")
        self.eval_img_ids = [*self.eval_img_ids, *eval_img_ids]

    def add_eval_frame_idxs(self, eval_frame_idxs):
        self._logger.info(f"[MOT_evaluator] add eval frame_idxs - # seq: {len(eval_frame_idxs)}")
        for k, v in eval_frame_idxs.items():
            if k in self.eval_frame_idxs:
                new_dict = {k: [*self.eval_frame_idxs[k], *v]}
            else:
                new_dict = {k: v}
            self.eval_frame_idxs.update(new_dict)

    def load_mot_annotations(self, dataset_name):
        meta = MetadataCatalog.get(dataset_name)
        if 'demo' in dataset_name:
            seq_name = dataset_name[5:]
            gt_filename = osp.join(meta.root_dir, meta.split, 'gt.txt')
            self.gt_frame_set[seq_name] = read_results(gt_filename, self.data_type, is_gt=True, min_vis=self.eval_min_vis)
            self.gt_ignore_frame_set[seq_name] = read_results(gt_filename, self.data_type, is_ignore=True, min_vis=self.eval_min_vis)
            self.seq_name_list.append(seq_name)
        else:
            split_dir = os.path.join(meta.root_dir, meta.split)
            seq_dirs = sorted(glob.glob(os.path.join(split_dir, "*")))
            if hasattr(meta, 'detector') and meta.detector is not None:
                seq_dirs = [sd for sd in seq_dirs if meta.detector in sd]
            for seq_dir in seq_dirs:
                seq_file = os.path.join(seq_dir, "seqinfo.ini")
                seq_meta = open(seq_file).read()
                seq_name = str(seq_meta[seq_meta.find('name=') + 5:seq_meta.find('\nimDir')])
                gt_filename = os.path.join(meta.root_dir, meta.split, str(seq_name), 'gt', 'gt.txt')
                self.gt_frame_set[seq_name] = read_results(gt_filename, self.data_type, is_gt=True, min_vis=self.eval_min_vis)
                self.gt_ignore_frame_set[seq_name] = read_results(gt_filename, self.data_type, is_ignore=True, min_vis=self.eval_min_vis)
                self.seq_name_list.append(seq_name)

    def load_mot_annotations_coco_format(self, dataset_name):
        meta = MetadataCatalog.get(dataset_name)
        if meta.split == "test" or meta.split == "demo":
            self._coco_api = {}
            self._metadata = meta
            return

        dataset_dicts = load_mot_dataset_dicts(dataset_name, meta.root_dir, meta.split, meta.detector, self.eval_min_vis, False)
        coco_dataset = convert_mot2coco(dataset_name, dataset_dicts)
        with contextlib.redirect_stdout(io.StringIO()):
            self._coco_api = COCOJSON(coco_dataset)
            self._metadata = meta

    def reset(self):
        self._predictions = []
        self.eval_img_ids = []
        self.eval_frame_idxs = {}
        self.acc = mm.MOTAccumulator(auto_id=True)

    def reset_mot_acc(self):
        self.acc = mm.MOTAccumulator(auto_id=True)

    def eval_frame(self, frame_id, trk_tlwhs, trk_ids, rtn_events=False):
        # results
        trk_tlwhs = np.copy(trk_tlwhs)
        trk_ids = np.copy(trk_ids)

        # gts
        gt_objs = self.gt_frame_dict.get(frame_id, [])
        gt_tlwhs, gt_ids = unzip_objs(gt_objs)[:2]

        # ignore boxes
        ignore_objs = self.gt_ignore_frame_dict.get(frame_id, [])
        ignore_tlwhs = unzip_objs(ignore_objs)[0]

        # remove ignored results
        keep = np.ones(len(trk_tlwhs), dtype=bool)
        iou_distance = mm.distances.iou_matrix(ignore_tlwhs, trk_tlwhs, max_iou=0.5)
        if len(iou_distance) > 0:
            match_is, match_js = mm.lap.linear_sum_assignment(iou_distance)
            match_is, match_js = map(lambda a: np.asarray(a, dtype=int), [match_is, match_js])
            match_ious = iou_distance[match_is, match_js]

            match_js = np.asarray(match_js, dtype=int)
            match_js = match_js[np.logical_not(np.isnan(match_ious))]
            keep[match_js] = False
            trk_tlwhs = trk_tlwhs[keep]
            trk_ids = trk_ids[keep]

        # get distance matrix
        iou_distance = mm.distances.iou_matrix(gt_tlwhs, trk_tlwhs, max_iou=0.5)

        # acc
        self.acc.update(gt_ids, trk_ids, iou_distance)

        if rtn_events and iou_distance.size > 0 and hasattr(self.acc, 'last_mot_events'):
            events = self.acc.last_mot_events  # only supported by https://github.com/longcw/py-motmetrics
        else:
            events = None
        return events

    def eval_file(self, seq_name):
        self.reset_mot_acc()

        result_filename = os.path.join(self._output_dir, 'mot', '{}.txt'.format(seq_name))
        result_frame_dict = read_results(result_filename, self.data_type, is_gt=False)
        self.gt_frame_dict = self.gt_frame_set[seq_name]
        self.gt_ignore_frame_dict = self.gt_ignore_frame_set[seq_name]
        frames = sorted(list(set(self.gt_frame_dict.keys()) | set(result_frame_dict.keys())))
        eval_frame_idxs = None if len(self.eval_frame_idxs) == 0 else self.eval_frame_idxs[seq_name]
        if self.pre_sample_flag and eval_frame_idxs is None:
            raise Exception("Zero frame is sampled. Please use smaller interval")
        for frame_id in frames:
            if eval_frame_idxs is not None and frame_id not in eval_frame_idxs:
                continue
            trk_objs = result_frame_dict.get(frame_id, [])
            trk_tlwhs, trk_ids = unzip_objs(trk_objs)[:2]
            self.eval_frame(frame_id, trk_tlwhs, trk_ids, rtn_events=False)

        return self.acc

    @staticmethod
    def get_summary(accs, names, metrics=('mota', 'num_switches', 'idp', 'idr', 'idf1', 'precision', 'recall')):
        names = copy.deepcopy(names)
        if metrics is None:
            metrics = mm.metrics.motchallenge_metrics
        metrics = copy.deepcopy(metrics)

        mh = mm.metrics.create()
        summary = mh.compute_many(
            accs,
            metrics=metrics,
            names=names,
            generate_overall=True
        )

        return summary

    @staticmethod
    def save_summary(summary, filename):
        import pandas as pd
        writer = pd.ExcelWriter(filename)
        summary.to_excel(writer)
        writer.save()

    def process(self, inputs, outputs):
        if isinstance(inputs[0], list):
            # inputs = list(zip(*inputs))
            inputs = inputs[0]

        for inputs_per_frame, outputs_per_frame in zip(inputs, outputs):
            instances = outputs_per_frame["instances"].to(self.cpu_device)
            boxes = instances.pred_boxes.tensor
            if hasattr(outputs_per_frame, 'online_ids'):
                tlwhs = outputs_per_frame['online_tlwhs']
                pred_ids = outputs_per_frame['online_ids']
            else:
                x0, y0, x1, y1 = boxes.unbind(-1)
                tlwhs = [x0 + 1, y0 + 1, (x1 - x0), (y1 - y0)]
                tlwhs = torch.stack(tlwhs, dim=-1).cpu().data.numpy()
                if hasattr(instances, 'pred_ids'):
                    pred_ids = instances.pred_ids
                else:
                    pred_ids = torch.zeros_like(instances.scores)

            prediction = {
                "image_id": inputs_per_frame["image_id"],
                "seq_name": inputs_per_frame["sequence_name"],
                "frame_id": inputs_per_frame['frame_idx'] + 1,
                "online_tlwhs": tlwhs,
                "online_ids": pred_ids,
                "conf": instances.scores,
                "instances": instances,
            }
            self._predictions.append(prediction)
            if self.vis_flag:
                filtered_prediction = self.filter(prediction, remove_id=0, score_th=0.4)
                out_dir = self.vis_out_dir
                if out_dir == '':
                    out_dir = os.path.join(self._output_dir, "demo")
                self.demo.visualize_output([inputs_per_frame], [filtered_prediction], output_dir=out_dir)

    def filter(self, prediction, remove_id=None, score_th=None):
        if remove_id is None and score_th is None:
            return prediction
        else:
            device = prediction['instances'].scores.device
            if score_th is not None:
                score_mask = prediction['instances'].scores >= score_th
            else:
                score_mask = torch.ones(len(prediction['instances']), dtype=torch.bool, device=device)

            if hasattr(prediction['instances'], 'pred_ids') and remove_id is not None:
                id_mask = prediction['instances'].pred_ids != remove_id
            else:
                id_mask = torch.ones(len(prediction['instances']), dtype=torch.bool, device=device)

            mask = torch.logical_and(score_mask, id_mask)
            filtered_prediction = {'instances': prediction['instances'][mask], 'image_id': prediction['image_id']}
            return filtered_prediction

    def evaluate(self):
        if self._distributed:
            synchronize()
            predictions = all_gather(self._predictions)
            predictions = list(itertools.chain(*predictions))
            eval_img_ids_gathered = all_gather(self.eval_img_ids)
            eval_frame_idxs_gathered = all_gather(self.eval_frame_idxs)
            if is_main_process():
                if self.pre_sample_flag:
                    self.eval_frame_idxs, self.eval_img_ids = {}, []
                    for eval_img_ids, eval_frame_idxs in zip(eval_img_ids_gathered, eval_frame_idxs_gathered):
                        self.add_eval_img_ids(eval_img_ids)
                        self.add_eval_frame_idxs(eval_frame_idxs)
                    self._logger.info(f"{len(self.eval_img_ids)} sampled frames used for evaluation")
            else:
                return
        else:
            predictions = self._predictions

        if self._output_dir:
            PathManager.mkdirs(self._output_dir)

        predictions = self.eliminate_duplicated_prediction(predictions)
        return copy.deepcopy(self._eval_predictions(predictions))

    def eliminate_duplicated_prediction(self, predictions):
        memory = []
        new_predictions = []
        for pred_per_frame in predictions:
            image_id = pred_per_frame['image_id']
            if image_id in memory:
                continue
            new_predictions.append(pred_per_frame)
            memory.append(image_id)
        return new_predictions

    def _eval_predictions(self, predictions):
        self._logger.info("Preparing results for MOT format ...")

        eval_types = []
        if self.eval_detection:
            eval_types.append("det")
        if self.eval_tracking:
            eval_types.append("mot")

        for eval_type in eval_types:
            PathManager.mkdirs(os.path.join(self._output_dir, eval_type))
            for seq_name in self.seq_name_list:
                result_filename = os.path.join(self._output_dir, eval_type, '{}.txt'.format(seq_name))
                # seq_predictions = [prediction for prediction in predictions if prediction["seq_name"] == seq_name]
                seq_predictions = [prediction for prediction in predictions if prediction["seq_name"].lower() == seq_name.lower()]
                write_results(result_filename, seq_predictions, 'mot', eval_type)
                self._logger.info("[{}] #Detections - {}".format(seq_name, sum([len(s['instances']) for s in seq_predictions])))
                self._logger.info("[{}] # Recovered Dets - {}".format(seq_name, sum([(s['conf'] <= self.eval_filter_det_score_thresh).sum() for s in seq_predictions])))
        dict_summary = OrderedDict()
        if "test" not in self.dataset_name and "demo" not in self.dataset_name:
            if self.eval_detection:
                if self.eval_det_by_seq_flag:
                    self.evaluate_detections_by_coco_api_by_sequence(predictions)
                dict_summary.update(self.evaluate_detections_by_coco_api(predictions))
                if self.eval_filter_det_flag:
                    predictions_filtered = []
                    for pred in predictions:
                        boxes = pred['instances'].pred_boxes.tensor
                        w = boxes[:, 2] - boxes[:, 0]
                        h = boxes[:, 3] - boxes[:, 1]
                        score_mask = pred['instances'].scores >= self.eval_filter_det_score_thresh
                        size_mask = w * h > self.eval_filter_det_size_thresh
                        ratio_mask = w / h <= self.eval_filter_det_ratio_thresh
                        mask = torch.logical_and(score_mask, size_mask)
                        mask = torch.logical_and(mask, ratio_mask)
                        predictions_filtered.append({'instances': pred['instances'][mask], 'image_id': pred['image_id']})
                    filtered_det_eval = self.evaluate_detections_by_coco_api(predictions_filtered)
                    filtered_det_eval['MOT-Det-Eval>={}'.format(self.eval_filter_det_score_thresh)] = filtered_det_eval.pop('MOT-Det-Eval')
                    dict_summary.update(filtered_det_eval)
            if self.eval_tracking:
                dict_summary.update(self.evaluation_tracking())
        return dict_summary

    def evaluate_detections_by_coco_api_by_sequence(self, predictions):
        sampled_img_ids = None if len(self.eval_img_ids) == 0 else self.eval_img_ids
        if self.pre_sample_flag and sampled_img_ids is None:
            raise Exception("Zero frame is sampled. Please use smaller interval")
        coco_results_per_seq, image_ids_per_seq = {}, {}
        for prediction in predictions:
            seq_name = prediction['seq_name']
            if seq_name not in coco_results_per_seq:
                coco_results_per_seq[seq_name] = []
                image_ids_per_seq[seq_name] = []
            coco_results_per_seq[seq_name].extend(instances_to_coco_json(prediction["instances"], prediction["image_id"]))
            image_ids_per_seq[seq_name].append(prediction['image_id'])

        task = 'bbox'
        AP50_per_seq = {}
        for seq_name in image_ids_per_seq.keys():
            self._logger.info("[Evaluation detection performance] {}".format(seq_name))
            coco_eval = (
                _evaluate_predictions_on_coco(
                    self._coco_api, coco_results_per_seq[seq_name], task, kpt_oks_sigmas=None, use_fast_impl=True, img_ids=image_ids_per_seq[seq_name],
                )
                if len(coco_results_per_seq[seq_name]) > 0
                else None  # cocoapi does not handle empty results very well
            )
            res = self._derive_coco_results(
                coco_eval, task, class_names=self._metadata.get("thing_classes")
            )
            AP50_per_seq[seq_name] = res['AP50']

        ap50_per_seq_str = "Evaluation detection results per sequence\n"
        ap50_per_seq_str += "{:<20} {:<15}\n".format('Sequence', 'AP50')
        for seq_name, ap50 in AP50_per_seq.items():
            ap50_per_seq_str += "{:<20} {:.3f}\n".format(seq_name, ap50)
        self._logger.info(ap50_per_seq_str)

    def evaluate_detections_by_coco_api(self, predictions):
        task = 'bbox'
        coco_results = []
        for prediction in predictions:
            coco_results.extend(instances_to_coco_json(prediction["instances"], prediction["image_id"]))
        sampled_img_ids = None if len(self.eval_img_ids) == 0 else self.eval_img_ids
        if self.pre_sample_flag and sampled_img_ids is None:
            raise Exception("Zero frame is sampled. Please use smaller interval")
        coco_eval = (
            _evaluate_predictions_on_coco(
                self._coco_api, coco_results, task, kpt_oks_sigmas=None, use_fast_impl=True, img_ids=sampled_img_ids,
            )
            if len(coco_results) > 0
            else None  # cocoapi does not handle empty results very well
        )
        res = self._derive_coco_results(
            coco_eval, task, class_names=self._metadata.get("thing_classes")
        )
        return {'MOT-Det-Eval': res}

    def _derive_coco_results(self, coco_eval, iou_type, class_names=None):
        metrics = {
            "bbox": ["AP", "AP50", "AP75", "APs", "APm", "APl"],
            "segm": ["AP", "AP50", "AP75", "APs", "APm", "APl"],
            "keypoints": ["AP", "AP50", "AP75", "APm", "APl"],
        }[iou_type]

        if coco_eval is None:
            self._logger.warn("No predictions from the model!")
            return {metric: float("nan") for metric in metrics}

        # the standard metrics
        results = {
            metric: float(coco_eval.stats[idx] * 100 if coco_eval.stats[idx] >= 0 else "nan")
            for idx, metric in enumerate(metrics)
        }
        self._logger.info(
            "Evaluation results for {}: \n".format(iou_type) + create_small_table(results)
        )
        if not np.isfinite(sum(results.values())):
            self._logger.info("Some metrics cannot be computed and is shown as NaN.")

        if class_names is None or len(class_names) <= 1:
            return results
        # Compute per-category AP
        # from https://github.com/facebookresearch/Detectron/blob/a6a835f5b8208c45d0dce217ce9bbda915f44df7/detectron/datasets/json_dataset_evaluator.py#L222-L252 # noqa
        precisions = coco_eval.eval["precision"]
        # precision has dims (iou, recall, cls, area range, max dets)
        assert len(class_names) == precisions.shape[2]

        results_per_category = []
        for idx, name in enumerate(class_names):
            # area range index 0: all area ranges
            # max dets index -1: typically 100 per image
            precision = precisions[:, :, idx, 0, -1]
            precision = precision[precision > -1]
            ap = np.mean(precision) if precision.size else float("nan")
            results_per_category.append(("{}".format(name), float(ap * 100)))

        # tabulate it
        N_COLS = min(6, len(results_per_category) * 2)
        results_flatten = list(itertools.chain(*results_per_category))
        results_2d = itertools.zip_longest(*[results_flatten[i::N_COLS] for i in range(N_COLS)])
        table = tabulate(
            results_2d,
            tablefmt="pipe",
            floatfmt=".3f",
            headers=["category", "AP"] * (N_COLS // 2),
            numalign="left",
        )
        self._logger.info("Per-category {} AP: \n".format(iou_type) + table)

        results.update({"AP-" + name: ap for name, ap in results_per_category})
        return results

    def evaluation_tracking(self):
        accs = []
        for seq_name in self.seq_name_list:
            accs.append(self.eval_file(seq_name))

        metrics = mm.metrics.motchallenge_metrics
        mh = mm.metrics.create()
        summary = self.get_summary(accs, self.seq_name_list, metrics)
        strsummary = mm.io.render_summary(
            summary,
            formatters=mh.formatters,
            namemap=mm.io.motchallenge_metric_names
        )
        self._logger.info("\n" + strsummary)
        results = summary.to_dict(into=OrderedDict)
        mota_summary = {'MOT-Tracking-eval': {metric: scores['OVERALL'] for metric, scores in results.items()}}
        return mota_summary

    @staticmethod
    def flatten_array(tp, fp):
        # Flatten out tp and fp into a numpy array
        i = 0
        for im in tp:
            if type(im) != type([]):
                i += im.shape[0]

        tp_flat = np.zeros(i)
        fp_flat = np.zeros(i)

        i = 0
        for tp_im, fp_im in zip(tp, fp):
            if type(tp_im) != type([]):
                s = tp_im.shape[0]
                tp_flat[i:s + i] = tp_im
                fp_flat[i:s + i] = fp_im
                i += s
        return tp_flat, fp_flat

    @staticmethod
    def calc_ap(tp, fp, npos):
        rec = tp / float(npos)
        # avoid divide by zero in case the first detection matches a difficult
        # ground truth (probably not needed in my code but doesn't harm if left)
        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        tmp = np.maximum(tp + fp, np.finfo(np.float64).eps)

        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
        return prec, rec, ap
