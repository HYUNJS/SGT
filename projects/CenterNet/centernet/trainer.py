import os, time
import numpy as np

from detectron2.data import MetadataCatalog, build_detection_train_loader, build_detection_test_loader
from detectron2.utils.events import get_event_storage
from detectron2.evaluation import COCOEvaluator, DatasetEvaluators

from projects.EpochTrainer.epoch_trainer.default_epoch_trainer import DefaultEpochTrainer
from projects.Datasets.MOT.build import build_mot_train_loader, build_mot_test_loader
from projects.Datasets.MOT.evaluation.mot_evaluation import MotEvaluator
from projects.Datasets.MIX.build import build_mix_train_loader
from projects.CenterNet.centernet.dataset_mapper import COCOCenterNetDatasetMapper, MOTCenterNetDatasetMapper
from projects.CenterNet.centernet.checkpoint.centernet_checkpoint import CenterNetCheckpointer

class Trainer(DefaultEpochTrainer):

    @classmethod
    def build_checkpointer(cls, model, output_dir, optimizer, scheduler):
        checkpointer = CenterNetCheckpointer(
            # Assume you want to save checkpoints together with logs/statistics
            model,
            output_dir,
            optimizer=optimizer,
            scheduler=scheduler,
        )
        return checkpointer

    @classmethod
    def build_train_loader(cls, cfg):
        if "mix" in cfg.DATASETS.TRAIN[0]:
            assert all(["mix" in t for t in cfg.DATASETS.TRAIN]), "Joint dataset only accepts only mix format"
            mapper = MOTCenterNetDatasetMapper(cfg, is_train=True)
            return build_mix_train_loader(cfg, mapper=mapper)
        elif "mot" in cfg.DATASETS.TRAIN[0]:
            mapper = MOTCenterNetDatasetMapper(cfg, is_train=True)
            return build_mot_train_loader(cfg, mapper=mapper)
        else:
            mapper = COCOCenterNetDatasetMapper(cfg, is_train=True)
            return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        if "mot" in cfg.DATASETS.TEST[0]:
            mapper = MOTCenterNetDatasetMapper(cfg, is_train=False)
            return build_mot_test_loader(cfg, dataset_name, mapper=mapper)
        else:
            mapper = COCOCenterNetDatasetMapper(cfg, is_train=False)
            return build_detection_test_loader(cfg, dataset_name, mapper=mapper)

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        if evaluator_type in ["coco", "coco_panoptic_seg"]:
            evaluator_list.append(COCOEvaluator(dataset_name, cfg, True, output_folder))
        if evaluator_type in ["mot"]:
            evaluator_list.append(MotEvaluator(dataset_name, cfg, True, output_folder))
        return DatasetEvaluators(evaluator_list)

    def run_step(self):
        start = time.perf_counter()
        data = next(self._trainer._data_loader_iter)
        data_time = time.perf_counter() - start

        loss_dict = self.model(data)
        loss = loss_dict['total_loss']
        self._write_metrics(loss_dict, data_time, prefix="train")

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def _write_metrics(self, loss_dict, data_time, prefix="train"):
        metrics_dict = {k: v.detach().cpu().item() for k, v in loss_dict.items()}
        metrics_dict["data_time"] = data_time
        all_metrics_dict = comm.gather(metrics_dict)

        if comm.is_main_process():
            storage = get_event_storage()

            data_time = np.max([x.pop("data_time") for x in all_metrics_dict])
            storage.put_scalar("data_time", data_time)

            metrics_dict = {k: np.mean([x[k] for x in all_metrics_dict]) for k in all_metrics_dict[0].keys()}
            if len(metrics_dict) > 1:
                storage.put_scalars(**metrics_dict)
