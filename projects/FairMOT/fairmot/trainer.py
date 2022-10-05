import os, torch, logging, time
from torch.nn.parallel import DistributedDataParallel
from collections import OrderedDict
import numpy as np

import detectron2.utils.comm as comm
from detectron2.utils.env import TORCH_VERSION
from detectron2.utils.events import get_event_storage
from detectron2.evaluation.evaluator import DatasetEvaluator
from detectron2.evaluation import print_csv_format

from projects.EpochTrainer.epoch_trainer.default_epoch_trainer import DefaultEpochTrainer
from projects.Datasets.MOT.build import build_mot_train_loader, build_mot_test_loader
from projects.Datasets.MOT.evaluation.mot_evaluation import MotEvaluator
from projects.Datasets.MIX.build import build_mix_train_loader
from projects.FairMOT.fairmot.fairmot_checkpointer import FairMOTCheckpointer
from projects.FairMOT.fairmot.dataset_mapper import MOTFairMOTDatasetMapper
from projects.FairMOT.fairmot.evaluator import inference_on_dataset


class Trainer(DefaultEpochTrainer):

    @classmethod
    def build_checkpointer(cls, model, output_dir, optimizer, scheduler):
        checkpointer = FairMOTCheckpointer(
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
            mapper = MOTFairMOTDatasetMapper(cfg, is_train=True)
            return build_mix_train_loader(cfg, mapper=mapper)
        else:
            mapper = MOTFairMOTDatasetMapper(cfg, is_train=True)
            return build_mot_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        mapper = MOTFairMOTDatasetMapper(cfg, is_train=False)
        return build_mot_test_loader(cfg, dataset_name, mapper=mapper)

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return MotEvaluator(dataset_name, cfg, True, output_folder)

    def resume_or_load(self, resume=True):
        checkpoint = self.checkpointer.resume_or_load(self.cfg.MODEL.DETECTOR.WEIGHTS, resume=resume)
        if resume and self.checkpointer.has_checkpoint():
            self.start_iter = checkpoint.get("iteration", -1) + 1
            # The checkpoint stores the training iteration that just finished, thus we start
            # at the next iteration (or iter zero if there's no checkpoint).
        if isinstance(self._trainer.model, DistributedDataParallel):
            # broadcast loaded data/model from the first rank, because other
            # machines may not have access to the checkpoint file
            if TORCH_VERSION >= (1, 7):
                self._trainer.model._sync_params_and_buffers()
            self.start_iter = comm.all_gather(self.start_iter)[0]

    @classmethod
    def test(cls, cfg, model, evaluators=None):
        """
        Args:
            cfg (CfgNode):
            model (nn.Module):
            evaluators (list[DatasetEvaluator] or None): if None, will call
                :meth:`build_evaluator`. Otherwise, must have the same length as
                ``cfg.DATASETS.TEST``.

        Returns:
            dict: a dict of result metrics
        """
        logger = logging.getLogger(__name__)
        if isinstance(evaluators, DatasetEvaluator):
            evaluators = [evaluators]
        if evaluators is not None:
            assert len(cfg.DATASETS.TEST) == len(evaluators), "{} != {}".format(
                len(cfg.DATASETS.TEST), len(evaluators)
            )

        results = OrderedDict()
        for idx, dataset_name in enumerate(cfg.DATASETS.TEST):
            data_loader = cls.build_test_loader(cfg, dataset_name)
            # When evaluators are passed in as arguments,
            # implicitly assume that evaluators can be created before data_loader.
            if evaluators is not None:
                evaluator = evaluators[idx]
            else:
                try:
                    evaluator = cls.build_evaluator(cfg, dataset_name)
                except NotImplementedError:
                    logger.warn(
                        "No evaluator found. Use `DefaultTrainer.test(evaluators=)`, "
                        "or implement its `build_evaluator` method."
                    )
                    results[dataset_name] = {}
                    continue
            results_i = inference_on_dataset(cfg, model, data_loader, evaluator)
            results[dataset_name] = results_i
            if comm.is_main_process():
                assert isinstance(
                    results_i, dict
                ), "Evaluator must return a dict on the main process. Got {} instead.".format(
                    results_i
                )
                logger.info("Evaluation results for {} in csv format:".format(dataset_name))
                print_csv_format(results_i)

        if len(results) == 1:
            results = list(results.values())[0]
        return results

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

    @classmethod
    def build_optimizer(cls, cfg, model):
        lr = cfg.SOLVER.BASE_LR
        weight_decay = cfg.SOLVER.WEIGHT_DECAY
        optimizer = torch.optim.Adam(model.detector.backbone.parameters(), lr=lr, weight_decay=weight_decay)
        optimizer.add_param_group({'params': model.detector.upsample.parameters()})
        optimizer.add_param_group({'params': model.detector.head.parameters()})
        optimizer.add_param_group({'params': model.detector.loss.parameters()})
        return optimizer
