import sys
import os.path as osp
curr_dir = osp.dirname(osp.abspath(__file__))
sys.path.append(f'{curr_dir}/../../')

import logging
import detectron2.utils.comm as comm

from detectron2.config import get_cfg
from detectron2.engine import default_argument_parser, default_setup, hooks, launch
from detectron2.utils.logger import setup_logger
from detectron2.evaluation import verify_results

from projects.EpochTrainer.epoch_trainer.config import add_epoch_trainer_config
from projects.Datasets.COCO.builtin import register_all_coco
from projects.Datasets.MOT.builtin import register_all_mot
from projects.Datasets.MIX.builtin import register_mix_tgt
from projects.Datasets.MOT.config import add_mot_dataset_config
from projects.Datasets.MIX.config import add_mix_dataset_config
from projects.Datasets.arg_parser import add_dataset_parser
from projects.CenterNet.centernet import add_centernet_config
from projects.CenterNet.centernet.checkpoint.centernet_checkpoint import CenterNetCheckpointer
from projects.CenterNet.centernet.trainer import Trainer


def setup(args):
    cfg = get_cfg()
    add_epoch_trainer_config(cfg)
    add_mot_dataset_config(cfg)
    add_mix_dataset_config(cfg)
    add_centernet_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    logger = setup_logger(distributed_rank=comm.get_rank(), name="projects")
    logger.setLevel(logging.INFO)
    return cfg


def main(args):
    cfg = setup(args)
    register_all_coco(args.data_dir)
    register_all_mot(args.data_dir, cfg)
    register_mix_tgt(args.data_dir, cfg)
    if args.eval_only:
        model = Trainer.build_model(cfg)
        CenterNetCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    """
    If you'd like to do anything fancier than the standard training logic,
    consider writing your own training loop (see plain_train_net.py) or
    subclassing the trainer.
    """
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    if cfg.TEST.AUG.ENABLED:
        trainer.register_hooks(
            [hooks.EvalHook(0, lambda: trainer.test_with_TTA(cfg, trainer.model))]
        )
    return trainer.train()


if __name__ == "__main__":
    detectron2_parser = default_argument_parser()
    args = add_dataset_parser(detectron2_parser).parse_args()
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )