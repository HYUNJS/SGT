import datetime
import logging
import time

import torch
import numpy as np

from detectron2.evaluation.evaluator import DatasetEvaluators, inference_context
from detectron2.utils.comm import get_world_size, all_gather
from detectron2.utils.logger import log_every_n_seconds


def add_loss_dict(a, b):
    if a is None:
        a = {}
    for k in b.keys():
        a[k] = b[k] if k not in a else a[k] + b[k]
    return a


def inference_on_dataset(cfg, model, data_loader, evaluator):
    """
    Run model on the data_loader and evaluate the metrics with evaluator.
    Also benchmark the inference speed of `model.forward` accurately.
    The model will be used in eval mode.

    Args:
        model (nn.Module): a module which accepts an object from
            `data_loader` and returns some outputs. It will be temporarily set to `eval` mode.

            If you wish to evaluate a model in `training` mode instead, you can
            wrap the given model and override its behavior of `.eval()` and `.train()`.
        data_loader: an iterable object with a length.
            The elements it generates will be the inputs to the model.
        evaluator (DatasetEvaluator): the evaluator to run. Use `None` if you only want
            to benchmark, but don't want to do any evaluation.

    Returns:
        The return value of `evaluator.evaluate()`
    """
    num_devices = get_world_size()
    logger = logging.getLogger(__name__)
    logger.info("Start inference on {} images".format(len(data_loader)))

    total = len(data_loader)  # inference data loader must have a fixed length
    if evaluator is None:
        # create a no-op evaluator
        evaluator = DatasetEvaluators([])
    evaluator.reset()

    num_warmup = min(5, total - 1)
    start_time = time.perf_counter()
    total_compute_time = 0

    len_dataset = len(data_loader)
    all_len_dataset = all_gather(len_dataset)
    max_len_dataset = max(all_len_dataset)
    with inference_context(model), torch.no_grad():
        for idx, inputs in enumerate(data_loader):
            if idx >= max_len_dataset:
                break
            if idx == num_warmup:
                start_time = time.perf_counter()
                total_compute_time = 0

            start_compute_time = time.perf_counter()
            outputs = model(inputs)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_compute_time += time.perf_counter() - start_compute_time
            evaluator.process(inputs, outputs)

            iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
            seconds_per_img = total_compute_time / iters_after_start
            if idx >= num_warmup * 2 or seconds_per_img > 5:
                total_seconds_per_img = (time.perf_counter() - start_time) / iters_after_start
                eta = datetime.timedelta(seconds=int(total_seconds_per_img * (total - idx - 1)))
                log_every_n_seconds(
                    logging.INFO,
                    "Inference done {}/{}. {:.4f} s / img. ETA={}".format(
                        idx + 1, total, seconds_per_img, str(eta)
                    ),
                    n=5,
                )

    # Measure the time only for this worker (before the synchronization barrier)
    total_time = time.perf_counter() - start_time
    total_time_str = str(datetime.timedelta(seconds=total_time))
    # NOTE this format is parsed by grep
    logger.info(
        "Total inference time: {} ({:.6f} s / img per device, on {} devices)".format(
            total_time_str, total_time / (total - num_warmup), num_devices
        )
    )
    total_compute_time_str = str(datetime.timedelta(seconds=int(total_compute_time)))
    logger.info(
        "Total inference pure compute time: {} ({:.6f} s / img per device, on {} devices)".format(
            total_compute_time_str, total_compute_time / (total - num_warmup), num_devices
        )
    )

    results = evaluator.evaluate()
    # An evaluator may return None when not in main process.
    # Replace it by an empty dict instead to make it easier for downstream code to handle
    if results is None:
        results = {}
    else:
        if len(results.keys()) == 3:
            tgt_trk_metrics = ['mota', 'idf1', 'mostly_tracked', 'partially_tracked', 'mostly_lost', 'num_false_positives', 'num_misses', 'num_switches', 'num_fragmentations']
            print_columns = ['above thres', 'all', 'MOTA', 'IDF1', 'MT', 'PT', 'ML', 'FP', 'FN', 'IDs', 'FM']
            ap_above_thres_key = [k for k in results.keys() if 'MOT-Det-Eval>=' in k][0]
            print_values = [results[ap_above_thres_key]['AP50'], results['MOT-Det-Eval']['AP50'], *[results['MOT-Tracking-eval'][k] for k in tgt_trk_metrics]]
            print_values[2] *= 100 # convert MOTA to percentage
            print_values[3] *= 100 # convert IDF1 to percentage
            print_values = [np.round(v, 1) for v in print_values]
            logger.info('[Record for Excel]')
            logger.info(print_columns)
            logger.info(print_values)
    return results
