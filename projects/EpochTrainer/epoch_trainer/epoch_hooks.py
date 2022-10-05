import detectron2.utils.comm as comm

from detectron2.engine.hooks import PeriodicWriter, EvalHook
from detectron2.evaluation.testing import flatten_results_dict


__all__ = [
    "EpochPeriodicWriter",
    "EpochEvalHook",
]


class EpochPeriodicWriter(PeriodicWriter):

    def __init__(self, writers, period=20):
        super().__init__(writers, period)

    def after_step(self):
        if (self.trainer.iter + 1) % self._period == 0 or (
            self.trainer.iter == self.trainer.end_iter - 1
        ):
            for writer in self._writers:
                writer.write()

    def after_train(self):
        for writer in self._writers:
            # If any new data is found (e.g. produced by other after_train),
            # write them before closing
            writer.write()
            if self.trainer.iter >= self.trainer.max_iter:
                writer.close()


class EpochEvalHook(EvalHook):

    def __init__(self, eval_period, eval_function):
        super().__init__(eval_period, eval_function)

    def _do_eval(self):
        results = self._func()

        if results:
            assert isinstance(
                results, dict
            ), "Eval function must return a dict. Got {} instead.".format(results)

            flattened_results = flatten_results_dict(results)
            for k, v in flattened_results.items():
                try:
                    v = float(v)
                except Exception as e:
                    raise ValueError(
                        "[EvalHook] eval_function should return a nested dict of float. "
                        "Got '{}: {}' instead.".format(k, v)
                    ) from e
            self.trainer.storage.put_scalars(**flattened_results, smoothing_hint=False)

        # Evaluation may take different time among workers.
        # A barrier make them start the next iteration together.
        comm.synchronize()

    def after_step(self):
        next_iter = self.trainer.iter + 1
        is_final = next_iter == self.trainer.max_iter
        if (self._period > 0 and is_final) or (self._period > 0 and next_iter % self._period == 0):
            self._do_eval()

    def after_train(self):
        if self.trainer.iter >= self.trainer.max_iter:
            del self._func