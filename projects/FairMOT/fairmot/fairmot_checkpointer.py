from torch.nn.parallel import DistributedDataParallel
from projects.CenterNet.centernet.checkpoint.centernet_checkpoint import CenterNetCheckpointer


class FairMOTCheckpointer(CenterNetCheckpointer):
    def __init__(self, model, save_dir="", **kwargs):
        if isinstance(model, DistributedDataParallel):
            model = model.module
        super().__init__(model.detector, save_dir, **kwargs)

