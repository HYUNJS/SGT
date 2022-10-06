import re, os, pickle, torch
import torch.nn as nn
from typing import Any, Dict, List, Optional
from fvcore.common.checkpoint import Checkpointer, _IncompatibleKeys
from fvcore.common.file_io import PathManager
from torch.nn.parallel import DistributedDataParallel

import detectron2.utils.comm as comm


def append_prefix(state_dict: Dict[str, Any], prefix: str) -> None:
    keys = sorted(state_dict.keys())
    if not all(len(key) == 0 or not key.startswith(prefix) for key in keys):
        return

    for key in keys:
        newkey = prefix + key
        state_dict[newkey] = state_dict.pop(key)

    # also strip the prefix in metadata, if any..
    try:
        metadata = state_dict._metadata  # pyre-ignore
    except AttributeError:
        pass
    else:
        for key in list(metadata.keys()):
            # for the metadata dict, the key can be:
            # '': for the DDP module, which we want to remove.
            # 'module': for the actual model.
            # 'module.xx.xx': for the rest.

            if len(key) == 0:
                continue
            newkey = prefix + key
            metadata[newkey] = metadata.pop(key)

class SGTCheckPointer(Checkpointer):
    def __init__(self, model, save_dir="",  *, save_to_disk=None, **checkpointables):
        if isinstance(model, DistributedDataParallel):
            model = model.module
        # super().__init__(model.detector, save_dir, **kwargs)
        # tracker = model.tracker
        is_main_process = comm.is_main_process()
        super().__init__(
            model,
            save_dir,
            save_to_disk=is_main_process if save_to_disk is None else save_to_disk,
            **checkpointables,
        )
        self.whole_weight_flag = False
        self.backbone_name = self.get_backbone_name() # default by dla
        backbone_convert_fn_dict = {
            'resnet': self._convert_weight_name_resnet,
            'dla': self._convert_weight_name_dla,
            'hourglass': self._convert_weight_name_hourglass,
        }
        self.convert_weight_name_fn = backbone_convert_fn_dict[self.backbone_name]
        self.dcnv2_flag = False

    def get_backbone_name(self):
        backbone_name = 'dla'
        backbone_class_name = type(self.model.detector.backbone).__name__.lower()
        if 'dla' in backbone_class_name:
            backbone_name = 'dla'
        elif 'resnet' in backbone_class_name:
            backbone_name = 'resnet'
        elif 'hourglass' in backbone_class_name:
            backbone_name = 'hourglass'
        else:
            raise NotImplementedError(f"Loading backbone {backbone_class_name} is not yet supported")
        return backbone_name

    def resume_or_load(self, paths: Dict[str, str], *, resume: bool = True) -> Dict[str, Any]:
        if paths['total'] != '':
            self.whole_weight_flag = True
            path = paths['total']
        else:
            assert paths['detector'] != '', "Specify either detector weight or total model weight"
            path = paths['detector']
        if resume and self.has_checkpoint():
            path = self.get_checkpoint_file()
            return self.load(path)
        else:
            return self.load(path, checkpointables=[])

    def load(self, path: str, checkpointables: Optional[List[str]] = None) -> Dict[str, Any]:
        if not path:
            # no checkpoint provided
            self.logger.info("No checkpoint found. Initializing model from scratch")
            return {}
        self.logger.info("Loading checkpoint from {}".format(path))
        if not os.path.isfile(path):
            path = self.path_manager.get_local_path(path)
            assert os.path.isfile(path), "Checkpoint {} not found!".format(path)

        checkpoint = self._load_file(path)
        incompatible = self._load_model(checkpoint)
        if (
                incompatible is not None
        ):  # handle some existing subclasses that returns None
            self._log_incompatible_keys(incompatible)

        for key in self.checkpointables if checkpointables is None else checkpointables:
            if key in checkpoint:
                self.logger.info("Loading {} from {}".format(key, path))
                obj = self.checkpointables[key]
                obj.load_state_dict(checkpoint.pop(key))

        # return any further checkpoint data
        return checkpoint

    def _load_model(self, checkpoint: Any) -> _IncompatibleKeys:
        """
        Load weights from a checkpoint.

            ckpt trained on this detectron2 framework: checkpoint_state_dict['model']
            Official COCO pretrained CenterNet ckpt: checkpoint_state_dict['model']['state_dict']

        Args:
            checkpoint (Any): checkpoint contains the weights.

        Returns:
            ``NamedTuple`` with ``missing_keys``, ``unexpected_keys``,
                and ``incorrect_shapes`` fields:
                * **missing_keys** is a list of str containing the missing keys
                * **unexpected_keys** is a list of str containing the unexpected keys
                * **incorrect_shapes** is a list of (key, shape in checkpoint, shape in model)

            This is just like the return value of
            :func:`torch.nn.Module.load_state_dict`, but with extra support
            for ``incorrect_shapes``.
        """
        checkpoint_state_dict = checkpoint
        if 'model' in checkpoint_state_dict:
            checkpoint_state_dict = checkpoint_state_dict['model']
        if 'state_dict' in checkpoint_state_dict:
            checkpoint_state_dict = checkpoint_state_dict['state_dict']
        model_state_dict = self.model.state_dict()
        converted_checkpoint_state_dict = self.convert_weight_name(checkpoint_state_dict, model_state_dict)

        checkpoint = {'model': converted_checkpoint_state_dict}
        incompatible = super()._load_model(checkpoint)
        if incompatible is None:  # support older versions of fvcore
            return None

        model_buffers = dict(self.model.named_buffers(recurse=False))
        for k in ["pixel_mean", "pixel_std"]:
            # Ignore missing key message about pixel_mean/std.
            # Though they may be missing in old checkpoints, they will be correctly
            # initialized from config anyway.
            if k in model_buffers:
                try:
                    incompatible.missing_keys.remove(k)
                except ValueError:
                    pass
        return incompatible

    def convert_weight_name(self, checkpoint_dict, model_state_dict):
        msg = 'If you see this, your model does not fully load the pre-trained weight'
        state_dict = {}
        for k in model_state_dict.keys():
            if 'dcnv2' in k:
                self.dcnv2_flag = True
                break

        for k, v in checkpoint_dict.items():
            new_k = self.convert_weight_name_fn(k)
            if not self.whole_weight_flag:
                new_k = 'detector.' + new_k
            if new_k in model_state_dict:
                if checkpoint_dict[k].shape == model_state_dict[new_k].shape:
                    state_dict[new_k] = checkpoint_dict[k]
                else:
                    state_dict[new_k] = model_state_dict[new_k]
                    self.logger.info('{} param in checkpoint is not matched with {} in model.'.format(k, new_k) + msg)
                    self.logger.info('{}.shape: {} || {}.shape: {}'
                                     .format(k, checkpoint_dict[k].shape, new_k, model_state_dict[new_k].shape))
            else:
                self.logger.info('{} param in checkpoint is not matched with {} in model.'.format(k, new_k) + msg)
        return state_dict

    def _convert_weight_name_dla(self, k):
        new_k = k
        if 'module' in k:
            # if use the pretrained weight provided by CenterNet author's repo (like ctdet_coco_dla_2x.pth)
            if 'module.base' in k:
                new_k = re.sub('module\.base\.', 'backbone.', k)
            elif 'module.dla_up' in k:
                new_k = re.sub('module\.dla_up', 'upsample.dla_up', k)
            elif 'module.ida_up' in k:
                new_k = re.sub('module\.ida_up', 'upsample.ida_up', k)
            elif 'module.hm' in k:
                if '0' in k:
                    new_k = re.sub('module\.hm\.0', 'head.cls_head.feat_conv', k)
                elif '2' in k:
                    new_k = re.sub('module\.hm\.2', 'head.cls_head.out_conv', k)
            elif 'module.wh' in k:
                if '0' in k:
                    new_k = re.sub('module\.wh\.0', 'head.wh_head.feat_conv', k)
                elif '2' in k:
                    new_k = re.sub('module\.wh\.2', 'head.wh_head.out_conv', k)
            elif 'module.reg' in k:
                if '0' in k:
                    new_k = re.sub('module\.reg\.0', 'head.reg_head.feat_conv', k)
                elif '2' in k:
                    new_k = re.sub('module\.reg\.2', 'head.reg_head.out_conv', k)
            elif 'module.backbone' in k:
                new_k = k.replace('module.', '')
            elif 'module.upsample' in k:
                new_k = k.replace('module.', '')
            elif 'module.head' in k:
                new_k = k.replace('module.', '')
        else:
            # if use the pretrained weight provided by FairMOT author's repo (like all_dla34.pth)
            if 'base' in k:
                new_k = re.sub('base\.', 'backbone.', k)
            elif 'upsample' not in k:
                if 'dla_up' in k:
                    new_k = re.sub('dla_up', 'upsample.dla_up', k)
                elif 'ida_up' in k:
                    new_k = re.sub('ida_up', 'upsample.ida_up', k)
            else:
                if 'hm' in k:
                    if '0' in k:
                        new_k = re.sub('hm\.0', 'head.cls_head.feat_conv', k)
                    elif '2' in k:
                        new_k = re.sub('hm\.2', 'head.cls_head.out_conv', k)
                elif 'wh' in k:
                    if '0' in k:
                        new_k = re.sub('wh\.0', 'head.wh_head.feat_conv', k)
                    elif '2' in k:
                        new_k = re.sub('wh\.2', 'head.wh_head.out_conv', k)
                elif 'reg' in k:
                    if '0' in k:
                        new_k = re.sub('reg\.0', 'head.reg_head.feat_conv', k)
                    elif '2' in k:
                        new_k = re.sub('reg\.2', 'head.reg_head.out_conv', k)
                elif 'id' in k:
                    if '0' in k:
                        new_k = re.sub('id\.0', 'head.id_head.feat_conv', k)
                    elif '2' in k:
                        new_k = re.sub('id\.2', 'head.id_head.out_conv', k)
        return new_k

    def _convert_weight_name_resnet(self, k):
        ## resnet_dcn version
        new_k = k
        if 'module' in k:
            # if use the pretrained weight provided by CenterNet author's repo (like ctdet_coco_resdcn18.pth)
            ## initial conv
            if 'module.conv' in k or 'module.bn' in k:
                new_k = re.sub('module', 'backbone.stage0', k)
            ## conv blocks
            elif 'module.layer' in k:
                new_k = re.sub('module\.layer', 'backbone.stage', k)
            ## deconv layers
            elif 'module.deconv_layers' in k:
                ori_layer_idx = int(k.split('.')[2])
                new_layer_idx = ori_layer_idx % 6
                new_level_idx = ori_layer_idx // 6 + 1
                if new_layer_idx == 0:
                    if 'conv_offset_mask' in k:
                        offset_ver = 'offset_mask_conv' if self.dcnv2_flag else 'offset_conv'
                        new_k = re.sub('module\.deconv_layers\.{}\.conv_offset_mask'.format(ori_layer_idx), 'upsample.deconv{}.dcn.{}'.format(new_level_idx, offset_ver), k)
                    else:
                        dcv_ver = 'dcnv2' if self.dcnv2_flag else 'dcn'
                        new_k = re.sub('module\.deconv_layers\.{}'.format(ori_layer_idx), 'upsample.deconv{}.dcn.{}'.format(new_level_idx, dcv_ver), k)
                elif new_layer_idx == 1:
                    new_k = re.sub('module\.deconv_layers\.{}'.format(ori_layer_idx), 'upsample.deconv{}.dcn_bn'.format(new_level_idx), k)
                elif new_layer_idx == 4:
                    new_k = re.sub('module\.deconv_layers\.{}'.format(ori_layer_idx), 'upsample.deconv{}.up_bn'.format(new_level_idx), k)
                elif new_layer_idx == 3:
                    new_k = re.sub('module\.deconv_layers\.{}'.format(ori_layer_idx), 'upsample.deconv{}.up_sample'.format(new_level_idx), k)
            ## head layers
            elif 'module.hm' in k:
                if '0' in k:
                    new_k = re.sub('module\.hm\.0', 'head.cls_head.feat_conv', k)
                elif '2' in k:
                    new_k = re.sub('module\.hm\.2', 'head.cls_head.out_conv', k)
            elif 'module.wh' in k:
                if '0' in k:
                    new_k = re.sub('module\.wh\.0', 'head.wh_head.feat_conv', k)
                elif '2' in k:
                    new_k = re.sub('module\.wh\.2', 'head.wh_head.out_conv', k)
            elif 'module.reg' in k:
                if '0' in k:
                    new_k = re.sub('module\.reg\.0', 'head.reg_head.feat_conv', k)
                elif '2' in k:
                    new_k = re.sub('module\.reg\.2', 'head.reg_head.out_conv', k)
        return new_k

    def _convert_weight_name_hourglass(self, k):
        ## Hourglass version
        new_k = k
        if 'module' in k:
            if 'hm.1' in k:
                if 'hm.1.0.conv' in k:
                    new_k = re.sub('module.hm.1.0.conv', 'head.cls_head.feat_conv', k)
                elif 'hm.1.1' in k:
                    new_k = re.sub('module.hm.1.1', 'head.cls_head.out_conv', k)
            elif 'wh.1' in k:
                if 'wh.1.0.conv' in k:
                    new_k = re.sub('module.wh.1.0.conv', 'head.wh_head.feat_conv', k)
                elif 'wh.1.1' in k:
                    new_k = re.sub('module.wh.1.1', 'head.wh_head.out_conv', k)
            elif 'reg.1' in k:
                if 'reg.1.0.conv' in k:
                    new_k = re.sub('module.reg.1.0.conv', 'head.reg_head.feat_conv', k)
                elif 'reg.1.1' in k:
                    new_k = re.sub('module.reg.1.1', 'head.reg_head.out_conv', k)
            else:
                new_k = re.sub('module.', 'backbone.', k)
        return new_k