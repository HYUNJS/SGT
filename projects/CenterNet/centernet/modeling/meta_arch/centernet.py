import math
import torch
import numpy as np
import os
import os.path as osp

from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from detectron2.structures import ImageList
from projects.CenterNet.centernet.centernet_loss import CenterNetLoss
from projects.Datasets.MOT.vis.simple_visualzation_demo import SimpleVisualizationDemo
from projects.CenterNet.centernet.build import build_centernet_backbone, build_centernet_upsample_layer, build_centernet_head
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.structures import Boxes
from detectron2.structures import Instances
from projects.CenterNet.centernet.generator import CenterNetDecoder
from projects.CenterNet.centernet.generator import CenterNetGT

__all__ = ["CenterNet"]


@META_ARCH_REGISTRY.register()
class CenterNet(nn.Module):

    def __init__(self, cfg):
        super().__init__()

        self.device = torch.device(cfg.MODEL.DEVICE)
        self.cfg = cfg

        self.identify_on = cfg.MODEL.IDENTITY_ON
        self.backbone = build_centernet_backbone(cfg)
        self.upsample = build_centernet_upsample_layer(cfg)
        self.head = build_centernet_head(cfg, self.upsample.out_channels)
        self.loss = CenterNetLoss(cfg)
        # self.freeze_loss_param()

        self.mean, self.std = cfg.MODEL.PIXEL_MEAN, cfg.MODEL.PIXEL_STD
        if cfg.INPUT.FORMAT == 'RGB':
            self.mean = self.mean[::-1]
            self.std = self.std[::-1]
        pixel_mean = torch.Tensor(self.mean).to(self.device).view(3, 1, 1)
        pixel_std = torch.Tensor(self.std).to(self.device).view(3, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        self.denormalizer = lambda x: x * pixel_std + pixel_mean
        self.to(self.device)

        self.input_align = cfg.MODEL.CENTERNET.INPUT_ALIGN
        self.num_classes = cfg.MODEL.CENTERNET.NUM_CLASSES
        self.reg_offset = cfg.MODEL.CENTERNET.REG_OFFSET
        self.down_ratio = cfg.MODEL.CENTERNET.DOWN_RATIO
        self.max_per_image = cfg.MODEL.CENTERNET.MAX_PER_IMAGE

        self.tlbr_flag = cfg.MODEL.CENTERNET.TLBR_FLAG
        self.norm_flag = cfg.INPUT.NORM_BY_MEAN_STD_FLAG

        self.inference_gt = cfg.MODEL.CENTERNET.INFERENCE_GT

    def freeze_loss_param(self):
        if self.identify_on:
            for p in self.loss.parameters():
                p.requires_grad = False

    def forward(self, batched_inputs, return_all=False):
        images = self.preprocess_image(batched_inputs)

        if not self.training:
            return self.inference(images, batched_inputs)

        features = self.backbone(images.tensor)
        up_fmap = self.upsample(features)
        if type(up_fmap).__name__ == 'list':
            up_fmap = up_fmap[-1]
        outputs = self.head(up_fmap)

        targets = self.get_ground_truth(batched_inputs)
        losses = self.loss(outputs, targets)
        # self.vis_heatmap(outputs, targets, batched_inputs, save_flag=False, K=100, id_features=up_fmap) # id_features=None to use reid head
        if return_all:
            losses = {'outputs': outputs, 'targets': targets, 'losses': losses, 'feat_map': up_fmap}
        return losses

    @torch.no_grad()
    def inference(self, images, batched_inputs, do_postprocess=True, return_all=False):
        assert not self.training

        aligned_img, img_info = self._align_image(images, batched_inputs)
        features = self.backbone(aligned_img)
        up_fmap = self.upsample(features)
        if type(up_fmap).__name__ == 'list':
            up_fmap = up_fmap[-1]
        results = self.head(up_fmap)

        if do_postprocess:
            gt_input = batched_inputs if self.inference_gt else None
            results = self._postprocess(results, img_info, gt_input)
        if return_all:
            results = {'outputs': results, 'feat_map': up_fmap, 'img_info': img_info}
        return results

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        if self.norm_flag:
            images = [self.normalizer(x / 255.) for x in images]
        else:
            images = [x / 255. for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images

    @torch.no_grad()
    def get_ground_truth(self, batched_inputs):
        return CenterNetGT.generate(self.cfg, batched_inputs, self.tlbr_flag)

    def _align_image(self, images, batched_inputs):
        if self.input_align:
            n, c, h, w = images.tensor.shape
            new_h, new_w = (h | 31) + 1, (w | 31) + 1
            center_wh = np.array([w // 2, h // 2], dtype=np.float32)
            size_wh = np.array([new_w, new_h], dtype=np.float32)
            img_info = dict(center=center_wh, size=size_wh,
                            height=new_h // self.down_ratio,
                            width=new_w // self.down_ratio)

            pad_value = [-x / y for x, y in zip(self.mean, self.std)]
            aligned_img = torch.Tensor(pad_value).reshape((1, -1, 1, 1)).expand(n, c, new_h, new_w)
            aligned_img = aligned_img.to(images.tensor.device)

            pad_w, pad_h = math.ceil((new_w - w) / 2), math.ceil((new_h - h) / 2)
            aligned_img[..., pad_h:h + pad_h, pad_w:w + pad_w] = images.tensor
        else:
            w = batched_inputs[0]['width']
            h = batched_inputs[0]['height']
            n, c, input_h, input_w = images.tensor.shape
            center_wh = np.array([w / 2., h / 2.], dtype=np.float32)
            size_wh = max(float(input_w) / float(input_h) * h, w) * 1.0
            img_info = dict(center=center_wh, size=size_wh,
                            height=input_h // self.down_ratio,
                            width=input_w // self.down_ratio)
            aligned_img = images.tensor

        return aligned_img, img_info

    def _postprocess(self, results, img_info, batched_inputs=None):
        if batched_inputs is not None and self.inference_gt:
            detections = CenterNetDecoder.decode_gt(results, batched_inputs)
        else:
            detections = CenterNetDecoder.decode(results, K=self.max_per_image, tlbr_flag=self.tlbr_flag)

        boxes = detections['bboxes']
        scores = detections['scores']
        classes = detections['classes']
        id_feature = detections.get('id_feat', None)
        scores = scores.reshape(-1)
        classes = classes.reshape(-1).to(torch.int64)

        boxes = CenterNetDecoder.transform_boxes(boxes, img_info)
        boxes = Boxes(boxes)

        output = dict(pred_boxes=boxes, scores=scores, pred_classes=classes)
        if self.identify_on:
            id_feature = id_feature.squeeze()
            output.update(dict(id_feature=id_feature))
        if 'fmap_feat' in detections:
            output.update(dict(fmap_feat=detections['fmap_feat'].squeeze()))

        ori_w, ori_h = img_info['center'] * 2
        det_instance = Instances((int(ori_h), int(ori_w)), **output)
        return [{"instances": det_instance}]

    @staticmethod
    def vis_heatmap(outputs, targets, input_for_vis, save_flag=False, K=500, save_root='', id_features=None):
        if save_flag and save_root == '':
            save_root = '/mnt/video_nfs4/jshyun/Fair_32_heatmap_vis/20210523'
        id_feat, id_feat_index = None, True
        fmap = outputs['hm'].detach().sigmoid()
        fmap_nms = CenterNetDecoder.pseudo_nms(fmap)
        scores, index, clses, ys, xs = CenterNetDecoder.topk_score(fmap_nms, K=K)
        if id_features is not None:
            save_root += '/wo_reid_head'
        else:
            assert outputs['id'] is not None
            id_features = outputs['id'].detach()
            save_root += '/reid_head'
        id_features = id_features.detach().permute(0, 2, 3, 1)
        for i in range(targets['hm'].size(0)):
            demo = SimpleVisualizationDemo()
            vis_output = demo.run_on_image(input_for_vis[i]['image'].permute(1, 2, 0).cpu().numpy()[:, :, ::-1],
                                           input_for_vis[i]['instances'])
            plt.subplot(2, 2, 1)
            plt.title("Detectron2 - img")
            plt.imshow(vis_output.get_image())
            plt.subplot(2, 2, 2)
            plt.title("GT heatmap")
            plt.imshow(targets['hm'][i][0].cpu().numpy(), cmap='gray')
            plt.subplot(2, 2, 3)
            plt.title("Pred heatmap >= 0.4")
            heatmap_pred_th = fmap[i][0] * (fmap[i][0] >= 0.4)
            plt.imshow(heatmap_pred_th.cpu().numpy(), cmap='gray')
            plt.subplot(2, 2, 4)
            plt.title("Top-{} heatmap".format(K))
            tgt_size = fmap[i][0].shape
            heatmap_ind = torch.zeros(tgt_size).flatten()
            heatmap_ind[index[i].cpu()] = 255
            heatmap_ind = heatmap_ind.reshape(tgt_size)
            plt.imshow(heatmap_ind.numpy(), cmap='gray')

            if save_flag:
                seq_name = input_for_vis[i]['sequence_name']
                file_name = input_for_vis[i]['file_name'].split('/')[-1]
                os.makedirs(osp.join(save_root, seq_name), exist_ok=True)
                if 'crowdhuman' in seq_name:
                    file_checker = 0
                    file_name_appender = lambda x: file_name[:-4] + '_%d'%x + file_name[-4:]
                    while osp.isfile(osp.join(save_root, seq_name, file_name_appender(file_checker))):
                        file_checker += 1
                    file_name = file_name_appender(file_checker)
                save_path = osp.join(save_root, seq_name, file_name)
                plt.savefig(save_path)
            else:
                plt.show()
            plt.close()

            if (i+1) % 2 == 0:
                id_feat_curr = id_features[i].flatten(0, 1)[index[i]]
                id_feat_curr = F.normalize(id_feat_curr, dim=1)
                simmap = torch.matmul(id_feat, id_feat_curr.t()).cpu().numpy()
                frame_idx += '_%d'%(input_for_vis[i]['frame_idx']+1)
                plt.figure()
                plt.title("Sparse Correlation Map")
                plt.imshow(simmap)
                if save_flag:
                    save_path = osp.join(save_root, seq_name, 'corr_map%s.jpg'%frame_idx)
                    plt.savefig(save_path)
                else:
                    plt.show()
                plt.close()
            else:
                id_feat = id_features[i].flatten(0, 1)[index[i]]
                id_feat = F.normalize(id_feat, dim=1)
                frame_idx = str(input_for_vis[i]['frame_idx']+1)