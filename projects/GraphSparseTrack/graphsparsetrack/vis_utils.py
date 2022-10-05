import torch, os
import torch.nn.functional as F
import torchvision.ops as ops
import os.path as osp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fvcore.common.file_io import PathManager
from PIL import Image, ImageOps

from detectron2.data.detection_utils import convert_PIL_to_numpy
from detectron2.data import MetadataCatalog
from detectron2.structures import Boxes, Instances
from detectron2.utils.visualizer import _SMALL_OBJECT_AREA_THRESH, ColorMode, Visualizer
from projects.Datasets.MOT.vis.colormap import id2color
from projects.Datasets.MOT.vis.simple_visualzation_demo import SimpleVisualizationDemo
from projects.GraphSparseTrack.graphsparsetrack.utils import pseudo_nms


def vis_iou_sim_mat(obj_feats_t1, obj_feats_t2, boxes_t1, boxes_t2, batched_inputs, save_flag=False, save_root=''):
    '''
    vis_iou_sim_mat(obj_feats_t1, obj_feats_t2, boxes_t1, boxes_t2, batched_inputs, save_flag=True,
                    save_root='/mnt/video_nfs4/jshyun/Fair_32_heatmap_vis/20210525/wo_reid_head')
    '''
    N = boxes_t2.size(0)
    obj_feats_t1 = F.normalize(obj_feats_t1.detach(), dim=2)
    obj_feats_t2 = F.normalize(obj_feats_t2.detach(), dim=2)
    if save_flag and save_root == '':
        save_root = '/mnt/video_nfs4/jshyun/Fair_32_heatmap_vis/tmp'

    for i in range(N):
        iou_mat = ops.box_iou(boxes_t1[i].detach(), boxes_t2[i].detach())
        sim_mat = torch.matmul(obj_feats_t1[i], obj_feats_t2[i].t())
        plt.subplot(1, 2, 1)
        plt.title('iou')
        plt.imshow(iou_mat.cpu().numpy())
        plt.subplot(1, 2, 2)
        plt.title('sim')
        plt.imshow(sim_mat.cpu().numpy())
        if save_flag:
            seq_name = batched_inputs[i*2]['sequence_name']
            frame_idx_t1 = batched_inputs[i*2]['frame_idx'] + 1
            frame_idx_t2 = batched_inputs[i*2+1]['frame_idx'] + 1
            save_path = osp.join(save_root, seq_name, 'iou_sim_mat_%d_%d.jpg'%(frame_idx_t1, frame_idx_t2))
            plt.savefig(save_path)
        else:
            plt.show()
        plt.close()

def vis_heatmap_num_points(detector_outs, batched_inputs, topk_scores, topk_ind, save_flag=False, save_root=''):
    '''
    vis_heatmap_num_points(detector_outs, batched_inputs, topk_scores, topk_ind, save_flag=True,
                           save_root='/mnt/video_nfs4/jshyun/Fair_32_heatmap_vis/20210525/wo_reid_head')
    '''
    pred_th = 0.4
    id_maps = torch.zeros_like(detector_outs['outputs']['hm'], dtype=torch.int64).flatten(1)
    gt_ids = detector_outs['targets']['id']
    tgt_inds = detector_outs['targets']['ind']
    fmap = detector_outs['outputs']['hm'].sigmoid()
    fmap = pseudo_nms(fmap)
    heatmap_inds = torch.zeros(detector_outs['outputs']['hm'].shape, dtype=torch.int64).flatten(1)
    heatmap_inds_th = torch.zeros(detector_outs['outputs']['hm'].shape, dtype=torch.int64).flatten(1)
    topk_scores = topk_scores.squeeze()
    N = len(id_maps)
    for i in range(N):
        id_maps[i][tgt_inds[i]] = (gt_ids[i] > 0) * 255
        heatmap_inds[i][topk_ind[i].cpu().numpy()] = 255
        heatmap_inds_th[i][topk_ind[i][topk_scores[i] >= pred_th]] = 255
    heatmap_preds = detector_outs['outputs']['hm'].detach().sigmoid().squeeze()
    tgt_size = fmap[0][0].shape
    heatmap_inds = heatmap_inds.reshape(N, *tgt_size)
    heatmap_inds_th = heatmap_inds_th.reshape(N, *tgt_size)
    id_maps = id_maps.reshape(N, *tgt_size).detach().cpu()
    heatmap_preds = ((heatmap_preds >= pred_th) * 255).cpu()
    rgb_img = torch.cat([id_maps.unsqueeze(3), heatmap_inds.unsqueeze(3), heatmap_preds.unsqueeze(3)], dim=3)
    rgb_img_th = torch.cat([id_maps.unsqueeze(3), heatmap_inds_th.unsqueeze(3), heatmap_preds.unsqueeze(3)], dim=3)
    if save_flag and save_root == '':
        save_root = '/mnt/video_nfs4/jshyun/Fair_32_heatmap_vis/tmp'

    for i in range(N):
        num_id = (id_maps[i] != 0).sum()
        num_pred_th = (heatmap_preds[i] != 0).sum()
        num_topk = (heatmap_inds[i] != 0).sum()
        num_topk_th = (heatmap_inds_th[i] != 0).sum()
        num_id_pred_th = (id_maps[i][(heatmap_preds[i] != 0)] != 0).sum()
        num_id_topk = (id_maps[i][(heatmap_inds[i] != 0)] != 0).sum()
        num_id_topk_th = (id_maps[i][(heatmap_inds_th[i] != 0)] != 0).sum()
        plt.subplot(3, 2, 1)
        plt.title("GT ID heatmap (1)")
        plt.imshow(id_maps[i].numpy(), cmap='gray')
        plt.text(0, -1, '#1=%d' % num_id, fontsize=8, color='red')
        plt.subplot(3, 2, 2)
        plt.title("Pred Heatmap >= {} (2)".format(str(pred_th)))
        plt.imshow(heatmap_preds[i].numpy(), cmap='gray')
        plt.text(0, -1, '#2=%d' % num_pred_th, fontsize=8, color='red')
        plt.subplot(3, 2, 3)
        plt.title("Topk Heatmap w/ NMS (3)")
        plt.imshow(heatmap_inds[i].numpy(), cmap='gray')
        plt.text(0, -1, '#3=%d' % num_topk, fontsize=8, color='red')
        plt.subplot(3, 2, 4)
        plt.title("Topk Heatmap w/ NMS >= {} (4)".format(str(pred_th)))
        plt.imshow(heatmap_inds_th[i].numpy(), cmap='gray')
        plt.text(0, -1, '#4=%d' % num_topk_th, fontsize=8, color='red')
        plt.subplot(3, 2, 5)
        plt.title("Stacked RGB map (1+2+3)")
        plt.imshow(rgb_img[i])
        plt.text(0, -1, '#1&2=%d | #1&3=%d' % (num_id_pred_th, num_id_topk), fontsize=8, color='red')
        plt.subplot(3, 2, 6)
        plt.title("Stacked RGB map (1+2+4)")
        plt.imshow(rgb_img_th[i])
        plt.text(0, -1, '#1&2=%d | #1&4=%d' % (num_id_pred_th, num_id_topk_th), fontsize=8, color='red')
        if save_flag:
            seq_name = batched_inputs[i]['sequence_name']
            frame_idx = batched_inputs[i]['frame_idx'] + 1
            save_path = osp.join(save_root, seq_name, 'heatmap_num_points_%d.jpg'%frame_idx)
            plt.savefig(save_path)
        else:
            plt.show()
        plt.close()

def vis_node_sim(obj_feats_t1, obj_feats_t2, new_feats_t1, new_feats_t2):
    tmp_sim = torch.bmm(obj_feats_t1.detach(), obj_feats_t2.detach().permute(0, 2, 1))
    tmp_t1 = F.normalize(new_feats_t1.detach(), dim=1).reshape(6, 100, -1)
    tmp_t2 = F.normalize(new_feats_t2.detach(), dim=1).reshape(6, 100, -1)
    tmp_new_sim = torch.bmm(tmp_t1, tmp_t2.permute(0, 2, 1))
    for b_idx in range(tmp_new_sim.size(0)):
        plt.subplot(1, 2, 1); plt.title("Initial node feature similarity");
        plt.imshow(tmp_sim[b_idx].cpu().numpy()); plt.xlabel('T2'); plt.ylabel('T1');
        plt.subplot(1, 2, 2); plt.title("Updated node feature similarity");
        plt.imshow(tmp_new_sim[b_idx].cpu().numpy()); plt.xlabel('T2'); plt.ylabel('T1');
        plt.show(); plt.close();

def vis_input(batched_input):
    frame_name = batched_input['sequence_name'] + '-' + str(batched_input['frame_idx'] + 1)
    vis_demo = SimpleVisualizationDemo()
    img = batched_input['image'].permute(1,2,0).cpu().numpy()[:, :, ::-1]
    img_out = vis_demo.run_on_image(img, batched_input['instances'])
    plt.imshow(img_out.get_image())
    plt.title(frame_name)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def vis_topk_input_pairs(batched_input_list, remap_id=False, save_path=''):
    tensor2numpy_img_fn = lambda x: x.permute(1, 2, 0).cpu().numpy()[:, :, ::-1]
    concat_seqname_frame_idx_fn = lambda x: x['sequence_name'] + '-' + ''.join(x['file_name'].split('/')[-1].split('.')[:-1])

    for input_t1, input_t2 in zip(batched_input_list[::2], batched_input_list[1::2]):
        vis_demo = TopkVisualizerDemo()
        if remap_id:
            gt_ids_t1 = input_t1['instances'].gt_ids
            gt_ids_t2 = input_t2['instances'].gt_ids
            id_remap_dict = {gt_id: i for i, gt_id in enumerate(np.unique(np.concatenate([gt_ids_t1, gt_ids_t2])))}
            remapped_gt_ids_t1 = pd.Series(gt_ids_t1).map(id_remap_dict).to_numpy()
            remapped_gt_ids_t2 = pd.Series(gt_ids_t2).map(id_remap_dict).to_numpy()
            input_t1['instances'].gt_ids = remapped_gt_ids_t1
            input_t2['instances'].gt_ids = remapped_gt_ids_t2

        plt.subplot(2, 1, 1)
        input_t1_name = concat_seqname_frame_idx_fn(input_t1)
        img_t1 = tensor2numpy_img_fn(input_t1['image'])
        img_out = vis_demo.run_on_image(img_t1, input_t1['instances'])
        plt.imshow(img_out.get_image())
        plt.title(input_t1_name)
        plt.axis('off')
        plt.tight_layout()

        plt.subplot(2, 1, 2)
        input_t2_name = concat_seqname_frame_idx_fn(input_t2)
        img_t2 = tensor2numpy_img_fn(input_t2['image'])
        img_out = vis_demo.run_on_image(img_t2, input_t2['instances'])
        plt.imshow(img_out.get_image())
        plt.title(input_t2_name)
        plt.axis('off')
        plt.tight_layout(pad=0)
        if save_path != '':
            plt.savefig(osp.join(save_path, f'{input_t1_name}_{input_t2_name}.jpg'), bbox_inches='tight', pad_inches=0)
        else:
            plt.show()

class TopkVisualizerDemo(object):
    def __init__(self, metadata=None, instance_mode=ColorMode.IMAGE):
        self.metadata = metadata if metadata is not None else MetadataCatalog.get("__unused")
        self.sub_metadata = MetadataCatalog.get("__unused")
        self.cpu_device = torch.device("cpu")
        self.instance_mode = instance_mode

    def read_image(self, file_name, format=None):
         with PathManager.open(file_name, "rb") as f:
            image = Image.open(f)
            image = ImageOps.exif_transpose(image)
            return convert_PIL_to_numpy(image, format)

    def run_on_image(self, image, predictions):
        visualizer = TopkVisualizer(image, self.metadata, self.sub_metadata, instance_mode=self.instance_mode)
        if isinstance(predictions, Instances):
            instances = predictions.to(self.cpu_device)
        elif "instances" in predictions:
            instances = predictions["instances"].to(self.cpu_device)
        vis_output = visualizer.draw_instance_predictions(predictions=instances)
        return vis_output


class TopkVisualizer(Visualizer):
    def __init__(self, img_rgb, metadata, sub_metadata=None, scale=1.0, instance_mode=ColorMode.IMAGE):
        super().__init__(img_rgb, metadata, scale, instance_mode)
        self.sub_metadata = sub_metadata
        self.CLASS_NAMES = {1: 'P', 2: 'G'}
        # self.LABEL_TYPES = {1: 'P', 2: 'G'}

    def draw_instance_predictions(self, predictions):
        if predictions.has("pred_boxes"):
            boxes = predictions.pred_boxes
        elif predictions.has("gt_boxes"):
            boxes = predictions.gt_boxes
        else:
            boxes = None

        scores = predictions.scores if predictions.has("scores") else None

        if predictions.has("pred_classes"):
            classes = predictions.pred_classes
        elif predictions.has("gt_classes"):
            classes = predictions.gt_classes
        else:
            classes = None

        if predictions.has("pred_ids"):
            ids = predictions.pred_ids
        elif predictions.has("gt_ids"):
            ids = predictions.gt_ids
        else:
            ids = None

        if predictions.has("pred_keypoints"):
            keypoints = predictions.pred_keypoints
        elif predictions.has("gt_keypoints"):
            keypoints = predictions.gt_keypoints
        else:
            keypoints = None

        self.overlay_instances(
            masks=None,
            boxes=boxes,
            clses=classes,
            keypoints=keypoints,
            ids=ids,
            assigned_colors=None,
            alpha=0.5,
        )
        return self.output

    def overlay_instances(
            self,
            *,
            boxes=None,
            clses=None,
            masks=None,
            keypoints=None,
            ids=None,
            confidences=None,
            visibilities=None,
            assigned_colors=None,
            alpha=0.5
    ):
        """
        Args:
            boxes (Boxes, RotatedBoxes or ndarray): either a :class:`Boxes`,
                or an Nx4 numpy array of XYXY_ABS format for the N objects in a single image,
                or a :class:`RotatedBoxes`,
                or an Nx5 numpy array of (x_center, y_center, width, height, angle_degrees) format
                for the N objects in a single image,
            clses (list[long]): class label
            masks (masks-like object): Supported types are:

                * :class:`detectron2.structures.PolygonMasks`,
                  :class:`detectron2.structures.BitMasks`.
                * list[list[ndarray]]: contains the segmentation masks for all objects in one image.
                  The first level of the list corresponds to individual instances. The second
                  level to all the polygon that compose the instance, and the third level
                  to the polygon coordinates. The third level should have the format of
                  [x0, y0, x1, y1, ..., xn, yn] (n >= 3).
                * list[ndarray]: each ndarray is a binary mask of shape (H, W).
                * list[dict]: each dict is a COCO-style RLE.
            keypoints (Keypoint or array like): an array-like object of shape (N, K, 3),
                where the N is the number of instances and K is the number of keypoints.
                The last dimension corresponds to (x, y, visibility or score).
            assigned_colors (list[matplotlib.colors]): a list of colors, where each color
                corresponds to each mask or box in the image. Refer to 'matplotlib.colors'
                for full list of formats that the colors are accepted in.

        Returns:
            output (VisImage): image object with visualizations.
        """
        num_instances = None
        if boxes is not None:
            boxes = self._convert_boxes(boxes)
            num_instances = len(boxes)
        if masks is not None:
            masks = self._convert_masks(masks)
            if num_instances:
                assert len(masks) == num_instances
            else:
                num_instances = len(masks)
        if keypoints is not None:
            if num_instances:
                assert len(keypoints) == num_instances
            else:
                num_instances = len(keypoints)
            keypoints = self._convert_keypoints(keypoints)
        if ids is not None:
            ids = self._convert_ids(ids)
            assert len(ids) == num_instances
        if assigned_colors is None:
            if ids is None:
                assigned_colors = [id2color(0, rgb=True, maximum=1)] * num_instances
            else:
                assigned_colors = [id2color(ids[i], rgb=True, maximum=1) for i in range(num_instances)]
        if num_instances == 0:
            return self.output
        if boxes is not None and boxes.shape[1] == 5:
            return self.overlay_rotated_instances(
                boxes=boxes, labels=None, assigned_colors=assigned_colors
            )

        # Display in largest to smallest order to reduce occlusion.
        areas = None
        if boxes is not None:
            areas = np.prod(boxes[:, 2:] - boxes[:, :2], axis=1)
        elif masks is not None:
            areas = np.asarray([x.area() for x in masks])

        if areas is not None:
            sorted_idxs = np.argsort(-areas).tolist()
            # Re-order overlapped instances in descending order.
            boxes = boxes[sorted_idxs] if boxes is not None else None
            masks = [masks[idx] for idx in sorted_idxs] if masks is not None else None
            assigned_colors = [assigned_colors[idx] for idx in sorted_idxs]
            keypoints = keypoints[sorted_idxs] if keypoints is not None else None

            # Attributes used in MOT datasets
            ids = [ids[idx] for idx in sorted_idxs] if ids is not None else None
            clses = [clses[idx] for idx in sorted_idxs] if clses is not None else None
            confidences = [confidences[idx] for idx in sorted_idxs] if confidences is not None else None
            visibilities = [visibilities[idx] for idx in sorted_idxs] if visibilities is not None else None

        for i in range(num_instances):
            color = assigned_colors[i]
            if boxes is not None:
                self.draw_box(boxes[i], edge_color=color)

            if masks is not None:
                for segment in masks[i].polygons:
                    self.draw_polygon(segment.reshape(-1, 2), color, alpha=alpha)

            if not(ids is None and confidences is None and visibilities is None ):
                # first get a box
                if boxes is not None:
                    x0, y0, x1, y1 = boxes[i]
                    text_pos = (x0, y0)  # if drawing boxes, put text on the box corner.
                    horiz_align = "left"
                elif masks is not None:
                    x0, y0, x1, y1 = masks[i].bbox()

                    # draw text in the center (defined by median) when box is not drawn
                    # median is less sensitive to outliers.
                    text_pos = np.median(masks[i].mask.nonzero(), axis=1)[::-1]
                    horiz_align = "center"
                else:
                    continue  # drawing the box confidence for keypoints isn't very useful.
                # for small objects, draw text at the side to avoid occlusion
                instance_area = (y1 - y0) * (x1 - x0)
                if (
                        instance_area < _SMALL_OBJECT_AREA_THRESH * self.output.scale
                        or y1 - y0 < 40 * self.output.scale
                ):
                    if y1 >= self.output.height - 5:
                        text_pos = (x1, y0)
                    else:
                        text_pos = (x0, y1)

                height_ratio = (y1 - y0) / np.sqrt(self.output.height * self.output.width)
                lighter_color = self._change_color_brightness(color, brightness_factor=0.7)
                font_size = (
                        np.clip((height_ratio - 0.02) / 0.08 + 1, 1.2, 2)
                        * 0.5
                        * self._default_font_size * 2
                )
                # get a text for each instance
                text = ""
                if clses is not None and clses[i] != 0:
                    text += "{}_".format(self.CLASS_NAMES[clses[i]])
                if ids is not None and ids[i] != 0:
                    text += " {}".format(ids[i])
                if confidences is not None:
                    text += " conf:{}".format(confidences[i])
                if visibilities is not None:
                    text += " visi:{}".format(visibilities[i])

                self.draw_text(
                    text,
                    text_pos,
                    color=lighter_color,
                    horizontal_alignment=horiz_align,
                    font_size=font_size,
                )

        # draw keypoints
        if keypoints is not None:
            for keypoints_per_instance in keypoints:
                self.draw_and_connect_keypoints(keypoints_per_instance)

        return self.output

    def _convert_ids(self, ids):
        return np.asarray(ids)
