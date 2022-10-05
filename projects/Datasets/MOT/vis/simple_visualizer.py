import math
import torch
from PIL import Image
import numpy as np
import matplotlib as mpl
import matplotlib.colors as mplc
import matplotlib.pyplot as plt

from detectron2.structures import BoxMode
from detectron2.utils.visualizer import _SMALL_OBJECT_AREA_THRESH, ColorMode, Visualizer, _create_text_labels
from projects.Datasets.MOT.vis.colormap import id2color


def one_imshow(img, format, show):
    if isinstance(img, torch.Tensor):
        image_np = img.cpu().data.numpy()
    else:
        image_np = img

    if isinstance(image_np, np.ndarray):
        if len(image_np.shape) == 3:
            if image_np.dtype != np.uint8:
                image_np = image_np * 255
                image_np = image_np.astype('uint8')
            image_np = np.squeeze(image_np)
            if image_np.shape[0] == 3:
                image_np = image_np.transpose([1, 2, 0])
            if format.upper() == 'BGR':
                image_np = image_np[:, :, ::-1]
            image_pil = Image.fromarray(image_np, 'RGB')
        else:
            image_pil = image_np
    else:
        image_pil = image_np

    plt.imshow(image_pil)
    if show:
        plt.show()

def imshow(imgs, format='RGB', show=True):
    if isinstance(imgs, list):
        for img in imgs:
            one_imshow(img, format, show)
    else:
        img = imgs
        one_imshow(img, format, show)

def imwrite(filename, img):
    if 'jpg' in filename or 'png' in filename:
        if isinstance(img, torch.Tensor):
            image_np = img.cpu().data.numpy() * 255
            image_np = np.squeeze(image_np)
            if len(image_np.shape) == 3:
                image_np = image_np.transpose([1, 2, 0])
                image_pil = Image.fromarray(image_np.astype('uint8'), 'RGB')
            elif len(image_np.shape) == 2:
                image_pil = Image.fromarray(image_np.astype('uint8'), 'L')
            else:
                image_pil = image_np
        elif isinstance(img, np.ndarray):
            if img.shape[0] == 3:
                img = img.transpose([1, 2, 0])
            image_pil = Image.fromarray(img.astype('uint8'), 'RGB')
        else:
            image_pil = img
        image_pil.save(filename)

class SimpleVisualizer(Visualizer):
    def __init__(self, img_rgb, metadata, sub_metadata=None, scale=1.0, instance_mode=ColorMode.IMAGE, vis_conf=False, tgt_ignore_id=0):
        super().__init__(img_rgb, metadata, scale, instance_mode)
        self.sub_metadata = sub_metadata
        self.vis_conf = vis_conf
        self.tgt_ignore_id = tgt_ignore_id

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

        labels = _create_text_labels(classes, scores, self.metadata.get("thing_classes", None))
        confidences = scores if self.vis_conf else None
        self.overlay_instances(
            masks=None,
            boxes=boxes,
            labels=labels,
            keypoints=keypoints,
            ids=ids,
            assigned_colors=None,
            alpha=0.5,
            confidences=confidences,
            tgt_ignore_id=self.tgt_ignore_id
        )
        return self.output

    def draw_dataset_dict_simple(self, dic):
        annos = dic.get("annotations", None)
        if annos:
            boxes = [BoxMode.convert(x["bbox"], x["bbox_mode"], BoxMode.XYXY_ABS) for x in annos]
            labels = [x["category_id"] for x in annos]
            ids = [x['track_id'] for x in annos]
            confidences = None
            visibilities = [x['visibility'] for x in annos] if self.vis_conf else None
            self.overlay_instances(labels=labels, boxes=boxes, ids=ids, confidences=confidences, visibilities=visibilities)
        return self.output

    def overlay_instances(
            self,
            *,
            boxes=None,
            labels=None,
            masks=None,
            keypoints=None,
            ids=None,
            confidences=None,
            visibilities=None,
            assigned_colors=None,
            alpha=0.5,
            tgt_ignore_id=0
    ):
        """
        Args:
            boxes (Boxes, RotatedBoxes or ndarray): either a :class:`Boxes`,
                or an Nx4 numpy array of XYXY_ABS format for the N objects in a single image,
                or a :class:`RotatedBoxes`,
                or an Nx5 numpy array of (x_center, y_center, width, height, angle_degrees) format
                for the N objects in a single image,
            labels (list[str]): the text to be displayed for each instance.
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
        if labels is not None:
            assert len(labels) == num_instances
        if assigned_colors is None:
            if ids is None:
                assigned_colors = [id2color(0, rgb=True, maximum=1)] * num_instances
            else:
                assigned_colors = [id2color(ids[i], rgb=True, maximum=1) for i in range(num_instances)]
        if num_instances == 0:
            return self.output
        if boxes is not None and boxes.shape[1] == 5:
            return self.overlay_rotated_instances(
                boxes=boxes, labels=labels, assigned_colors=assigned_colors
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
            labels = [labels[k] for k in sorted_idxs] if labels is not None else None
            masks = [masks[idx] for idx in sorted_idxs] if masks is not None else None
            assigned_colors = [assigned_colors[idx] for idx in sorted_idxs]
            keypoints = keypoints[sorted_idxs] if keypoints is not None else None

            # Attributes used in MOT datasets
            ids = [ids[idx] for idx in sorted_idxs] if ids is not None else None
            confidences = [confidences[idx] for idx in sorted_idxs] if confidences is not None else None
            visibilities = [visibilities[idx] for idx in sorted_idxs] if visibilities is not None else None

        for i in range(num_instances):
            color = assigned_colors[i]
            if boxes is not None:
                self.draw_box(boxes[i], edge_color=color)

            if masks is not None:
                for segment in masks[i].polygons:
                    self.draw_polygon(segment.reshape(-1, 2), color, alpha=alpha)

            # if labels is not None:
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
                # font_size = (
                #         np.clip((height_ratio - 0.02) / 0.08 + 1, 1.2, 2)
                #         * 0.5
                #         * self._default_font_size * 2
                # )
                font_size = 12.0
                # get a text for each instance
                text = ""
                text_list = []
                if ids is not None and ids[i] != tgt_ignore_id:
                    id_text = "{}".format(ids[i])
                    # text += id_text
                    text_list.append(id_text)
                if confidences is not None:
                    conf_text = "{:.2f}".format(confidences[i])
                    # text += "-"+conf_text
                    text_list.append(conf_text)
                if visibilities is not None:
                    vis_text = "vis:{:.2f}".format(visibilities[i])
                    # text += "-"+vis_text
                    text_list.append(vis_text)
                text = '-'.join(text_list)

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

    def draw_text(
            self,
            text,
            position,
            *,
            font_size=None,
            color="g",
            horizontal_alignment="center",
            rotation=0
    ):
        """
        Args:
            text (str): class label
            position (tuple): a tuple of the x and y coordinates to place text on image.
            font_size (int, optional): font of the text. If not provided, a font size
                proportional to the image width is calculated and used.
            color: color of the text. Refer to `matplotlib.colors` for full list
                of formats that are accepted.
            horizontal_alignment (str): see `matplotlib.text.Text`
            rotation: rotation angle in degrees CCW

        Returns:
            output (VisImage): image object with text drawn.
        """
        if not font_size:
            font_size = self._default_font_size

        # since the text background is dark, we don't want the text to be dark
        color = np.maximum(list(mplc.to_rgb(color)), 0.2)
        color[np.argmax(color)] = max(0.8, np.max(color))

        x, y = position
        self.output.ax.text(
            x,
            y,
            text,
            size=font_size * self.output.scale,
            family="sans-serif",
            bbox={"facecolor": "black", "alpha": 0.8, "pad": 0.7, "edgecolor": "none"},
            verticalalignment="top",
            horizontalalignment=horizontal_alignment,
            color=color,
            zorder=10,
            rotation=rotation,
        )
        return self.output

    def draw_box(self, box_coord, alpha=0.5, edge_color="g", line_style="-"):
        """
        Args:
            box_coord (tuple): a tuple containing x0, y0, x1, y1 coordinates, where x0 and y0
                are the coordinates of the image's top left corner. x1 and y1 are the
                coordinates of the image's bottom right corner.
            alpha (float): blending efficient. Smaller values lead to more transparent masks.
            edge_color: color of the outline of the box. Refer to `matplotlib.colors`
                for full list of formats that are accepted.
            line_style (string): the string to use to create the outline of the boxes.

        Returns:
            output (VisImage): image object with box drawn.
        """
        x0, y0, x1, y1 = box_coord
        width = x1 - x0
        height = y1 - y0

        linewidth = max(self._default_font_size / 4, 1)

        self.output.ax.add_patch(
            mpl.patches.Rectangle(
                (x0, y0),
                width,
                height,
                fill=False,
                edgecolor=edge_color,
                linewidth=linewidth * self.output.scale,
                alpha=alpha,
                linestyle=line_style,
            )
        )
        return self.output

    def draw_rotated_box_with_label(
            self, rotated_box, alpha=0.5, edge_color="g", line_style="-", label=None
    ):
        """
        Args:
            rotated_box (tuple): a tuple containing (cnt_x, cnt_y, w, h, angle),
                where cnt_x and cnt_y are the center coordinates of the box.
                w and h are the width and height of the box. angle represents how
                many degrees the box is rotated CCW with regard to the 0-degree box.
            alpha (float): blending efficient. Smaller values lead to more transparent masks.
            edge_color: color of the outline of the box. Refer to `matplotlib.colors`
                for full list of formats that are accepted.
            line_style (string): the string to use to create the outline of the boxes.
            label (string): label for rotated box. It will not be rendered when set to None.

        Returns:
            output (VisImage): image object with box drawn.
        """
        cnt_x, cnt_y, w, h, angle = rotated_box
        area = w * h
        # use thinner lines when the box is small
        linewidth = self._default_font_size / (
            6 if area < _SMALL_OBJECT_AREA_THRESH * self.output.scale else 3
        )

        theta = angle * math.pi / 180.0
        c = math.cos(theta)
        s = math.sin(theta)
        rect = [(-w / 2, h / 2), (-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2)]
        # x: left->right ; y: top->down
        rotated_rect = [(s * yy + c * xx + cnt_x, c * yy - s * xx + cnt_y) for (xx, yy) in rect]
        for k in range(4):
            j = (k + 1) % 4
            self.draw_line(
                [rotated_rect[k][0], rotated_rect[j][0]],
                [rotated_rect[k][1], rotated_rect[j][1]],
                color=edge_color,
                linestyle="--" if k == 1 else line_style,
                linewidth=linewidth,
            )

        if label is not None:
            text_pos = rotated_rect[1]  # topleft corner

            height_ratio = h / np.sqrt(self.output.height * self.output.width)
            label_color = self._change_color_brightness(edge_color, brightness_factor=0.7)
            font_size = (
                    np.clip((height_ratio - 0.02) / 0.08 + 1, 1.2, 2) * 0.5 * self._default_font_size
            )
            self.draw_text(label, text_pos, color=label_color, font_size=font_size, rotation=angle)

        return self.output

    def _convert_ids(self, ids):
        return np.asarray(ids)
