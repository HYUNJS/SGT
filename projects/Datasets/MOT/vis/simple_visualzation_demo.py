import os
import cv2
import copy
import torch
from PIL import Image, ImageOps
from fvcore.common.file_io import PathManager

from detectron2.utils.visualizer import ColorMode
from detectron2.data import MetadataCatalog
from detectron2.data.detection_utils import convert_PIL_to_numpy
from detectron2.structures import Boxes, Instances
from projects.Datasets.MOT.vis.simple_visualizer import SimpleVisualizer, imwrite, imshow


class SimpleVisualizationDemo(object):
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
        visualizer = SimpleVisualizer(image, self.metadata, self.sub_metadata, instance_mode=self.instance_mode)
        if isinstance(predictions, Instances):
            instances = predictions.to(self.cpu_device)
        elif "instances" in predictions:
            instances = predictions["instances"].to(self.cpu_device)
        vis_output = visualizer.draw_instance_predictions(predictions=instances)
        return vis_output

    def visualize_input(self, inputs, show_flag=True):
        vis_input_list = []
        for input_per_image in inputs:
            image = self.read_image(input_per_image['file_name'], "RGB")
            if 'transforms' in input_per_image:
                transforms = input_per_image['transforms']
                for t in transforms.transforms:
                    image = t.apply_image(image)
            vis_output = self.run_on_image(image, input_per_image['instances'])
            if show_flag:
                imshow(vis_output.get_image())
            else:
                vis_input_list.append(vis_output.get_image())
        if not show_flag:
            return vis_input_list

    def visualize_output(self, inputs, outputs, post_process=True, output_dir=None):
        with torch.no_grad():
            if not isinstance(inputs, list):
                inputs = [inputs]
            if not isinstance(outputs, list):
                outputs = [outputs]
            for input_per_image, output_per_image in zip(inputs, outputs):
                if "ori_image" in input_per_image:
                    image = input_per_image["ori_image"]
                    if input_per_image["image_format"] == "BGR":
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                else:
                    image = self.read_image(input_per_image['file_name'], "RGB")
                instances = copy.deepcopy(output_per_image['instances'])
                if (not post_process) and ('transforms' in input_per_image):
                    transforms = input_per_image['transforms']
                    instances._image_size = image.shape[:2]
                    for t in transforms.transforms:
                        image = t.apply_image(image)
                        # instances = self.transform_instances(instances, t)
                vis_output = self.run_on_image(image, instances)
                if output_dir is None:
                    imshow(vis_output.get_image())
                else:
                    seq_dir = os.path.join(output_dir, input_per_image["sequence_name"])
                    os.makedirs(seq_dir, exist_ok=True)
                    file_name = "%06d.jpg" % (input_per_image['frame_idx'] + 1)
                    output_path = os.path.join(seq_dir, file_name)
                    imwrite(output_path, vis_output.get_image())

    def transform_instances(self, instances, transforms):
        boxes = instances.pred_boxes.tensor
        device = boxes.device
        np_boxes = boxes.cpu().data.numpy()
        np_boxes = transforms.apply_box(np_boxes).clip(min=0)
        new_boxes = torch.from_numpy(np_boxes).to(device)
        instances.pred_boxes = Boxes(new_boxes)
        return instances
