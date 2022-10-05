import cv2
import math
import random
import numpy as np

from detectron2.data.transforms.augmentation import Augmentation
from projects.Datasets.Transforms.transform import HueTransform, LetterBoxTransform, AffineTransform, FairAffineTransform
from fvcore.transforms.transform import NoOpTransform


def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)
    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs
    return src_result


def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


class RandomHue(Augmentation):

    def __init__(self, format, fraction=0.5, prob=0.5):
        super().__init__()
        self._init(locals())

    def get_transform(self, img):
        do = self._rand_range() < self.prob
        if do:
            S_rand, V_rand = random.random(), random.random()
            return HueTransform(self.fraction, S_rand, V_rand, self.format)
        else:
            return NoOpTransform()

class LetterBox(Augmentation):

    def __init__(self, image_size, pad_color=(127.5, 127.5, 127.5)):
        super().__init__()
        self.height = image_size[0]
        self.width = image_size[1]
        self.pad_color = pad_color

    def get_transform(self, img):
        shape = img.shape[:2] # shape = [height, width]
        ratio = min(float(self.height) / shape[0], float(self.width) / shape[1])
        new_shape = (round(shape[1] * ratio), round(shape[0] * ratio))  # new_shape = [width, height]
        dw = (self.width - new_shape[0]) / 2.0  # width padding
        dh = (self.height - new_shape[1]) / 2.0  # height padding
        top, bottom = round(dh - 0.1), round(dh + 0.1)
        left, right = round(dw - 0.1), round(dw + 0.1)
        dst_constant = (top, bottom, left, right)
        padding = (dw, dh)
        return LetterBoxTransform(new_shape, dst_constant, ratio, padding, color=self.pad_color)


class CenterAffine(Augmentation):
    """
    Affine Transform for CenterNet
    """

    def __init__(self, border, output_size, random_aug=True):
        """
        Args:
            border(int): border size of image
            output_size(tuple): a tuple represents (width, height) of image
            random_aug(bool): whether apply random augmentation on annos or not
        """
        super().__init__()
        self._init(locals())

    def get_transform(self, img):
        """
        generate one `AffineTransform` for input image
        """
        img_shape = img.shape[:2]
        center, scale = self.generate_center_and_scale(img_shape)
        src, dst = self.generate_src_and_dst(center, scale, self.output_size)
        return AffineTransform(src, dst, self.output_size)

    @staticmethod
    def _get_border(border, size):
        """
        decide the border size of image
        """
        # NOTE This func may be reimplemented in the future
        i = 1
        size //= 2
        while size <= border // i:
            i *= 2
        return border // i

    def generate_center_and_scale(self, img_shape):
        r"""
        generate center and scale for image randomly

        Args:
            shape(tuple): a tuple represents (height, width) of image
        """
        height, width = img_shape
        center = np.array([width / 2, height / 2], dtype=np.float32)
        scale = float(max(img_shape))
        if self.random_aug:
            scale = scale * np.random.choice(np.arange(0.6, 1.4, 0.1))
            h_border = self._get_border(self.border, height)
            w_border = self._get_border(self.border, width)
            center[0] = np.random.randint(low=w_border, high=width - w_border)
            center[1] = np.random.randint(low=h_border, high=height - h_border)
        else:
            raise NotImplementedError("Non-random augmentation not implemented")

        return center, scale

    @staticmethod
    def generate_src_and_dst(center, size, output_size):
        r"""
        generate source and destination for affine transform
        """
        if not isinstance(size, np.ndarray) and not isinstance(size, list):
            size = np.array([size, size], dtype=np.float32)
        src = np.zeros((3, 2), dtype=np.float32)
        src_w = size[0]
        src_dir = [0, src_w * -0.5]
        src[0, :] = center
        src[1, :] = src[0, :] + src_dir
        src[2, :] = src[1, :] + (src_dir[1], -src_dir[0])

        dst = np.zeros((3, 2), dtype=np.float32)
        dst_w, dst_h = output_size
        dst_dir = [0, dst_w * -0.5]
        dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
        dst[1, :] = dst[0, :] + dst_dir
        dst[2, :] = dst[1, :] + (dst_dir[1], -dst_dir[0])
        return src, dst


class RandomAffine(Augmentation):

    def __init__(self, scale, shift, degree=0, prob=0.5, border_value=[0, 0, 0]):
        super().__init__()
        self.scale_factor = scale
        self.shift_factor = shift
        self.prob = prob
        self.degree_factor = degree
        self.border_value = border_value

    def get_transform(self, img):
        do = self._rand_range() < self.prob
        if do:
            height, width = img.shape[:2]
            center, scale, rot = self.generate_center_scale_and_rot(height, width)
            src, dst = self.generate_src_and_dst(center, scale, rot, height, width)
            return AffineTransform(src, dst, (width, height), self.border_value)
        else:
            return NoOpTransform()

    def generate_center_scale_and_rot(self, height, width):
        center = np.array([width / 2, height / 2], dtype=np.float32)
        scale = float(max(width, height))
        # shift_ratio = np.clip(np.random.randn() * self.shift_factor, -2 * self.shift_factor, 2 * self.shift_factor)
        width_ratio = np.clip(np.random.randn() * self.shift_factor, -2 * self.shift_factor, 2 * self.shift_factor)
        height_ratio = np.clip(np.random.randn() * self.shift_factor, -2 * self.shift_factor, 2 * self.shift_factor)
        scale_ratio = np.clip(np.random.randn() * self.scale_factor + 1, 1 - self.scale_factor, 1 + self.scale_factor)
        # center[0] += width * shift_ratio
        # center[1] += height * shift_ratio
        center[0] += scale * width_ratio
        center[1] += scale * height_ratio
        scale = scale * scale_ratio
        rot = np.clip(np.random.randn() * self.degree_factor, - self.degree_factor * 2, self.degree_factor * 2)
        return center, scale, rot

    @staticmethod
    def generate_src_and_dst(center, scale, rot, height, width, shift=np.array([0, 0], dtype=np.float32)):

        if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
            scale = np.array([scale, scale], dtype=np.float32)

        scale_tmp = scale
        src_w = scale_tmp[0]
        dst_w = width
        dst_h = height

        rot_rad = np.pi * rot / 180
        src_dir = get_dir([0, src_w * -0.5], rot_rad)
        dst_dir = np.array([0, dst_w * -0.5], np.float32)

        src = np.zeros((3, 2), dtype=np.float32)
        dst = np.zeros((3, 2), dtype=np.float32)
        src[0, :] = center + scale_tmp * shift
        src[1, :] = center + src_dir + scale_tmp * shift
        dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
        dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5], np.float32) + dst_dir

        src[2:, :] = get_3rd_point(src[0, :], src[1, :])
        dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])
        return src, dst


class FairAffine(Augmentation):
    def __init__(self, prob=1.0, degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-2, 2), border_value=(127.5, 127.5, 127.5)):
        super().__init__()
        self._init(locals())

    def get_transform(self, img):
        do = self._rand_range() < self.prob
        if do:
            height, width = img.shape[:2]
            M, a, s = self.get_warp_matrix(height, width)
            return FairAffineTransform(width, height, self.border_value, M, a)
        else:
            return NoOpTransform()

    def get_warp_matrix(self, height, width):
        border = 0  # width of added border (optional)
        # Rotation and Scale
        R = np.eye(3)
        a = random.random() * (self.degrees[1] - self.degrees[0]) + self.degrees[0]
        # a += random.choice([-180, -90, 0, 90])  # 90deg rotations added to small rotations
        s = random.random() * (self.scale[1] - self.scale[0]) + self.scale[0]
        R[:2] = cv2.getRotationMatrix2D(angle=a, center=(height / 2, width / 2), scale=s)

        # Translation
        T = np.eye(3)
        T[0, 2] = (random.random() * 2 - 1) * self.translate[0] * height + border  # x translation (pixels)
        T[1, 2] = (random.random() * 2 - 1) * self.translate[1] * width + border  # y translation (pixels)

        # Shear
        S = np.eye(3)
        S[0, 1] = math.tan((random.random() * (self.shear[1] - self.shear[0]) + self.shear[0]) * math.pi / 180)  # x shear (deg)
        S[1, 0] = math.tan((random.random() * (self.shear[1] - self.shear[0]) + self.shear[0]) * math.pi / 180)  # y shear (deg)

        M = S @ T @ R  # Combined rotation matrix. ORDER IS IMPORTANT HERE!!
        return M, a, s