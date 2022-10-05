import cv2
import math
import random
import numpy as np

from fvcore.transforms.transform import Transform


class HueTransform(Transform):
    def __init__(self, fraction, S_rand, V_rand, format='BGR'):
        super().__init__()
        self._set_attributes(locals())
        if format == 'BGR':
            cvt_format_in = cv2.COLOR_BGR2HSV
            cvt_format_out = cv2.COLOR_HSV2BGR
        elif format == 'RGB':
            cvt_format_in = cv2.COLOR_RGB2HSV
            cvt_format_out = cv2.COLOR_HSV2RGB
        else:
            raise Exception("Image format only accepts BGR and RGB")
        self.cvt_format_in = cvt_format_in
        self.cvt_format_out = cvt_format_out

    def apply_coords(self, coords):
        return coords

    def apply_segmentation(self, segmentation):
        return segmentation

    def apply_image(self, img, interp=None):
        return self.hue(img)

    def hue(self, img):
        img_hsv = cv2.cvtColor(img, self.cvt_format_in)
        S = img_hsv[:, :, 1].astype(np.float32)
        V = img_hsv[:, :, 2].astype(np.float32)

        a = (self.S_rand * 2 - 1) * self.fraction + 1
        S *= a
        if a > 1:
            np.clip(S, a_min=0, a_max=255, out=S)

        a = (self.V_rand * 2 - 1) * self.fraction + 1
        V *= a
        if a > 1:
            np.clip(V, a_min=0, a_max=255, out=V)

        img_hsv[:, :, 1] = S.astype(np.uint8)
        img_hsv[:, :, 2] = V.astype(np.uint8)
        return cv2.cvtColor(img_hsv, self.cvt_format_out)

    def inverse(self):
        return NotImplemented


class LetterBoxTransform(Transform):

    def __init__(self, new_shape, dst_constant, ratio, padding, interpolation=None, border_type=None, color=(127.5, 127.5, 127.5)):
        super().__init__()
        self._set_attributes(locals())
        self.interpolation = cv2.INTER_AREA if interpolation is None else interpolation
        self.border_type = cv2.BORDER_CONSTANT if border_type is None else border_type
        self.color = color

    def apply_image(self, img, interp=None):
        top, bottom, left, right = self.dst_constant
        img = cv2.resize(img, self.new_shape, interpolation=self.interpolation)  # resized, no border
        ret = cv2.copyMakeBorder(img, top, bottom, left, right, self.border_type, value=self.color)  # padded rectangular
        return ret

    def apply_coords(self, coords):
        pad_w, pad_h = self.padding
        coords[:, 0] = self.ratio * coords[:, 0] + pad_w
        coords[:, 1] = self.ratio * coords[:, 1] + pad_h
        return coords

    def apply_box(self, coords):
        padw, padh = self.padding
        coords_out = coords.copy()
        coords_out[:, 0] = self.ratio * coords[:, 0] + padw
        coords_out[:, 1] = self.ratio * coords[:, 1] + padh
        coords_out[:, 2] = self.ratio * coords[:, 2] + padw
        coords_out[:, 3] = self.ratio * coords[:, 3] + padh
        return coords_out

    def apply_segmentation(self, segmentation):
        return NotImplemented

    def inverse(self):
        return NotImplemented


class AffineTransform(Transform):
    """
    Augmentation from CenterNet
    """

    def __init__(self, src, dst, output_size, border_value=[0, 0, 0]):
        """
        output_size:(w, h)
        """
        super().__init__()
        affine = cv2.getAffineTransform(np.float32(src), np.float32(dst))
        self._set_attributes(locals())

    def apply_image(self, img: np.ndarray) -> np.ndarray:
        """
        Apply AffineTransform for the image(s).

        Args:
            img (ndarray): of shape HxW, HxWxC, or NxHxWxC. The array can be
                of type uint8 in range [0, 255], or floating point in range
                [0, 1] or [0, 255].
        Returns:
            ndarray: the image(s) after applying affine transform.
        """
        return cv2.warpAffine(img, self.affine, self.output_size, flags=cv2.INTER_LINEAR, borderValue=self.border_value)

    def apply_coords(self, coords: np.ndarray) -> np.ndarray:
        """
        Affine the coordinates.

        Args:
            coords (ndarray): floating point array of shape Nx2. Each row is
                (x, y).
        Returns:
            ndarray: the flipped coordinates.

        Note:
            The inputs are floating point coordinates, not pixel indices.
            Therefore they are flipped by `(W - x, H - y)`, not
            `(W - 1 - x, H 1 - y)`.
        """
        # aug_coord (N, 3) shape, self.affine (2, 3) shape
        w, h = self.output_size
        aug_coords = np.column_stack((coords, np.ones(coords.shape[0])))
        coords = np.dot(aug_coords, self.affine.T)
        coords[..., 0] = np.clip(coords[..., 0], 0, w - 1)
        coords[..., 1] = np.clip(coords[..., 1], 0, h - 1)
        return coords

    def apply_segmentation(self, segmentation):
        return NotImplemented

    def inverse(self):
        # return AffineTransform(self.dst, self.src, self.output_size)
        return NotImplemented

class FairAffineTransform(Transform):
    """
    Affine Transformation from FairMOT official repository
    """
    def __init__(self, width, height, border_value, M, a):
        super().__init__()
        self._set_attributes(locals())

    def apply_image(self, img: np.ndarray) -> np.ndarray:
        return cv2.warpPerspective(img, self.M, dsize=(self.width, self.height), flags=cv2.INTER_LINEAR, borderValue=self.border_value)  # BGR order borderValue

    def apply_coords(self, coords: np.ndarray) -> np.ndarray:
        return coords

    def apply_box(self, coords: np.ndarray) -> np.ndarray:
        if len(coords) > 0:
            n = coords.shape[0]
            points = coords.copy()
            area0 = (points[:, 2] - points[:, 0]) * (points[:, 3] - points[:, 1])

            # warp points
            xy = np.ones((n * 4, 3))
            xy[:, :2] = points[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
            xy = (xy @ self.M.T)[:, :2].reshape(n, 8)

            # create new boxes
            x = xy[:, [0, 2, 4, 6]]
            y = xy[:, [1, 3, 5, 7]]
            xy = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

            # apply angle-based reduction
            radians = self.a * math.pi / 180
            reduction = max(abs(math.sin(radians)), abs(math.cos(radians))) ** 0.5
            x = (xy[:, 2] + xy[:, 0]) / 2
            y = (xy[:, 3] + xy[:, 1]) / 2
            w = (xy[:, 2] - xy[:, 0]) * reduction
            h = (xy[:, 3] - xy[:, 1]) * reduction
            xy = np.concatenate((x - w / 2, y - h / 2, x + w / 2, y + h / 2)).reshape(4, n).T

            # reject warped points outside of image
            #np.clip(xy[:, 0], 0, width, out=xy[:, 0])
            #np.clip(xy[:, 2], 0, width, out=xy[:, 2])
            #np.clip(xy[:, 1], 0, height, out=xy[:, 1])
            #np.clip(xy[:, 3], 0, height, out=xy[:, 3])
            w = xy[:, 2] - xy[:, 0]
            h = xy[:, 3] - xy[:, 1]
            area = w * h
            ar = np.maximum(w / (h + 1e-16), h / (w + 1e-16))
            i = (w > 4) & (h > 4) & (area / (area0 + 1e-16) > 0.1) & (ar < 10)

            return xy[i]
        else:
            return coords

    def apply_segmentation(self, segmentation):
        return NotImplemented

    def inverse(self):
        # return AffineTransform(self.dst, self.src, self.output_size)
        return NotImplemented