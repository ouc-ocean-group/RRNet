import random
import torch
import PIL
import numpy as np
from . import functional as F
from torchvision.transforms import Compose


# All the input data in these transforms is a tuple consists of the image and the annotations.

class HorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, data):
        assert isinstance(data[0], torch.Tensor)
        if random.random() > self.p:
            return data
        else:
            w = data[0].size(2)
            return F.flip_img(data[0]), F.flip_annos(data[1], w)


class ToTensor(object):
    def __call__(self, data):
        return F.img_to_tensor(data[0]), F.annos_to_tensor(data[1])


class Normalize(object):
    def __init__(self, mean=(0, 0, 0), std=(1, 1, 1)):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        assert isinstance(data[0], torch.Tensor)
        return F.normalize(data[0], self.mean, self.std), data[1]


class RandomCrop(object):
    def __init__(self, size):
        self.h, self.w = size

    def __call__(self, data):
        assert isinstance(data[0], torch.Tensor)
        assert isinstance(data[1], torch.Tensor)

        h, w = data[0].size()[-2:]
        if (self.w, self.h) == (w, h):
            return data
        assert self.w < w and self.h < h

        rx, ry = random.random() * (w - self.w), random.random() * (h - self.h)
        crop_coordinate = int(ry), int(rx), int(ry) + self.h, int(rx) + self.w
        cropped_annos = F.crop_annos(data[1].clone(), crop_coordinate, self.h, self.w)
        if cropped_annos.size(0) == 0:
            # rand_idx = random.randint(0, data[1].size(0) - 1)
            rand_idx = 0
            include_bbox = data[1][rand_idx, :]
            y2, x2 = include_bbox[0] + np.random.uniform(-1, 1) * 0.3 * float(include_bbox[2]) + self.h, \
                     include_bbox[1] + np.random.uniform(-1, 1) * 0.3 * float(include_bbox[3]) + self.w
            offset_y, offset_x = max(y2 - h, 0), max(x2 - w, 0)
            y2, x2 = y2 - offset_y, x2 - offset_x
            crop_coordinate = y2 - self.h, x2 - self.w, y2, x2
            cropped_annos = F.crop_annos(data[1].clone(), crop_coordinate, self.h, self.w)
        cropped_img = F.crop_tensor(data[0], crop_coordinate)
        return cropped_img, cropped_annos


class ColorJitter(object):
    def __init__(self, brightness=0.5, contrast=0.5, saturation=0.5):
        self.brightness = [max(1 - brightness, 0), 1 + brightness]
        self.contrast = [max(1 - contrast, 0), 1 + contrast]
        self.saturation = [max(1 - saturation, 0), 1 + saturation]

    def __call__(self, data):
        assert isinstance(data[0], PIL.Image.Image) or \
               isinstance(data[0], PIL.PngImagePlugin.PngImageFile) or \
               isinstance(data[0], PIL.JpegImagePlugin.JpegImageFile)
        return F.color_jitter(data[0], self.brightness, self.contrast, self.saturation), data[1]
