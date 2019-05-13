import random
import torch
import PIL
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

        return F.crop_tensor(data[0], crop_coordinate), F.crop_annos(data[1], crop_coordinate, self.h, self.w)


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

