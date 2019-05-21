import random
import torch
import PIL
import numpy as np
import math
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


class TransToHM(object):
    def __call__(self, data):
        # trans anns to (hm,wh,reg) format
        max_objs = len(data[1])
        height, width = data[0].shape[1], data[0].shape[2]
        # init var
        hm = np.zeros((12, height, width), dtype=np.float32)
        wh = np.zeros((max_objs, 2), dtype=np.float32)
        reg = np.zeros((max_objs, 2), dtype=np.float32)
        ind = np.zeros((max_objs), dtype=np.int64)
        reg_mask = np.zeros((max_objs), dtype=np.uint8)
        for k in range(max_objs):
            an = data[1][k]
            # select bbox change box(x,y,w,h) to (x1,y1,x2,y2)
            bbox = an[0:4]
            bbox[2] += bbox[0]
            bbox[3] += bbox[1]
            # box class (object_category)
            cls_id = an[5]

            # cal and draw heatmap by gaussian
            h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
            if h > 0 and w > 0:
                # draw heatmap
                radius = F.gaussian_radius((math.ceil(h), math.ceil(w)))
                radius = max(0, int(radius))
                ct = np.array(
                    [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
                ct_int = ct.astype(np.int32)
                F.draw_umich_gaussian(hm[cls_id], ct_int, radius)
                # cal wh
                wh[k] = 1. * w, 1. * h
                ind[k] = ct_int[1] * width + ct_int[0]
                reg[k] = ct - ct_int
                reg_mask[k] = 1
            hm=torch.tensor(hm)
            wh=torch.tensor(wh)
            ind=torch.tensor(ind)
            reg=torch.tensor(reg)
            reg_mask=torch.tensor(reg_mask)
        return data[0], hm, wh, ind, reg, reg_mask
