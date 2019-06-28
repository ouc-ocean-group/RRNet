import random
import torch
from torch.nn.functional import pad
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


class NormalizeNTimes(object):
    def __init__(self, mean=(0, 0, 0), std=(1, 1, 1), times=4):
        self.mean = mean
        self.std = std
        self.times = times

    def __call__(self, data):
        assert len(data[0]) == self.times
        norm = [F.normalize(data[0][i][0], self.mean, self.std) for i in range(self.times)]
        annos = data[1]
        return norm, annos


class RandomCrop(object):
    def __init__(self, size):
        self.h, self.w = size

    def __call__(self, data):
        assert isinstance(data[0], torch.Tensor)
        assert isinstance(data[1], torch.Tensor)
        img = data[0]
        h, w = img.size()[-2:]
        if (self.w, self.h) == (w, h):
            return data
        if self.w > w or self.h > h:
            img = pad(img, [0, max(self.w - w, 0), 0, max(self.h - h, 0)])
        h, w = img.size()[-2:]
        rx, ry = random.random() * (w - self.w), random.random() * (h - self.h)
        crop_coordinate = int(rx), int(ry), int(rx) + self.w, int(ry) + self.h
        cropped_annos = F.crop_annos(data[1].clone(), crop_coordinate, self.h, self.w)
        if cropped_annos.size(0) == 0:
            rand_idx = torch.randint(0, data[1].size(0), (1,))
            include_bbox = data[1][rand_idx, :].squeeze()
            x1, y1, x2, y2 = include_bbox[0], include_bbox[1], \
                             include_bbox[0] + include_bbox[2], include_bbox[1] + include_bbox[3]
            max_x1_ = min(x1, w - self.w)
            max_y1_ = min(y1, h - self.h)
            min_x1_ = max(0, x2 - self.w)
            min_y1_ = max(0, y2 - self.h)
            min_x1, max_x1 = sorted([max_x1_, min_x1_])
            min_y1, max_y1 = sorted([max_y1_, min_y1_])
            x1 = np.random.randint(min_x1, max_x1) if min_x1 != max_x1 else min_x1
            y1 = np.random.randint(min_y1, max_y1) if min_y1 != max_y1 else min_y1
            crop_coordinate = (int(x1), int(y1), int(x1) + self.w, int(y1) + self.h)
            cropped_annos = F.crop_annos(data[1].clone(), crop_coordinate, self.h, self.w)
        cropped_img = F.crop_tensor(img, crop_coordinate)
        return cropped_img, cropped_annos


class RandomCropNTimes(object):
    def __init__(self, size, times=4):
        self.h, self.w = size
        self.times = times

    def __call__(self, data):
        assert isinstance(data[0], torch.Tensor)
        assert isinstance(data[1], torch.Tensor)

        img = data[0]
        h, w = img.size()[-2:]
        if (self.w, self.h) == (w, h):
            imgs = data[0].unsqueeze(0).repeat(self.times, 1, 1, 1)
            annos = data[1].unsqueeze(0).repeat(self.times, 1, 1)
            return imgs, annos
        if self.w > w or self.h > h:
            img = pad(img, [0, max(self.w - w, 0), 0, max(self.h - h, 0)])
        h, w = img.size()[-2:]
        cropped_imgs, cropped_annoss = [], []
        for t in range(self.times):
            rx, ry = random.random() * (w - self.w), random.random() * (h - self.h)
            crop_coordinate = int(rx), int(ry), int(rx) + self.w, int(ry) + self.h
            cropped_annos = F.crop_annos(data[1].clone(), crop_coordinate, self.h, self.w)
            if cropped_annos.size(0) == 0:
                rand_idx = torch.randint(0, data[1].size(0), (1,))
                include_bbox = data[1][rand_idx, :].squeeze()
                x1, y1, x2, y2 = include_bbox[0], include_bbox[1], \
                                 include_bbox[0] + include_bbox[2], include_bbox[1] + include_bbox[3]
                max_x1_ = min(x1, w - self.w)
                max_y1_ = min(y1, h - self.h)
                min_x1_ = max(0, x2 - self.w)
                min_y1_ = max(0, y2 - self.h)
                min_x1, max_x1 = sorted([max_x1_, min_x1_])
                min_y1, max_y1 = sorted([max_y1_, min_y1_])
                x1 = np.random.randint(min_x1, max_x1) if min_x1 != max_x1 else min_x1
                y1 = np.random.randint(min_y1, max_y1) if min_y1 != max_y1 else min_y1
                crop_coordinate = (int(x1), int(y1), int(x1) + self.w, int(y1) + self.h)
                cropped_annos = F.crop_annos(data[1].clone(), crop_coordinate, self.h, self.w)
            cropped_img = F.crop_tensor(img, crop_coordinate)
            cropped_imgs.append(cropped_img.unsqueeze(0))
            cropped_annoss.append(cropped_annos)
        return cropped_imgs, cropped_annoss


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


class MaskIgnore(object):
    def __init__(self, mean=(0.485, 0.456, 0.406), ignore_idx=0):
        self.mean = mean
        self.ignore_idx = ignore_idx

    def __call__(self, data):
        assert isinstance(data[0], torch.Tensor)
        assert isinstance(data[1], torch.Tensor)

        return F.mask_ignore(data, self.mean, self.ignore_idx)


class MaskIgnoreNTimes(object):
    def __init__(self, mean=(0.485, 0.456, 0.406), ignore_idx=0, times=4):
        self.mean = mean
        self.ignore_idx = ignore_idx
        self.times = times

    def __call__(self, data):
        assert len(data[0]) == self.times
        assert len(data[1]) == self.times

        ig_data = [list(F.mask_ignore((data[0][i], data[1][i]), self.mean, self.ignore_idx)) for i in range(self.times)]
        imgs, annos = [], []
        for i in range(self.times):
            imgs.append(ig_data[i][0])
            annos.append(ig_data[i][1])
        return imgs, annos


class Multiscale(object):
    def __init__(self, scale=(0.5, 0.75, 1, 1.25, 1.5)):
        self.scale = scale

    def __call__(self, data):
        randnum = random.randint(0, len(self.scale) - 1)
        return F.multiscale(data, self.scale[randnum])


class ToHeatmap(object):
    def __init__(self, scale_factor=4, cls_num=10):
        self.scale_factor = scale_factor
        self.cls_num = cls_num

    def __call__(self, data):
        img, annos, hm, wh, ind, offset, reg_mask = F.to_heatmap(data, self.scale_factor, self.cls_num)
        return img, annos, hm, wh, ind, offset, reg_mask


class To9BoxHeatmap(object):
    def __init__(self, scale_factor=4, bias_factor=0.5):
        self.scale_factor = scale_factor
        self.bias_factor = bias_factor

    def __call__(self, data):
        img, annos, hms, whs, inds, offset, reg_mask = F.to_9box_heatmap(data, self.scale_factor, self.bias_factor)
        return img, annos, hms, whs, inds, offset, reg_mask
