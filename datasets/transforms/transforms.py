import random
import torch
from torch.nn.functional import pad
import PIL
import numpy as np
from . import functional as F
import math
import cv2
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
        if self.w > w or self.h > h:
            padded_img = pad(data[0], [0, self.w-w, 0, self.h-h])
            return padded_img, data[1]

        rx, ry = random.random() * (w - self.w), random.random() * (h - self.h)
        crop_coordinate = int(rx), int(ry), int(rx) + self.w, int(ry) + self.h
        cropped_annos = F.crop_annos(data[1].clone(), crop_coordinate, self.h, self.w)
        if cropped_annos.size(0) == 0:
            rand_idx = torch.randint(0, data[1].size(0), (1,))
            include_bbox = data[1][rand_idx, :].squeeze()
            x1, y1, x2, y2 = include_bbox[0], include_bbox[1], \
                             include_bbox[0] + include_bbox[2], include_bbox[1] + include_bbox[3]
            max_x1_ = min(x1, w-self.w)
            max_y1_ = min(y1, h-self.h)
            min_x1_ = max(0, x2-self.w)
            min_y1_ = max(0, y2-self.h)
            min_x1, max_x1 = sorted([max_x1_, min_x1_])
            min_y1, max_y1 = sorted([max_y1_, min_y1_])
            x1 = np.random.randint(min_x1, max_x1) if min_x1 != max_x1 else min_x1
            y1 = np.random.randint(min_y1, max_y1) if min_y1 != max_y1 else min_y1
            crop_coordinate = (int(x1), int(y1), int(x1) + self.w, int(y1) + self.h)
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



def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result


def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)

def get_affine_transform(center,
                         scale,
                         rot,
                         output_size,
                         shift=np.array([0, 0], dtype=np.float32),
                         inv=0):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale], dtype=np.float32)

    scale_tmp = scale
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

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

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.], dtype=np.float32).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]

class TransToHM(object):

    def __call__(self, data, max_n, data_n):
        # trans anns to (hm,wh,reg) format
        max_objs = max_n
        data_objs = data_n
        height, width = data[0].shape[1], data[0].shape[2]
        # init var
        hm = np.zeros((10, height // 4, width // 4), dtype=np.float32)
        # hm = np.zeros((10, height // 2, width // 2), dtype=np.float32)

        wh = np.zeros((max_objs, 2), dtype=np.float32)
        reg = np.zeros((max_objs, 2), dtype=np.float32)
        ind = np.zeros((max_objs), dtype=np.int64)
        reg_mask = np.zeros((max_objs), dtype=np.uint8)
        for k in range(data_n):
            an = data[1][k]
            # select bbox change box(x,y,w,h) to (x1,y1,x2,y2)
            bbox = an[0:4]
            bbox[2] += bbox[0]
            bbox[3] += bbox[1]
            bbox[0]  = bbox[0] // 4
            # bbox[0]  = bbox[0] // 2
            bbox[1]  = bbox[1] // 4
            # bbox[1]  = bbox[1] // 2
            bbox[2]  = bbox[2] // 4
            # bbox[2]  = bbox[2] // 2
            bbox[3]  = bbox[3] // 4
            # bbox[3]  = bbox[3] // 2

            # box class (object_category)
            cls_id = an[5] - 1

            # cal and draw heatmap by gaussian
            h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
            if h > 0 and w > 0:
                # draw heatmap
                radius = F.gaussian_radius((math.ceil(h), math.ceil(w)))
                radius = max(0, int(radius))
                ct = np.array(
                    [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
                ct_int = ct.astype(np.int32)

                F.draw_umich_gaussian(hm[int(cls_id)], ct_int, radius)
                # F.draw_umich_gaussian_withEllipse(hm[int(cls_id)], ct_int, radius, 1, w, h)
                # cal wh
                # wh.append()
                wh[k] = [1. * w, 1. * h]
                # ind[k] = ct_int[1] * (width // 2) + ct_int[0]
                ind[k] = ct_int[1] * (width // 4)+ ct_int[0]
                reg[k] = ct - ct_int
                reg_mask[k] = 1
        hm=torch.tensor(hm)
        wh=torch.tensor(wh)
        ind=torch.tensor(ind)
        reg=torch.tensor(reg)
        reg_mask=torch.tensor(reg_mask)
        return data[0], hm, wh, ind, reg, reg_mask


class TransToHM_Origin(object):

    def __call__(self, data, max_n, data_n):
        # trans anns to (hm,wh,reg) format
        max_objs = max_n
        data_objs = data_n
        height, width = data[0].shape[1], data[0].shape[2]
        c = np.array([data[0].shape[2] / 2., data[0].shape[1] / 2.], dtype=np.float32)
        s = np.array([data[0].shape[2], data[0].shape[1]], dtype=np.float32)
        trans_output = get_affine_transform(c, s, 0, [width // 4, height // 4])
        # init var
        hm = np.zeros((10, height // 4, width // 4), dtype=np.float32)
        # hm = np.zeros((10, height // 2, width // 2), dtype=np.float32)

        wh = np.zeros((max_objs, 2), dtype=np.float32)
        reg = np.zeros((max_objs, 2), dtype=np.float32)
        ind = np.zeros((max_objs), dtype=np.int64)
        reg_mask = np.zeros((max_objs), dtype=np.uint8)
        for k in range(data_n):
            an = data[1][k]
            # select bbox change box(x,y,w,h) to (x1,y1,x2,y2)
            bbox = np.array(an[0:4], dtype=np.float32)
            bbox[2] += bbox[0]
            bbox[3] += bbox[1]

            bbox[:2] = affine_transform(bbox[:2], trans_output)
            bbox[2:] = affine_transform(bbox[2:], trans_output)
            bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, width // 4 - 1)
            bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, height // 4 - 1)

            # box class (object_category)
            cls_id = an[5] - 1

            # cal and draw heatmap by gaussian
            h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
            if h > 0 and w > 0:
                # draw heatmap
                radius = F.gaussian_radius((math.ceil(h), math.ceil(w)))
                radius = max(0, int(radius))
                ct = np.array(
                    [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
                ct_int = ct.astype(np.int32)

                F.draw_umich_gaussian(hm[int(cls_id)], ct_int, radius)
                # F.draw_umich_gaussian_withEllipse(hm[int(cls_id)], ct_int, radius, 1, w, h)
                # cal wh
                # wh.append()
                wh[k] = [1. * w, 1. * h]
                # ind[k] = ct_int[1] * (width // 2) + ct_int[0]
                ind[k] = ct_int[1] * (width // 4)+ ct_int[0]
                reg[k] = ct - ct_int
                reg_mask[k] = 1
        hm=torch.tensor(hm)
        wh=torch.tensor(wh)
        ind=torch.tensor(ind)
        reg=torch.tensor(reg)
        reg_mask=torch.tensor(reg_mask)
        return data[0], hm, wh, ind, reg, reg_mask

class MaskIgnore(object):
    def __init__(self, mean=(0.485, 0.456, 0.406), ignore_idx=0):
        self.mean = mean
        self.ignore_idx = ignore_idx

    def __call__(self, data):
        assert isinstance(data[0], torch.Tensor)
        assert isinstance(data[1], torch.Tensor)

        return F.mask_ignore(data, self.mean, self.ignore_idx)

