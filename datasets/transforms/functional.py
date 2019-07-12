import torch
import random
from PIL import Image
import PIL.ImageEnhance as ImageEnhance
import torchvision.transforms.functional as torchtransform
from utils.metrics.metrics import bbox_iou
import numpy as np
import torch.nn.functional as F
import math
import cv2


def flip_img(data):
    """
    Flip Image Tensor.
    :param data: Image tensor.
    :return: Flipped Image.
    """
    return data.flip(dims=(2,))


def flip_annos(data, w):
    """
    Flip annotations.
    :param data: annotation tensor.
    :return: Flipped annotations.
    """
    data[:, 0] = w - data[:, 0] - data[:, 2]
    return data


def img_to_tensor(data):
    """
    Transform PIL Image to torch tensor.
    :param data: PIL Image.
    :return: tensor
    """
    return torchtransform.to_tensor(data)


def annos_to_tensor(data):
    """
    Transform annotations list to tensor.
    :param data: annotations list.
    :return: annotations tensor.
    """
    annos = []
    if isinstance(data[0], str):
        for d in data:
            split_d = [int(x) for x in d.strip().split(',')]
            annos.append(split_d)
    else:
        annos = data
    annos_tensor = torch.tensor(annos).float()
    # annos_tensor[:, [0, 1, 2, 3]] = annos_tensor[:, [1, 0, 3, 2]]
    return annos_tensor


def roadmap_to_tensor(data):
    """
    Transform road map to tensor.
    :param data: road map ndarray.
    :return: road map tensor.
    """
    if data is not None:
        roadmap_tensor = torch.tensor(data[:, :, 0]).float() / 255
    else:
        roadmap_tensor = None
    return roadmap_tensor


def resize(data, scale_factor):
    img = data[0]
    anno = data[1]
    roadmap = data[2]
    height, width = img.size[1], img.size[0]
    out_height, out_width = int(height*scale_factor), int(width*scale_factor)
    if roadmap is not None:
        roadmap = cv2.resize(roadmap, (out_width, out_height), interpolation=cv2.INTER_NEAREST)
    img = img.resize((out_width, out_height), Image.BILINEAR)
    anno[:, :4] = anno[:, :4] * scale_factor
    return img, anno, roadmap


def get_img_size(data):
    """
    Return the size of the input data.
    :param data: PIL Image.
    :return: tensor
    """
    return data.size


def crop_pil(data, coordinate):
    """
    Crop the PIL Image.
    :param data: PIL Image.
    :param coordinate: crop coordinate.
    :return: cropped image.
    """
    return data.crop(coordinate)


def crop_tensor(data, crop_coor):
    """
    Crop the torch tensor.
    :param data: tensor.
    :param crop_coor: crop coordinate.
    :return: cropped tensor.
    """
    return data[:, int(crop_coor[1]):int(crop_coor[3]), int(crop_coor[0]):int(crop_coor[2])]


def crop_annos(data, crop_coor, h, w):
    """
    Crop the annotations tensor.
    :param data: annotations tensor: xywh
    :param crop_coor: crop coordinate: xywh
    :return: cropped annotations tensor xywh.
    """
    # Here we need to use iou to get the valid bounding box in cropped area.
    crop_coor_tensor = torch.tensor(crop_coor).float().unsqueeze(0)
    data[:, 2:4] = data[:, :2] + data[:, 2:4]
    data[:, :4] -= crop_coor_tensor[:, :2].repeat(1, 2)
    data[data[:, 0] < 0, 0] = 0
    data[data[:, 1] < 0, 1] = 0
    data[data[:, 2] > w, 2] = w
    data[data[:, 3] > h, 3] = h

    data[:, 2] = data[:, 2] - data[:, 0]
    data[:, 3] = data[:, 3] - data[:, 1]
    return data


def normalize(data, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    """
    Normalize the input data.
    :param data: tensor.
    :param mean: mean value.
    :param std: std value.
    :return: normalized data.
    """
    return torchtransform.normalize(data, mean, std)


def denormalize(data, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    """
    Denormalize the input data.
    :param data: tensor.
    :param mean: mean value.
    :param std: std value.
    :return: denormalized data.
    """
    data = torchtransform.normalize(data, mean=[0., 0., 0.], std=[1 / x for x in std])
    data = torchtransform.normalize(data, mean=[-1 * x for x in mean], std=[1., 1., 1.])
    return data


def color_jitter(data, brightness, contrast, saturation):
    """
    Color Jitter for the input data.
    :param data: tensor.
    :param brightness: brightness value.
    :param contrast: contrast value.
    :param saturation: saturation value.
    :return: denormalized data.
    """
    r_brightness = random.uniform(brightness[0], brightness[1])
    r_contrast = random.uniform(contrast[0], contrast[1])
    r_saturation = random.uniform(saturation[0], saturation[1])
    im = ImageEnhance.Brightness(data).enhance(r_brightness)
    im = ImageEnhance.Contrast(im).enhance(r_contrast)
    im = ImageEnhance.Color(im).enhance(r_saturation)
    return im


def gaussian_radius(det_size, min_overlap=0.7):
    height, width = det_size

    a1 = 1
    b1 = (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = (b1 ** 2 - 4 * a1 * c1).sqrt()
    r1 = (b1 + sq1) / 2.

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = (b2 ** 2 - 4 * a2 * c2).sqrt()
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = (b3 ** 2 - 4 * a3 * c3).sqrt()
    r3 = (b3 + sq3) / 2
    r = torch.cat((r1, r2, r3), dim=1).min(dim=1)[0]
    return r


def gaussian2d(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    m = m.numpy()
    n = n.numpy()
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2 * sigma.numpy() * sigma.numpy()))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    h = torch.from_numpy(h).float()
    return h


def draw_umich_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1

    gaussian = gaussian2d((diameter, diameter), sigma=diameter / 6)

    x, y = center[0], center[1]

    height, width = heatmap.size()[0:2]
    left, right = torch.min(x, radius), torch.min(width - x, radius + 1)
    top, bottom = torch.min(y, radius), torch.min(height - y, radius + 1)

    masked_heatmap = heatmap[int(y - top):int(y + bottom), int(x - left):int(x + right)]
    masked_gaussian = gaussian[int(radius - top):int(radius + bottom), int(radius - left):int(radius + right)]
    if min(list(masked_gaussian.size())) > 0 and min(list(masked_heatmap.size())) > 0:
        torch.max(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap


def to_heatmap(data, scale_factor=4, cls_num=10):
    """
    Transform annotations to heatmap.
    :param data: (img, annos), tensor
    :param scale_factor:
    :param cls_num:
    :return:
    """
    img = data[0]
    annos = data[1].clone()

    h, w = img.size(1), img.size(2)

    hm = torch.zeros(cls_num, h // scale_factor, w // scale_factor)

    annos[:, 2] += annos[:, 0]
    annos[:, 3] += annos[:, 1]
    annos[:, :4] = annos[:, :4] / scale_factor
    cls_idx = annos[:, 5] - 1
    bboxs_h, bboxs_w = annos[:, 3:4] - annos[:, 1:2], annos[:, 2:3] - annos[:, 0:1]

    wh = torch.cat([bboxs_w, bboxs_h], dim=1)

    ct = torch.cat(((annos[:, 0:1] + annos[:, 2:3]) / 2., (annos[:, 1:2] + annos[:, 3:4]) / 2.), dim=1)
    ct_int = ct.floor()
    offset = ct - ct_int
    reg_mask = ((bboxs_h > 0) * (bboxs_w > 0))
    ind = ct_int[:, 1:2] * (w // 4) + ct_int[:, 0:1]
    radius = gaussian_radius((bboxs_h.ceil(), bboxs_w.ceil()))
    radius = radius.floor().clamp(min=0)
    for k, cls in enumerate(cls_idx):
        draw_umich_gaussian(hm[cls.long().item()], ct_int[k], radius[k])
    return data[0], data[1], hm, wh, ind, offset, reg_mask


def draw_umich_gaussian_with_ellipse(heatmap, center, radius, k=1, bbox_w=1, bbox_h=1):
    # TODO: for tensor.
    diameter_w = int(bbox_w / 2)
    diameter_h = int(bbox_h / 2)
    if diameter_h == 0:
        diameter_h = 1
    elif diameter_h % 2 == 0:
        diameter_h = diameter_h + 1
    if diameter_w == 0:
        diameter_w = 1
    elif diameter_w % 2 == 0:
        diameter_w = diameter_w + 1
    gaussian = gaussian2d((diameter_h, diameter_w), sigma=(diameter_w + diameter_h) / 12)

    x, y = int(center[0]), int(center[1])

    # masked_heatmap = np.asarray(heatmap[y - top:y + bottom, x - left:x + right])
    masked_heatmap = heatmap[y - (diameter_h // 2):y + (diameter_h // 2) + 1,
                     x - (diameter_w // 2):x + (diameter_w // 2) + 1]
    masked_gaussian = gaussian
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:  # TODO debug
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap


def mask_ignore(data, mean=(0.485, 0.456, 0.406), ignore_cls=0):
    """
    Mask the ignore region with mean value, and remove all the bbox annotations of ignore region.
    :param data: (Tensor, Tensor) image and annotation
    :param mean: Mean value.
    :param ignore_cls: Ignore class idx.
    :return: (Tensor, Tensor) transformed image and annotation
    """
    mean = torch.tensor(mean).unsqueeze(1).unsqueeze(1)
    img = data[0]
    roadmap = data[2]
    ign_idx = data[1][:, 5] == ignore_cls

    ign_bboxes = data[1][ign_idx, :4]

    for ign_bbox in ign_bboxes:
        x, y, w, h = ign_bbox[:4]
        img[:, int(y):int(y + h), int(x):int(x + w)] = mean
        if roadmap is not None:
            roadmap[int(y):int(y + h), int(x):int(x + w)] = 0

    anno = data[1][1 - ign_idx, :]

    return img, anno, roadmap


DIRECTION = torch.tensor([[-1, -1], [-1, 0], [-1, 1],
                          [0, -1], [0, 0], [0, 1],
                          [1, -1], [1, 0], [1, 1]], dtype=torch.float)


def to_twostage_heatmap(data, scale_factor=4):
    """
    Transform annotations to heatmap.
    :param data: (img, annos), tensor
    :param scale_factor:
    :param bias_factor:
    :return:
    """
    img = data[0]
    annos = data[1].clone()

    h, w = img.size(1), img.size(2)

    hm = torch.zeros(1, h // scale_factor, w // scale_factor)

    annos[:, 2] += annos[:, 0]
    annos[:, 3] += annos[:, 1]
    annos[:, :4] = annos[:, :4] / scale_factor
    cls_idx = torch.zeros_like(annos[:, 5])
    bboxs_h, bboxs_w = annos[:, 3:4] - annos[:, 1:2], annos[:, 2:3] - annos[:, 0:1]

    wh = torch.cat([bboxs_w, bboxs_h], dim=1)

    ct = torch.cat(((annos[:, 0:1] + annos[:, 2:3]) / 2., (annos[:, 1:2] + annos[:, 3:4]) / 2.), dim=1)
    ct_int = ct.floor()
    offset = ct - ct_int
    reg_mask = ((bboxs_h > 0) * (bboxs_w > 0))
    ind = ct_int[:, 1:2] * (w // scale_factor) + ct_int[:, 0:1]
    radius = gaussian_radius((bboxs_h.ceil(), bboxs_w.ceil()))
    radius = radius.floor().clamp(min=0)
    for k, cls in enumerate(cls_idx):
        draw_umich_gaussian(hm[cls.long().item()], ct_int[k], radius[k])
    return data[0], data[1], hm, wh, ind, offset, reg_mask


def fill_duck(data, cls_list, factor):
    try:
        img, annos, roadmap = data

        # I. Get valid area.
        valid_idx = roadmap.view(-1)
        idx = torch.nonzero(valid_idx).view(-1)
        if idx.size(0) == 0:
            return img, annos
        xs = idx % roadmap.size(1)
        ys = idx // roadmap.size(1)
        coor = torch.stack((xs, ys), dim=1)

        annos_cls = annos[:, 5]

        # II Calculate scale factor for depth.
        people_flag = annos_cls == 1
        people_bbox = annos[people_flag, :4]
        if people_bbox.size(0) != 0:
            people_diag = people_bbox[:, 2:4].pow(2).sum(dim=1).sqrt()
            topk = min(3, people_diag.size(0))
            max_diag, max_idx = torch.topk(people_diag, k=topk)
            min_diag, min_idx = torch.topk(people_diag, k=1, largest=False)
            y_diff = people_bbox[max_idx, 1] - people_bbox[min_idx, 1]
            scale_factor = ((max_diag - min_diag) / (y_diff.abs() + 1e-5)).mean()
        else:
            scale_factor = 1

        # III. For relation class.

        people_flag = annos_cls == 2
        people_select_annos = annos[people_flag, :]

        relation_flag = torch.zeros_like(annos_cls).byte()

        if people_select_annos.size(0) != 0:
            iou = bbox_iou(people_select_annos[:, :4], annos[:, :4], x1y1x2y2=False)
            if iou.size(1) > 2:
                max_v, max_i = torch.topk(iou, dim=1, k=2)
                flag = max_v[:, 1] > 0
                max_i = max_i[flag, :]
                people_idx = max_i[:, 0]
                vechile_idx = max_i[:, 1]

                relation_flag[people_idx] = 1
                relation_flag[vechile_idx] = 1

        # IV. Calculate aug N.
        cls = cls_list.repeat(annos.size(0), 1)
        normal_flag = (cls == annos_cls.unsqueeze(1).repeat(1, cls.size(1)).long()).sum(dim=1) > 0
        normal_flag = normal_flag * (1 - relation_flag)

        total_n = max(int(factor * valid_idx.sum()), 5)
        relation_n = relation_flag.float().sum() / 2
        normal_n = normal_flag.float().sum()
        if relation_n + normal_n == 0:
            return img, annos
        r_n = int(relation_n / (relation_n + normal_n) * total_n)
        n_n = total_n - r_n

        # V. Fill image
        paste_idx = torch.randint(low=0, high=coor.size(0), size=(total_n,))
        paste_coors = coor[paste_idx]

        new_annos = []
        # 1. Sample normal object.
        if n_n != 0:
            normal_annos = annos[normal_flag, :]
            sample_idx = torch.randint(low=0, high=normal_annos.size(0), size=(n_n,))
            sample_annos = normal_annos[sample_idx]
            for i, anno in enumerate(sample_annos):
                paste_coor = paste_coors[i].float()

                # Apply depth scale.
                anno_ct_y = anno[1] + anno[3] / 2
                diff = (anno_ct_y - paste_coor[1]).abs() * scale_factor
                anno_diag = (anno[2].pow(2) + anno[3].pow(2)).sqrt()
                if anno_ct_y > paste_coor[1]:
                    # Do reduce.
                    factor = 1 - diff / anno_diag
                else:
                    factor = 1 + diff / anno_diag
                cropped_obj = img[:, int(anno[1]):int(anno[1]+anno[3]), int(anno[0]):int(anno[0]+anno[2])]
                factor = factor.clamp(min=0.5, max=2)
                cropped_obj = F.interpolate(
                    cropped_obj.unsqueeze(0),
                    scale_factor=float(factor),
                    mode='bilinear',
                    align_corners=True
                )[0]
                obj_h, obj_w = cropped_obj.size()[-2:]
                paste_coor[0] -= obj_w / 2
                paste_coor[1] -= obj_h / 2
                paste_coor[0] = paste_coor[0].clamp(min=1, max=img.size(2)-obj_w - 1)
                paste_coor[1] = paste_coor[1].clamp(min=1, max=img.size(1)-obj_h - 1)
                img[:, int(paste_coor[1]):int(paste_coor[1]+obj_h),
                int(paste_coor[0]):int(paste_coor[0]+obj_w)] = cropped_obj
                new_annos.append(torch.tensor([[int(paste_coor[0]), int(paste_coor[1]), int(obj_w), int(obj_h), anno[4], anno[5], anno[6], anno[7]]]))

        # 2. Sample Relation Object.
        if r_n != 0:
            people_annos = annos[people_idx, :]
            vechile_annos = annos[vechile_idx, :]

            sample_idx = torch.randint(low=0, high=people_annos.size(0), size=(r_n,))
            sample_people_annos = people_annos[sample_idx]
            sample_vechile_annos = vechile_annos[sample_idx]
            sample_people_annos[:, 2:4] += sample_people_annos[:, 0:2]
            sample_vechile_annos[:, 2:4] += sample_vechile_annos[:, 0:2]

            for i in range(r_n):
                paste_coor = paste_coors[i + n_n].float()

                people_anno = sample_people_annos[i]
                vechile_anno = sample_vechile_annos[i]

                min_x = int(min(people_anno[0], vechile_anno[0]))
                min_y = int(min(people_anno[1], vechile_anno[1]))
                max_x = int(max(people_anno[2], vechile_anno[2]))
                max_y = int(max(people_anno[3], vechile_anno[3]))

                # Apply depth scale.
                anno_ct_y = (min_y + max_y) / 2
                diff = (anno_ct_y - paste_coor[1]).abs() * scale_factor
                anno_diag = math.sqrt((max_x-min_x)**2 + (max_y-min_y)**2)
                if anno_ct_y > paste_coor[1]:
                    # Do reduce.
                    factor = 1 - diff / anno_diag
                else:
                    factor = 1 + diff / anno_diag
                cropped_obj = img[:, min_y:max_y, min_x:max_x]
                factor = factor.clamp(min=0.5, max=2)
                cropped_obj = F.interpolate(
                    cropped_obj.unsqueeze(0),
                    scale_factor=float(factor),
                    mode='bilinear',
                    align_corners=True
                )[0]

                obj_h, obj_w = cropped_obj.size()[-2:]
                paste_coor[0] -= obj_w / 2
                paste_coor[1] -= obj_h / 2
                paste_coor[0] = paste_coor[0].clamp(min=1, max=img.size(2)-obj_w - 1)
                paste_coor[1] = paste_coor[1].clamp(min=1, max=img.size(1)-obj_h - 1)
                img[:, int(paste_coor[1]):int(paste_coor[1]+obj_h),
                int(paste_coor[0]):int(paste_coor[0]+obj_w)] = cropped_obj
                x_bias = min_x - paste_coor[0]
                y_bias = min_y - paste_coor[1]
                new_people = people_anno
                new_people[2:4] -= new_people[0:2]
                new_people[2:4] *= factor
                new_people[0] -= x_bias
                new_people[1] -= y_bias

                new_vechile = vechile_anno
                new_vechile[2:4] -= new_vechile[0:2]
                new_vechile[2:4] *= factor
                new_vechile[0] -= x_bias
                new_vechile[1] -= y_bias

                new_annos.append(new_people.unsqueeze(0).floor())
                new_annos.append(new_vechile.unsqueeze(0).floor())
        new_annos = torch.cat(new_annos)
        annos = torch.cat((annos, new_annos))

        return img, annos
    except:
        return data[0], data[1]

