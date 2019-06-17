import torch
import random
import PIL.ImageEnhance as ImageEnhance
import torchvision.transforms.functional as torchtransform
from utils.metrics.metrics import bbox_iou
import numpy as np


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
    _, olap = bbox_iou(data[:, :4], crop_coor_tensor, overlap=True)
    keep_flag = (olap > 0.5).view(-1)
    keep_data = data[keep_flag, :]
    if keep_data.size(0) == 0:
        return keep_data
    keep_data[:, :4] -= crop_coor_tensor[:, :2].repeat(1, 2)
    keep_data[keep_data[:, 0] < 0, 0] = 0
    keep_data[keep_data[:, 1] < 0, 1] = 0
    keep_data[keep_data[:, 2] > w, 2] = w
    keep_data[keep_data[:, 3] > h, 3] = h

    keep_data[:, 2] = keep_data[:, 2] - keep_data[:, 0]
    keep_data[:, 3] = keep_data[:, 3] - keep_data[:, 1]
    return keep_data


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
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / 2
    return min(r1, r2, r3)


def gaussian_radius_tensor(det_size, min_overlap=0.7):
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
    reg_mask = ((bboxs_h > 0) * (bboxs_w > 0)).unsqueeze(1)
    ind = ct_int[:, 1] * (w // 4) + ct_int[0]

    radius = gaussian_radius_tensor((bboxs_h.ceil(), bboxs_w.ceil()))
    radius = radius.floor().clamp(min=0)
    for k, cls in enumerate(cls_idx):
        draw_umich_gaussian(hm[cls.long().item()], ct_int[k], radius[k])

    return data[0], hm, wh, ind, offset, reg_mask


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
    gaussian = gaussian2D((diameter_h, diameter_w), sigma=(diameter_w + diameter_h) / 12)

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
    ign_idx = data[1][:, 5] == ignore_cls

    ign_bboxes = data[1][ign_idx, :4]

    for ign_bbox in ign_bboxes:
        x, y, w, h = ign_bbox[:4]
        img[:, int(y):int(y + h), int(x):int(x + w)] = mean

    anno = data[1][1 - ign_idx, :]

    return img, anno
