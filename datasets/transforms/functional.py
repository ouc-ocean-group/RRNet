import torch
import random
import PIL.ImageEnhance as ImageEnhance
import torchvision.transforms.functional as torchtransform
from utils.metrics.metrics import overlap


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
    data[:, 1] = w - data[:, 1] - data[:, 3]
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
    if isinstance(data, list):
        for d in data:
            split_d = [int(x) for x in d.strip().split(',')]
            annos.append(split_d)
    else:
        annos = data
    annos_tensor = torch.tensor(annos)
    annos_tensor[:, [0, 1, 2, 3]] = annos_tensor[:, [1, 0, 3, 2]]
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
    return data[:, crop_coor[0]:crop_coor[2], crop_coor[1]:crop_coor[3]]


def crop_annos(data, crop_coor, h, w):
    """
    Crop the annotations tensor.
    :param data: annotations tensor: yxhw
    :param crop_coor: crop coordinate: yxyx
    :return: cropped annotations tensor.
    """
    # Here we need to use iou to get the valid bounding box in cropped area.
    crop_coor_tensor = torch.tensor(crop_coor).unsqueeze(0).repeat(data.size(0), 1)
    data[:, 2:4] = data[:, :2] + data[:, 2:4]
    olap = overlap(data[:, :4], crop_coor_tensor)
    keep_flag = olap > 0.5
    keep_data = data[keep_flag, :]
    if keep_data.size(0) == 0:
        return keep_data
    keep_data[:, :4] -= crop_coor_tensor[keep_flag, :2].repeat(1, 2)
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
