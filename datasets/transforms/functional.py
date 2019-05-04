import torch
from PIL import Image
import random
import PIL.ImageEnhance as ImageEnhance
import torchvision.transforms.functional as torchtransform


def flip_img(data):
    """
    Flip PIL Image.
    :param data: PIL Image.
    :return: Flipped Image.
    """
    return data.transpose(Image.FLIP_LEFT_RIGHT)


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
    for d in data:
        split_d = [int(x) for x in d.strip().split(',')]
        annos.append(split_d)
    annos_tensor = torch.tensor(annos)
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


def crop_tensor(data, coordinate):
    """
    Crop the torch tensor.
    :param data: tensor.
    :param coordinate: crop coordinate.
    :return: cropped tensor.
    """
    return data[:, coordinate[0]:coordinate[1], coordinate[2]:coordinate[3]]


def crop_annos(data, coordinate):
    """
    Crop the annotations tensor.
    :param data: annotations tensor.
    :param coordinate: crop coordinate.
    :return: cropped annotations tensor.  438,249,11,15,1,2,0,1
    """
    # Here we need to use iou to get the valid bounding box in cropped area.
    # miou = MIOU(data, coordinate)
    return cropped_annotations


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
