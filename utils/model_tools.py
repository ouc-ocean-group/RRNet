from backbones.resnet import resnet50, resnet101
from modules.fpn import FPN
from detectors.retinanet_detector import RetinaNetDetector
from backbones.trident import trident_res50v2, trident_res50v2_deform, trident_res101v2, trident_res101v2_deform


def get_backbone(backbone, pretrained=False):
    if backbone == 'resnet50':
        return resnet50(pretrained=pretrained)
    elif backbone == 'resnet101':
        return resnet101(pretrained=pretrained)
    elif backbone == 'trires50':
        return trident_res50v2()
    elif backbone == 'trires50deform':
        return trident_res50v2_deform()
    elif backbone == 'trires101':
        return trident_res101v2()
    elif backbone == 'trires101deform':
        return trident_res101v2_deform()
    else:
        return resnet50(pretrained=pretrained)


def get_fpn(fp_name):
    if fp_name == 'fpn':
        return FPN()
    else:
        return FPN()


def get_detector(det_name, planes):
    if det_name == 'retinanet_detector':
        return RetinaNetDetector(planes)
    else:
        return RetinaNetDetector(planes)
