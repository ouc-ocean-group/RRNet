from backbones.hourglass import hourglass_net
from backbones.resnet import resnet50, resnet101
from detectors.centernet_detector import CenterNetDetector, CenterNet_HM_Detector
from detectors.retinanet_detector import RetinaNetDetector
from modules.fpn import FPN


def get_backbone(backbone, pretrained=False, num_stacks=2):
    if backbone == 'resnet50':
        return resnet50(pretrained=pretrained)
    elif backbone == 'resnet101':
        return resnet101(pretrained=pretrained)
    elif backbone == 'hourglass':
        return hourglass_net(num_stacks=num_stacks)
    else:
        return resnet50(pretrained=pretrained)


def get_fpn(fp_name):
    if fp_name == 'fpn':
        return FPN()
    else:
        return FPN()


def get_detector(det_name, planes, num_stacks=2, hm=False):
    if det_name == 'retinanet_detector':
        return RetinaNetDetector(planes)
    elif det_name == 'centernet_detector':
        return CenterNetDetector(planes, hm=hm, num_stacks=num_stacks)
    elif det_name == 'centernet_hm_detector':
        return CenterNet_HM_Detector(planes, hm=hm, num_stacks=num_stacks)
    else:
        return RetinaNetDetector(planes)
