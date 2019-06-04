from backbones.resnet import resnet10, resnet50, resnet101
from backbones.hourglass import hourglass_net
from modules.fpn import FPN
from modules.nas.fpn import NASSuperFPN, NASSearchedFPN
from detectors.centernet_detector import CenterNetDetector
from detectors.centernet_detector import CenterNet_WH_Detector
from detectors.retinanet_detector import RetinaNetDetector
# from backbones.trident import trident_res50v2, trident_res50v2_deform, trident_res101v2, trident_res101v2_deform


def get_backbone(backbone, pretrained=False, num_stacks=2):
    if backbone == 'resnet10':
        return resnet10(pretrained=pretrained)
    elif backbone == 'resnet50':
        return resnet50(pretrained=pretrained)
    elif backbone == 'resnet101':
        return resnet101(pretrained=pretrained)
    # elif backbone == 'trires50':
    #     return trident_res50v2()
    # elif backbone == 'trires50deform':
    #     return trident_res50v2_deform()
    # elif backbone == 'trires101':
    #     return trident_res101v2()
    # elif backbone == 'trires101deform':
    #     return trident_res101v2_deform()
    elif backbone == 'hourglass':
        return hourglass_net(num_stacks=num_stacks)
    else:
        return resnet50(pretrained=pretrained)


def get_fpn(fp_name, cfg=None):
    if fp_name == 'fpn':
        return FPN()
    elif fp_name == 'nasfpn':
        return NASSuperFPN(cfg)
    elif fp_name == 'searchedfpn':
        return NASSearchedFPN(cfg)
    else:
        return FPN()


def get_detector(det_name, planes, num_stacks=2, hm=False):
    if det_name == 'retinanet_detector':
        return RetinaNetDetector(planes)
    elif det_name == 'centernet_detector':
        return CenterNetDetector(planes, hm=hm, num_stacks=num_stacks)
    elif det_name == 'centernet_WH_detector':
        return CenterNet_WH_Detector(planes, hm=hm, num_stacks=num_stacks)
    else:
        return RetinaNetDetector(planes)
