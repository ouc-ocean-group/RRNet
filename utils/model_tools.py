import torch
from backbones.resnet import resnet50,resnet101
from fpns.demofpn import DemoFPN
from detectors.demo_detector import DemoDetector



def get_backbone(bb_name,pretrained=False):
    if bb_name=='resnet50':
        return resnet50(pretrained=pretrained)
    elif bb_name=='resnet101':
        return resnet101(pretrained=pretrained)
    else:
        return resnet50(pretrained=pretrained)

def get_fpn(fp_name):
    if fp_name=='demofpn':
        return DemoFPN()
    else:
        return DemoFPN()

def get_detector(det_name,planes):
    if det_name=='demodetector':
        return DemoDetector(planes)
    else:
        return DemoDetector(planes)
