import torch
import torch.nn as nn
from utils.model_tools import get_backbone, get_fpn, get_detector


class NASRetinaNet(nn.Module):
    def __init__(self, cfg):
        super(NASRetinaNet, self).__init__()
        self.num_anchors = cfg.Model.num_anchors
        self.num_classes = cfg.num_classes
        self.backbone = get_backbone(cfg.Model.backbone, cfg.Train.pretrained)
        self.fpn = get_fpn(cfg.Model.fpn)
        self.cls = get_detector(cfg.Model.cls_detector, self.num_anchors * self.num_classes)
        self.loc = get_detector(cfg.Model.loc_detector, self.num_anchors * 4)

    def forward(self, input, p_seq, l_seq):
        loc_pres = []
        cls_pres = []
        l1, l2, l3, l4 = self.backbone(input)
        fms = self.fpn((l2, l3, l4), p_seq, l_seq)
        for fm in fms:
            loc_pre = self.loc(fm)
            cls_pre = self.cls(fm)
            # [N, anchors*4,H,W] -> [N,H,W, anchors*4] -> [N,H*W*anchors, 4]
            loc_pre = loc_pre.permute(0, 2, 3, 1).contiguous().view(input.size(0), -1, 4)
            # [N,anchors*classes,H,W] -> [N,H,W,anchors*classes] -> [N,H*W*anchors,classes]
            cls_pre = cls_pre.permute(0, 2, 3, 1).contiguous().view(input.size(0), -1, self.num_classes)
            loc_pres.append(loc_pre)
            cls_pres.append(cls_pre)
        loc_pre = torch.cat(loc_pres, 1)
        cls_pre = torch.cat(cls_pres, 1)
        return loc_pre, cls_pre


def build_net(cfg):
    return NASRetinaNet(cfg)
