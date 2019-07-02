import torch
import torch.nn as nn
from utils.model_tools import get_backbone
from detectors.centernet_detector import CenterNetDetector
from detectors.centernet_detector import CenterNet_WH_Detector
from modules.feat_projector import FeatProjector


class CenterNet(nn.Module):
    def __init__(self, cfg):
        super(CenterNet, self).__init__()
        self.num_stacks = cfg.Model.num_stacks
        self.num_classes = cfg.num_classes
        self.backbone = get_backbone(cfg.Model.backbone, num_stacks=self.num_stacks)
        self.hm = CenterNetDetector(planes=cfg.num_classes, num_stacks=self.num_stacks, hm=True)
        self.wh = CenterNet_WH_Detector(planes=2, num_stacks=self.num_stacks)
        self.reg = CenterNetDetector(planes=2, num_stacks=self.num_stacks)
        self.feat_projector = FeatProjector(256, 4)

    def forward(self, input):
        pre_feats = self.backbone(input)
        hms = []
        whs = []
        regs = []
        ori_feats = []
        projected_feats = []
        for i in range(self.num_stacks):
            ori_feats.append(pre_feats[i])
            feats = torch.relu(pre_feats[i])
            projected_feat = self.feat_projector(feats)
            projected_feats.append(projected_feat)

            hm = self.hm(feats, i)
            wh = self.wh(feats, i)
            reg = self.reg(feats, i)
            hms.append(hm)
            whs.append(wh)
            regs.append(reg)

        return hms, whs, regs, ori_feats, projected_feats

