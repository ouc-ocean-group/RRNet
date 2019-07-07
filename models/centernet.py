import torch
import torch.nn as nn
from utils.model_tools import get_backbone
from detectors.centernet_detector import CenterNetDetector
from detectors.centernet_detector import CenterNetWHDetector


class CenterNet(nn.Module):
    def __init__(self, cfg):
        super(CenterNet, self).__init__()
        self.num_stacks = cfg.Model.num_stacks
        self.num_classes = cfg.num_classes
        self.backbone = get_backbone(cfg.Model.backbone, num_stacks=self.num_stacks)
        self.hm = CenterNetDetector(planes=cfg.num_classes, num_stacks=self.num_stacks, hm=True)
        self.wh = CenterNetWHDetector(planes=1, num_stacks=self.num_stacks)
        self.reg = CenterNetDetector(planes=2, num_stacks=self.num_stacks)

    def forward(self, input):
        pre_feats = self.backbone(input)
        hms = []
        whs = []
        regs = []
        for i in range(self.num_stacks):
            feat = pre_feats[i]
            feat = torch.relu(feat)
            hm = self.hm(feat, i)
            wh = self.wh(feat, i)
            reg = self.reg(feat, i)
            hms.append(hm)
            whs.append(wh)
            regs.append(reg)

        return hms, whs, regs