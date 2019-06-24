import torch.nn as nn
from utils.model_tools import get_backbone
from detectors.centernet_detector import CenterNetDetector
from detectors.centernet_detector import CenterNet_WH_Detector


class Box9Net(nn.Module):
    def __init__(self, cfg):
        super(Box9Net, self).__init__()
        self.num_stacks = 1
        self.num_classes = cfg.num_classes
        self.backbone = get_backbone(cfg.Model.backbone, num_stacks=self.num_stacks)
        self.hm = CenterNetDetector(planes=9, num_stacks=self.num_stacks, hm=True)
        self.wh = CenterNet_WH_Detector(planes=18, num_stacks=self.num_stacks)
        self.offset_reg = CenterNetDetector(planes=2, num_stacks=self.num_stacks)

        self.head_detector = FasterRCNNDetecotr()

    def forward(self, input):
        pre_feats = self.backbone(input)
        hm = self.hm(pre_feats, 0)
        wh = self.wh(pre_feats, 0)
        offset_reg = self.offset_reg(pre_feats, 0)

        cls, bbox_reg = self.head_detector(pre_feats, hm, wh, offset_reg)

        return hm, wh, offset_reg, cls, bbox_reg

