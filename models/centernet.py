import torch
import torch.nn as nn
from utils.model_tools import get_backbone, get_fpn, get_detector
from configs.centernet_config import Config


class CenterNet(nn.Module):
    def __init__(self, cfg):
        super(CenterNet, self).__init__()
        self.num_stacks = cfg.Model.num_stacks
        self.num_classes = cfg.num_classes
        self.backbone = get_backbone(cfg.Model.backbone, num_stacks=self.num_stacks)
        self.hm = get_detector(cfg.Model.hm_detector, cfg.num_classes, num_stacks=self.num_stacks, hm=True)
        self.wh = get_detector(cfg.Model.wh_detector, 2, num_stacks=self.num_stacks)
        self.reg = get_detector(cfg.Model.reg_detector, 2, num_stacks=self.num_stacks)

    def forward(self, input):
        pre_feats = self.backbone(input)
        hms = []
        whs = []
        regs = []
        for i in range(self.num_stacks):
            hm = self.hm(pre_feats[i], i)
            wh = self.wh(pre_feats[i], i)
            reg = self.reg(pre_feats[i], i)
            hms.append(hm)
            whs.append(wh)
            regs.append(reg)
        return hms, whs, regs


def build_net(cfg):
    return CenterNet(cfg)

if __name__ == '__main__':
    cfg = Config
    a = torch.ones(1, 3, 256, 256)
    model = CenterNet(cfg).cuda(cfg.Distributed.gpu_id)
    hms, whs, regs= model(a)
    print(hms[1])
