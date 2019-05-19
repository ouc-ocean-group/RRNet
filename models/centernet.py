import torch
import torch.nn as nn
from utils.model_tools import get_backbone, get_fpn, get_detector

class CenterNet(nn.Module):
    def __init(self,cfg):
        super(self,CenterNet).__init__()
        self.num_stacks=cfg.Model.num_stacks
        self.num_classes = cfg.num_classes
        self.backbone=get_backbone(cfg.Model.backbone, num_stacks=self.num_stacks)
        self.hm= get_detector(cfg.Model.hm_detector, cfg.num_classes , num_stacks=self.num_stacks, hm=True)
        self.wh= get_detector(cfg.Model.wh_detector, 2, num_stacks=self.num_stacks)
        self.reg= get_detector(cfg.Model.reg_detector, 2, num_stacks=self.num_stacks)



    def forward(self, input):
        pre_feats = self.backbone(input)
        outs=[]
        for i in range(self.num_stacks):
            out={}
            out['hm']=self.hm(pre_feats[i],i)
            out['wh']=self.wh(pre_feats[i],i)
            out['reg']=self.reg(pre_feats[i],i)
            outs.append(out)
        return outs


def build_net(cfg):
    CenterNet(cfg)