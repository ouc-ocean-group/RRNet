import torch
import torch.nn as nn
from .functional import focal_loss

class FocalLoss(nn.Module):
    def __init__(self,alpha=0.25,gamma=2,classnum=13):
        super(FocalLoss,self).__init__()
        self.gamma=gamma
        self.alpha=alpha
        self.classnum=classnum


    def forward(self, input ,target):
        return focal_loss(input,target,self.alpha,self.gamma,self.classnum)