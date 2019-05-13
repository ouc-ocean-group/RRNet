import torch
import torch.nn as nn

class FocalLoss(nn.Module):
    def __init__(self,alpha=None,gamma=2,classnum=20):
        super(FocalLoss,self).__init__()
        if alpha:
            self.alpha=alpha
        else:
            self.alpha=torch.ones([classnum])
        self.gamma=gamma


    def forward(self, input ,target):

        return input