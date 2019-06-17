import torch.nn as nn
from .functional import focal_loss, focal_loss_for_hm


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, ignore=-1):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.ignore = ignore

    def forward(self, input, target):
        return focal_loss(input, target, self.gamma)


class FocalLossHM(nn.Module):
    def __init__(self):
        super(FocalLossHM, self).__init__()

    def forward(self, out, target):
        return focal_loss_for_hm(out, target)