import torch.nn as nn
from .functional import focal_loss


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, class_num=13, ignore=-1):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.class_num = class_num
        self.ignore = ignore

    def forward(self, input, target):
        return focal_loss(input, target, self.alpha, self.gamma, self.class_num, self.ignore)
