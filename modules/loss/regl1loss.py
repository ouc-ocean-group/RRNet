import torch.nn as nn
import torch.nn.functional as F


class RegL1Loss(nn.Module):
    def __init__(self):
        super(RegL1Loss, self).__init__()

    def forward(self, output, mask, ind, target):
        pred = output.permute(0, 2, 3, 1).contiguous()
        pred = pred.view(pred.size(0), -1, pred.size(3))
        ind = ind.long().expand(ind.size(0), ind.size(1), pred.size(2))
        pred = pred.gather(1, ind)
        mask = mask.expand_as(pred).float()
        loss = F.l1_loss(pred * mask, target * mask, reduction='sum')
        loss = loss / (mask.sum() + 1e-4)
        return loss
