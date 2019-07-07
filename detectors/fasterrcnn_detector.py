import torch.nn as nn
from backbones.resnet import Bottleneck
import torch.nn.functional as F


class FasterRCNNDetector(nn.Module):
    def __init__(self):
        super(FasterRCNNDetector, self).__init__()

        self.top_layer = Bottleneck(inplanes=256, planes=64)
        self.regressor = nn.Conv2d(256, 4, kernel_size=1)

    def forward(self, feat):
        feat = self.top_layer(feat)
        feat = F.adaptive_avg_pool2d(feat, 1)
        reg = self.regressor(feat)
        reg = reg.view(reg.size(0), reg.size(1))
        return reg
