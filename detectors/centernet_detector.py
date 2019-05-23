import torch
import torch.nn as nn
import torch.nn.functional as F


class CenterNetDetector(nn.Module):
    def __init__(self, planes, hm=True, num_stacks=2):
        super( CenterNetDetector,self).__init__()
        self.hm = hm
        self.num_stacks = num_stacks
        self.detect_layer = nn.ModuleList([nn.Sequential(
            BasicCov(3, 256, 256, with_bn=False),
            nn.Conv2d(256, planes, (1, 1))
        ) for _ in range(self.num_stacks)
        ])
        if self.hm:
            for heat in self.detect_layer:
                heat[-1].bias.data.fill_(-2.19)

    def forward(self, input, index):
        output = self.detect_layer[index](input)
        output = F.interpolate(output, scale_factor=4, mode='bilinear', align_corners=True)
        return output


class BasicCov(nn.Module):
    def __init__(self, k, inp_dim, out_dim, stride=1, with_bn=True):
        super(BasicCov, self).__init__()

        pad = (k - 1) // 2
        self.conv = nn.Conv2d(inp_dim, out_dim, (k, k), padding=(pad, pad), stride=(stride, stride), bias=not with_bn)
        self.bn = nn.BatchNorm2d(out_dim) if with_bn else nn.Sequential()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        conv = self.conv(x)
        bn = self.bn(conv)
        relu = self.relu(bn)
        return relu
