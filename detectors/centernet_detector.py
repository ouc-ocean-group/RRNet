import torch
import torch.nn as nn
import torch.nn.functional as F


class CenterNetDetector(nn.Module):
    def __init__(self, planes, hm=True, num_stacks=2):
        super(CenterNetDetector, self).__init__()
        self.hm = hm
        self.num_stacks = num_stacks
        self.detect_layer = nn.ModuleList([nn.Sequential(
            BasicCov(3, 256, 256, with_bn=False),
            # BasicCov(3, 48, 256, with_bn=False),
            nn.Conv2d(256, planes, (1, 1))
        ) for _ in range(self.num_stacks)
        ])
        if self.hm:
            for heat in self.detect_layer:
                heat[-1].bias.data.fill_(-2.19)

    def forward(self, input, index):
        output = self.detect_layer[index](input)
        return output


class CenterNet_WH_Detector(nn.Module):
    def __init__(self, planes, hm=True, num_stacks=2):
        super(CenterNet_WH_Detector, self).__init__()
        self.hm = hm
        self.num_stacks = num_stacks
        self.stackone_conv = BasicCov(3, 256, 256, with_bn=False)
        # self.stackone_conv = BasicCov(3, 48, 256, with_bn=False)
        self.stackone_H = HCov(17, 256, 1, with_bn=False)
        self.stackone_W = WCov(17, 256, 1, with_bn=False)
        self.stacktwo_conv = BasicCov(3, 256, 256, with_bn=False)
        self.stacktwo_H = HCov(17, 256, 1, with_bn=False)
        self.stacktwo_W = WCov(17, 256, 1, with_bn=False)

    def forward(self, input, index):
        if index == 0:
            x = self.stackone_conv(input)
            H = self.stackone_H(x)
            W = self.stackone_W(x)

            output = torch.cat((W, H), dim=1)
        elif index == 1:
            x = self.stacktwo_conv(input)
            H = self.stacktwo_H(x)
            W = self.stacktwo_W(x)
            output = torch.cat((W, H), dim=1)
        else:
            print('W, H error!, index != 0, 1')

        # output = F.interpolate(output, scale_factor=2, mode='bilinear', align_corners=True)
        return output


class HCov(nn.Module):
    def __init__(self, k, inp_dim, out_dim, stride=1, with_bn=True):
        super(HCov, self).__init__()
        pad = (k - 1) // 2
        self.conv = nn.Conv2d(inp_dim, out_dim, (k, 1), padding=(pad, 0), stride=(stride, stride), bias=not with_bn)

    def forward(self, x):
        conv = self.conv(x)
        return conv


class WCov(nn.Module):
    def __init__(self, k, inp_dim, out_dim, stride=1, with_bn=True):
        super(WCov, self).__init__()
        pad = (k - 1) // 2
        self.conv = nn.Conv2d(inp_dim, out_dim, (1, k), padding=(0, pad), stride=(stride, stride), bias=not with_bn)

    def forward(self, x):
        conv = self.conv(x)
        return conv


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


class CenterNet_HM_Detector(nn.Module):
    def __init__(self, planes, hm=True, num_stacks=2):
        super(CenterNet_HM_Detector, self).__init__()
        self.hm = hm
        self.num_stacks = num_stacks
        self.detect_layer = nn.ModuleList([nn.Sequential(
            BasicCov(3, 256, 64, with_bn=False),
            PAM_Module(64),
            nn.Conv2d(64, planes, (1, 1))
        ) for _ in range(self.num_stacks)
        ])
        if self.hm:
            for heat in self.detect_layer:
                heat[-1].bias.data.fill_(-2.19)

    def forward(self, input, index):
        output = self.detect_layer[index](input)
        # output = F.interpolate(output, scale_factor=2, mode='bilinear', align_corners=True)
        return output


class PAM_Module(nn.Module):
    """ Position attention module"""

    # Ref from SAGAN
    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x
        return out
