import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SharedConv(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size, stride, dilation):
        super(SharedConv, self).__init__()
        """
            SharedConv:
                input: [branch1, branch2, branch3]
                    dilation: [1,2,3] (padding=[1,2,3]) if kernel_size == 3 else [1,1,1] (padding=[0,0,0])
                output: [out1, out2, out3]
        """
        assert kernel_size in [1, 3]
        self.weight = nn.Parameter(torch.Tensor(dim_out, dim_in, kernel_size, kernel_size))
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation

    def forward(self, x):
        out = []
        out.append(F.conv2d(x[0], weight=self.weight, stride=self.stride,
                            dilation=self.dilation[0], padding=self.dilation[0] if self.kernel_size == 3 else 0))
        out.append(F.conv2d(x[1], weight=self.weight, stride=self.stride,
                            dilation=self.dilation[1], padding=self.dilation[1] if self.kernel_size == 3 else 0))
        out.append(F.conv2d(x[2], weight=self.weight, stride=self.stride,
                            dilation=self.dilation[2], padding=self.dilation[2] if self.kernel_size == 3 else 0))
        return out


class ResTridentUnit(nn.Module):
    def __init__(self, dim_in, dim_out, stride=1):
        """
            Contains three Convs[1x1, 3x3, 1x1]
        :param dim_in:
        :param dim_out:
        :param stride:
        """
        super(ResTridentUnit, self).__init__()
        dim_mid = dim_out // 4
        self.bn1 = nn.ModuleList([
            nn.BatchNorm2d(dim_in),
            nn.BatchNorm2d(dim_in),
            nn.BatchNorm2d(dim_in)
        ])
        self.relu = nn.ModuleList([
            nn.ReLU(inplace=True),
            nn.ReLU(inplace=True),
            nn.ReLU(inplace=True)
        ])
        self.conv1 = SharedConv(dim_in=dim_in, dim_out=dim_mid, kernel_size=1, stride=1, dilation=[1, 1, 1])

        self.bn2 = nn.ModuleList([
            nn.BatchNorm2d(dim_mid),
            nn.BatchNorm2d(dim_mid),
            nn.BatchNorm2d(dim_mid)
        ])
        self.conv2 = SharedConv(dim_in=dim_mid, dim_out=dim_mid, kernel_size=3, stride=stride, dilation=[1, 2, 3])
        self.bn3 = nn.ModuleList([
            nn.BatchNorm2d(dim_mid),
            nn.BatchNorm2d(dim_mid),
            nn.BatchNorm2d(dim_mid)
        ])
        self.conv3 = SharedConv(dim_in=dim_mid, dim_out=dim_out, kernel_size=1, stride=1, dilation=[1, 1, 1])
        self.downsample = None
        if stride == 2:
            self.downsample = SharedConv(dim_in=dim_in, dim_out=dim_out, kernel_size=1, stride=2, dilation=[1, 1, 1])

    def forward(self, x):
        residual = x
        x = [self.relu[i](self.bn1[i](x[i])) for i in range(len(x))]
        x = self.conv1(x)
        x = [self.relu[i](self.bn2[i](x[i])) for i in range(len(x))]
        x = self.conv2(x)
        x = [self.relu[i](self.bn3[i](x[i])) for i in range(len(x))]
        x = self.conv3(x)
        if self.downsample is not None:
            residual = self.downsample(residual)
        x = [x[i] + residual[i] for i in range(len(x))]
        return x


class BottleNeckV2(nn.Module):
    expansion = 4

    def __init__(self, dim_in, dim_out, stride, downsample=None):
        super(BottleNeckV2, self).__init__()
        dim_mid = dim_out // 4
        self.downsample = downsample
        self.conv1 = nn.Conv2d(dim_in, dim_mid, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(dim_in)
        self.conv2 = nn.Conv2d(dim_mid, dim_mid, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(dim_mid)
        self.conv3 = nn.Conv2d(dim_mid, dim_out, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(dim_mid)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        return out


class ResTridentStage(nn.Module):
    def __init__(self, dim_in, dim_out, stride=1, num_blocks=23, num_branchs=3):
        super(ResTridentStage, self).__init__()
        self.num_branchs = num_branchs

        layers = []
        downsample = nn.Sequential(
            nn.Conv2d(dim_in, dim_out,
                      kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(dim_out),
        )
        self.block = BottleNeckV2(dim_in=dim_in, dim_out=dim_out, stride=stride, downsample=downsample)

        for i in range(1, num_blocks):
            layers.append(ResTridentUnit(dim_in=dim_out, dim_out=dim_out))
        self.layer = nn.Sequential(*layers)

    @staticmethod
    def stack_branch_data(x):
        x = torch.cat(x[:], dim=0)
        return x

    def forward(self, x):
        x = self.block(x)
        x = [x] * self.num_branchs
        x = self.layer(x)
        x = self.stack_branch_data(x)
        return x


class ResV2TridentNet(nn.ModuleList):
    def __init__(self, block, layers):
        super(ResV2TridentNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, 256, layers[0])
        self.layer2 = self._make_layer(block, 256, 512, layers[1], stride=2)
        # self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer3 = ResTridentStage(dim_in=512,dim_out=1024, stride=2, num_blocks=layers[2])
        self.layer4 = self._make_layer(block, 1024, 2048, layers[3], stride=1)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, dim_in, dim_out, blocks, stride=1):
        downsample = None
        if stride != 1 or dim_in != dim_out:
            downsample = nn.Sequential(
                nn.Conv2d(dim_in, dim_out,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(dim_out),
            )
        layers = []
        layers.append(block(dim_in, dim_out, stride, downsample))
        for i in range(1, blocks):
            layers.append(block(dim_out, dim_out, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        l1 = self.layer1(x)
        l2 = self.layer2(l1)
        l3 = self.layer3(l2)
        l4 = self.layer4(l3)

        return l1,l2,l3,l4

def trident_res101v2():
    m = ResV2TridentNet(block=BottleNeckV2, layers=[3, 4, 23, 3])
    return m

def trident_res50v2():
    m = ResV2TridentNet(block=BottleNeckV2, layers=[3, 4, 6, 3])
    return m

if __name__ == '__main__':

    x = torch.randn(2,3,128,256)
    m = trident_res50v2()
    print(m)
    _,_,y1,y2 = m(x)
    print(y1.size())