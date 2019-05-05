import torch
import torch.nn as nn
import torch.nn.functional as F
from .kp import KP
from .utils import convolution, residual

def make_hg_layer(kernel, dim0, dim1, mod, layer=convolution, **kwargs):
    layers  = [layer(kernel, dim0, dim1, stride=2)]
    layers += [layer(kernel, dim1, dim1) for _ in range(mod - 1)]
    return nn.Sequential(*layers)

def make_pool_layer(dim):
    return nn.Sequential()


class HourglassNet(nn.Module):
    def __init__(self):
        super(HourglassNet, self).__init__()
        self.n = 5
        self.dims = [256, 256, 384, 384, 384, 512]
        self.modules = [2, 2, 2, 2, 2, 4]

        self.kp = KP(self.n, 1, self.dims, self.modules,
                     make_pool_layer=make_pool_layer,
                     make_hg_layer=make_hg_layer,
                     kp_layer=residual, cnv_dim=256)

    def forward(self, x):
        output = self.kp(x)

        return output




if __name__ == "__main__":
    hourglassnet = HourglassNet()
    input = torch.randn(1, 3, 256, 256)
    out = hourglassnet(input)
    print(out.size())


