import torch.nn as nn


class RetinaNetDetector(nn.Module):
    def __init__(self, planes):
        super(RetinaNetDetector, self).__init__()
        layers = []
        for _ in range(4):
            layers.append(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1))
            layers.append(nn.ReLU(True))
        layers.append(nn.Conv2d(256, planes, kernel_size=3, stride=1, padding=1))
        self.detect_layer = nn.Sequential(*layers)

    def forward(self, x):
        return self.detect_layer(x)



