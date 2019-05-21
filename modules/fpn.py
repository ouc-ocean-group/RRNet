import torch.nn as nn
import torch.nn.functional as F


class FPN(nn.Module):
    def __init__(self):
        super(FPN, self).__init__()
        # self.conv6 = nn.Conv2d(2048, 256, kernel_size=3, stride=2, padding=1)
        # self.conv7 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)

        # Lateral layers
        self.lat_layer1 = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0)
        self.lat_layer2 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
        self.lat_layer3 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)

        # Top-down layers
        self.top_layer1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.top_layer2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

    @staticmethod
    def upsample_add(x, y):
        """
        Upsample and add two feature maps.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
        Returns:
          (Variable) added feature map.
        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        """

        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False) + y

    def forward(self, c3, c4, c5):
        # p6 = self.conv6(c5)
        # p7 = self.conv7(F.relu(p6))
        # Top-down
        p5 = self.lat_layer1(c5)
        p4 = self.upsample_add(p5, self.lat_layer2(c4))
        p4 = self.top_layer1(p4)
        p3 = self.upsample_add(p4, self.lat_layer3(c3))
        p3 = self.top_layer2(p3)
        return p3, p4, p5
