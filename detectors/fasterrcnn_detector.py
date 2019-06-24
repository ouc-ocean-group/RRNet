import torch.nn as nn
from backbones.resnet import Bottleneck


class FasterRCNNDetector(nn.Module):
    def __init__(self, cls_num):
        super(FasterRCNNDetector, self).__init__()

        self.top_layer = Bottleneck(inplanes=256, planes=128)

        self.classifier = nn.Conv2d(512, cls_num, kernel_size=1)
        self.regressor = nn.Conv2d(512, 4, kernel_size=1)

    def forward(self, x, hm, wh, offset, inds):
        bboxes = self.get_bboxes(hm, wh, offset, inds)
        roi_feat = self.roi_align(x, bboxes)
        feat = self.top_layer(roi_feat)
        cls = self.classifier(feat)
        reg = self.regressor(feat)

        return cls, reg

    def get_bboxes(self, hm, wh, offset, inds):
        """
        Transform and get bbox.
        :param hm: b x 9 x h x w
        :param wh: b x 36 x h x w
        :param offset: b x 2 x h x w
        :param inds: b x
        :return:
        """
