import torch
import torch.nn as nn
from utils.model_tools import get_backbone
from utils.functional import roi_align
from detectors.centernet_detector import CenterNetDetector
from detectors.centernet_detector import CenterNetWHDetector
from detectors.fasterrcnn_detector import FasterRCNNDetector


class TwoStageNet(nn.Module):
    def __init__(self, cfg):
        super(TwoStageNet, self).__init__()
        self.num_stacks = 1
        self.num_classes = cfg.num_classes
        self.backbone = get_backbone(cfg.Model.backbone, num_stacks=self.num_stacks)
        self.hm = CenterNetDetector(planes=9, num_stacks=self.num_stacks, hm=True)
        self.wh = CenterNetWHDetector(planes=18, num_stacks=self.num_stacks)
        self.offset_reg = CenterNetDetector(planes=18, num_stacks=self.num_stacks)
        self.head_detector = FasterRCNNDetector(self.num_classes)

    def forward(self, x, inds=None, k=100):
        # I. Forward Backbone
        pre_feat = self.backbone(x)

        # II. Forward Stage 1 to generate heatmap, wh and offset.
        hm, wh, offset = self.forward_stage1(pre_feat)
        # III. Generate the true indices for Stage 1.
        value, inds, _ = self.get_inds(hm, inds, k)
        # Select stage1_wh and offset by indices.
        stage1_wh = self.select_tensor(wh, inds)
        stage1_offset = self.select_tensor(offset, inds)
        # IV. Generate the true indices for Stage 2.
        if inds is not None:
            value, inds, stage1_cls = self.get_inds(hm, None, k)
        for_stage2_wh = self.select_tensor(wh, inds)
        for_stage2_offset = self.select_tensor(offset, inds)
        # V. Generate the ROIAlign features.
        roied_feat = self.roi_align(pre_feat, for_stage2_wh, for_stage2_offset, inds)
        # VI. Forward Stage 2 to predict class and wh offset.
        stage2_cls, stage2_wh = self.forward_stage2(roied_feat)
        return hm, stage1_wh, stage1_offset, stage1_cls, for_stage2_wh, for_stage2_offset, stage2_cls, stage2_wh, inds, value

    @staticmethod
    def select_tensor(tensor, inds):
        """
        Select tensor by indices.
        :param tensor: (b, 9*c, h, w)
        :param inds: (b*n*9) or (b, n, 9), n is from 0 to h*w
        :return: (b*n*9, c)
        """
        bs, c9, h, w = tensor.size()
        c = int(c9 / 9)
        tensor = tensor.view(bs, 9, c, h*w).permute(0, 3, 1, 2).contiguous().view(-1, c)  # (b*[h*w]*9, c)
        selected_tensor = tensor[inds, :]
        return selected_tensor

    @staticmethod
    def roi_align(feat, wh, offset, inds):
        """
        Transform and get bbox.
        :param feat: (b, c, h, w)
        :param wh: (b*n*9, 4)
        :param offset: (b*n*9, 2)
        :param inds: (b, n, 9), n is from 0 to h*w
        :return:
        """
        bs, _, h, w = feat.size()
        # I. Convert indices to xy.
        xs = (inds % w).view(-1).float()  # (b*n*9)
        ys = (inds // w).view(-1).float()  # (b*n*9)

        xs = xs + offset[:, 0]  # (b*n*9)
        ys = ys + offset[:, 1]  # (b*n*9)

        k = int(ys.size(0) / bs)
        batch_idx = torch.arange(0, bs, device=feat.device, dtype=torch.float) \
            .unsqueeze(1).repeat(1, k).view(-1)

        x1 = xs + wh[:, 0]
        y1 = ys + wh[:, 1]
        x2 = xs + wh[:, 2]
        y2 = ys + wh[:, 3]
        bix1x2y1y2 = torch.stack((batch_idx, x1, y1, x2, y2), dim=1)
        roi_feat = roi_align(feat, bix1x2y1y2, (3, 3))  # (b x n x 9) x c x 3 x 3
        return roi_feat

    @staticmethod
    def get_inds(hm, inds=None, k=100):
        """
        Get the indices for bbox generating, then use these bbox an ROIAlign to generate the final class and
        wh offset in stage2.
        :param hm: (b, 9, h, w)
        :param inds: None or (b, n, 9)
        :param k: keep top k indices if inds is None.
        :return:
        """
        value = None
        cls = None
        bs, _, h, w = hm.size()
        inds_offset = (torch.arange(0, bs, device=hm.device, dtype=torch.long) * (h * w)).unsqueeze(1)
        if inds is None:
            hm = hm.permute(0, 2, 3, 1).contiguous().view(bs, -1)
            top_v, top_inds = torch.topk(hm, k)  # (bs, k)
            value = top_v.view(-1)
            cls = top_inds.view(-1) % 9
            inds = top_inds.view(-1) // 9 % (h*w)
        else:
            k = inds.size(1) * 9
            inds = inds.view(-1)
        inds = inds.long() + inds_offset.repeat(1, k).view(-1)
        return value, inds.long(), cls

    def forward_stage1(self, feat):
        hms = []
        whs = []
        regs = []
        for i in range(self.num_stacks):
            feat = feat[i]

            hm = self.hm(feat, i)
            wh = self.wh(feat, i)
            reg = self.reg(feat, i)
            hms.append(hm)
            whs.append(wh)
            regs.append(reg)

    def forward_stage2(self, feats,):
        stage_cls, stage2_wh = self.head_detector(feats)
        return stage_cls, stage2_wh
