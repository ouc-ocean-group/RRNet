import torch
import torch.nn as nn
import torchvision
from utils.model_tools import get_backbone
from detectors.centernet_detector import CenterNetDetector
from detectors.centernet_detector import CenterNetWHDetector
from detectors.fasterrcnn_detector import FasterRCNNDetector


class TwoStageNet(nn.Module):
    def __init__(self, cfg):
        super(TwoStageNet, self).__init__()
        self.num_stacks = cfg.Model.num_stacks
        self.num_classes = cfg.num_classes
        self.backbone = get_backbone(cfg.Model.backbone, num_stacks=self.num_stacks)
        self.hm = CenterNetDetector(planes=1, num_stacks=self.num_stacks, hm=True)
        self.wh = CenterNetWHDetector(planes=1, num_stacks=self.num_stacks)
        self.offset_reg = CenterNetDetector(planes=2, num_stacks=self.num_stacks)
        self.head_detector = FasterRCNNDetector(self.num_classes)

    def forward(self, x, k=1500):
        # I. Forward Backbone
        pre_feat = self.backbone(x)

        # II. Forward Stage 1 to generate heatmap, wh and offset.
        hms, whs, offsets = self.forward_stage1(pre_feat)
        # III. Generate the true indices for Stage 1.
        value, inds, = self.get_inds(hms[1], k)  # inds: (b, k)

        # IV. Stage 2.
        bxyxys = []
        scores = []
        for b_idx in range(inds.size(0)):
            # Select stage1_wh and offset by indices.
            stage1_wh = self.select_tensor(whs[1][b_idx], inds[b_idx])
            stage1_offset = self.select_tensor(offsets[1][b_idx], inds[b_idx])
            # Do nms
            xyxy = self.inds_to_xyxy(inds[b_idx], stage1_wh, stage1_offset, hms[1].size(3))
            keep_idx = torchvision.ops.nms(xyxy, value[b_idx].sigmoid(), 0.7)
            xyxy = xyxy[keep_idx, :]
            score = value[b_idx][keep_idx]
            scores.append(score)
            batch_idx = torch.ones((xyxy.size(0), 1), device=xyxy.device) * b_idx
            bxyxy = torch.cat((batch_idx, xyxy), dim=1)
            bxyxys.append(bxyxy)
        bxyxys = torch.cat(bxyxys, dim=0)
        scores = torch.cat(scores, dim=0)
        #  Generate the ROIAlign features.
        roi_feat = torchvision.ops.roi_align(pre_feat[-1], bxyxys, (3, 3))
        # Forward Stage 2 to predict class and wh offset.
        stage2_cls, stage2_reg = self.forward_stage2(roi_feat)
        return hms, whs, offsets, stage2_cls, stage2_reg, bxyxys, scores

    @staticmethod
    def inds_to_xyxy(inds, wh, offset, width):
        xs = (inds % width).float()  # (n)
        ys = (inds // width).float()  # (n)

        wh = wh.clamp(min=0)
        point_x = xs + offset[:, 0]  # (n)
        point_y = ys + offset[:, 1]  # (n)

        x1 = (point_x - wh[:, 0]/2)
        y1 = (point_y - wh[:, 1]/2)
        x2 = x1 + wh[:, 0]
        y2 = y1 + wh[:, 1]
        xyxy = torch.stack((x1, y1, x2, y2), dim=1)
        return xyxy

    @staticmethod
    def select_tensor(tensor, inds):
        """
        Select tensor by indices.
        :param tensor: (c, h, w)
        :param inds: (k) k is from 0 to h*w
        :return: (n, c)
        """
        c, h, w = tensor.size()
        tensor = tensor.view(c, -1).permute(1, 0)  # ([h*w], c)
        selected_tensor = tensor[inds, :]
        return selected_tensor

    @staticmethod
    def get_inds(hm, k=2000):
        """
        Get the indices for bbox generating, then use these bbox an ROIAlign to generate the final class and
        wh offset in stage2.
        :param hm: (b, 1, h, w)
        :param inds: None or (b, n, 1)
        :param k: keep top k indices if inds is None.
        :return:
        """
        bs, _, h, w = hm.size()
        hm = hm.view(bs, -1)
        top_v, top_inds = torch.topk(hm, k)  # (bs, k)
        return top_v, top_inds.long()

    def forward_stage1(self, feats):
        hms = []
        whs = []
        offsets = []
        for i in range(self.num_stacks):
            feat = feats[i]

            hm = self.hm(feat, i)
            wh = self.wh(feat, i)
            offset = self.offset_reg(feat, i)
            hms.append(hm)
            whs.append(wh)
            offsets.append(offset)
        return hms, whs, offsets

    def forward_stage2(self, feats,):
        stage_cls, stage2_wh = self.head_detector(feats)
        return stage_cls, stage2_wh
