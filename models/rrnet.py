import torch
import torch.nn as nn
import torchvision
from utils.model_tools import get_backbone
from detectors.centernet_detector import CenterNetDetector
from detectors.centernet_detector import CenterNetWHDetector
from detectors.fasterrcnn_detector import FasterRCNNDetector
from ext.nms.nms_wrapper import soft_nms


class RRNet(nn.Module):
    def __init__(self, cfg):
        super(RRNet, self).__init__()
        self.num_stacks = cfg.Model.num_stacks
        self.num_classes = cfg.num_classes
        self.nms_type = cfg.Model.nms_type_for_stage1
        self.nms_per_class = cfg.Model.nms_per_class_for_stage1

        self.backbone = get_backbone(cfg.Model.backbone, num_stacks=self.num_stacks)
        self.hm = CenterNetDetector(planes=self.num_classes, num_stacks=self.num_stacks, hm=True)
        self.wh = CenterNetWHDetector(planes=1, num_stacks=self.num_stacks)
        self.offset_reg = CenterNetDetector(planes=2, num_stacks=self.num_stacks)
        self.head_detector = FasterRCNNDetector()

    def forward(self, x, k=1500):
        # I. Forward Backbone
        pre_feat = self.backbone(x)
        # II. Forward Stage 1 to generate heatmap, wh and offset.
        hms, whs, offsets = self.forward_stage1(pre_feat)
        # III. Generate the true xywh for Stage 1.
        bboxs = self.transform_bbox(hms[-1], whs[-1], offsets[-1], k)  # (bs, k, 6)

        # IV. Stage 2.
        bxyxys = []
        scores = []
        clses = []
        for b_idx in range(bboxs.size(0)):
            # Do nms
            bbox = bboxs[b_idx]
            bbox = self.nms(bbox)
            xyxy = bbox[:, :4]
            scores.append(bbox[:, 4])
            clses.append(bbox[:, 5])
            batch_idx = torch.ones((xyxy.size(0), 1), device=xyxy.device) * b_idx
            bxyxy = torch.cat((batch_idx, xyxy), dim=1)
            bxyxys.append(bxyxy)
        bxyxys = torch.cat(bxyxys, dim=0)
        scores = torch.cat(scores, dim=0)
        clses = torch.cat(clses, dim=0)
        #  Generate the ROIAlign features.
        roi_feat = torchvision.ops.roi_align(torch.relu(pre_feat[-1]), bxyxys, (3, 3))
        # Forward Stage 2 to predict and wh offset.
        stage2_reg = self.forward_stage2(roi_feat)
        return hms, whs, offsets, stage2_reg, bxyxys, scores, clses

    def nms(self, bbox):
        device = bbox.device
        keep_bboxs = []
        if self.nms_per_class:
            cls_unique = bbox[:, 5].unique()
            for cls in cls_unique:
                cls_idx = bbox[:, 5] == cls
                bbox_for_nms = bbox[cls_idx]
                if self.nms_type == 'soft_nms':
                    bbox_for_nms = bbox_for_nms.detach().cpu().numpy()
                    keep_bbox = soft_nms(bbox_for_nms, Nt=0.7, threshold=0.1, method=2)
                    keep_bbox = torch.from_numpy(keep_bbox).to(device)
                else:
                    keep_idx = torchvision.ops.nms(bbox_for_nms[:, :4], bbox_for_nms[:, 4], 0.7)
                    keep_bbox = bbox_for_nms[keep_idx]
                keep_bboxs.append(keep_bbox)
            keep_bboxs = torch.cat(keep_bboxs)
        else:
            if self.nms_type == 'soft_nms':
                keep_bboxs = soft_nms(bbox.detach().cpu().numpy(), Nt=0.7, threshold=0.1, method=2)
                keep_bboxs = torch.from_numpy(keep_bboxs).to(device)
            else:
                keep_idx = torchvision.ops.nms(bbox[:, :4], bbox[:, 4], 0.7)
                keep_bboxs = bbox[keep_idx]
        return keep_bboxs

    @staticmethod
    def _gather_feat(feat, ind, mask=None):
        dim = feat.size(2)
        ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
        feat = feat.gather(1, ind)
        if mask is not None:
            mask = mask.unsqueeze(2).expand_as(feat)
            feat = feat[mask]
            feat = feat.view(-1, dim)
        return feat

    def _topk(self, scores, k=1500):
        batch, cat, height, width = scores.size()

        topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), k)

        topk_inds = topk_inds % (height * width)
        topk_ys = (topk_inds / width).int().float()
        topk_xs = (topk_inds % width).int().float()

        topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), k)
        topk_clses = (topk_ind / k).int()
        topk_inds = self._gather_feat(
            topk_inds.view(batch, -1, 1), topk_ind).view(batch, k)
        topk_ys = self._gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, k)
        topk_xs = self._gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, k)

        return topk_score, topk_inds, topk_clses, topk_ys, topk_xs

    def _transpose_and_gather_feat(self, feat, ind):
        feat = feat.permute(0, 2, 3, 1).contiguous()
        feat = feat.view(feat.size(0), -1, feat.size(3))
        feat = self._gather_feat(feat, ind)
        return feat

    def transform_bbox(self, hm, wh, offset, k=250):
        batchsize, cls_num, h, w = hm.size()
        hm = torch.sigmoid(hm)

        scores, inds, clses, ys, xs = self._topk(hm, k)

        offset = self._transpose_and_gather_feat(offset, inds)
        offset = offset.view(batchsize, k, 2)
        xs = xs.view(batchsize, k, 1) + offset[:, :, 0:1]
        ys = ys.view(batchsize, k, 1) + offset[:, :, 1:2]
        wh = self._transpose_and_gather_feat(wh, inds).clamp(min=0)

        wh = wh.view(batchsize, k, 2)
        clses = clses.view(batchsize, k, 1).float()
        scores = scores.view(batchsize, k, 1)

        pred_x = (xs - wh[..., 0:1] / 2)
        pred_y = (ys - wh[..., 1:2] / 2)
        pred_w = wh[..., 0:1]
        pred_h = wh[..., 1:2]
        pred = torch.cat([pred_x, pred_y, pred_w + pred_x, pred_h + pred_y, scores, clses], dim=2)
        return pred

    def forward_stage1(self, feats):
        hms = []
        whs = []
        offsets = []
        for i in range(self.num_stacks):
            feat = feats[i]
            feat = torch.relu(feat)
            hm = self.hm(feat, i)
            wh = self.wh(feat, i)
            offset = self.offset_reg(feat, i)
            hms.append(hm)
            whs.append(wh)
            offsets.append(offset)
        return hms, whs, offsets

    def forward_stage2(self, feats,):
        stage2_reg = self.head_detector(feats)
        return stage2_reg
