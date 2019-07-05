import os
from models.box9_net import Box9Net
from modules.loss.focalloss import FocalLossHM
import numpy as np
from modules.loss.regl1loss import RegL1Loss
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from .base_operator import BaseOperator

from datasets import make_dataloader
from utils.vis.logger import Logger
from datasets.transforms.functional import denormalize
from utils.vis.annotations import visualize
from ext.nms.nms_wrapper import nms
from utils.metrics.metrics import bbox_iou


class TwoStageOperator(BaseOperator):
    def __init__(self, cfg):
        self.cfg = cfg

        model = Box9Net(cfg).cuda(cfg.Distributed.gpu_id)
        self.optimizer = optim.Adam(model.parameters(), lr=cfg.Train.lr)

        self.lr_sch = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=cfg.Train.lr_milestones, gamma=0.1)

        self.training_loader, self.validation_loader = make_dataloader(cfg, collate_fn='box9net')

        super(TwoStageOperator, self).__init__(cfg=self.cfg, model=model, lr_sch=self.lr_sch)

        # TODO: change it to our class
        self.focal_loss = FocalLossHM()
        self.l1_loss = RegL1Loss()

        self.main_proc_flag = cfg.Distributed.gpu_id == 0

    def criterion(self, outs, targets):
        hm, stage1_wh, offset, stage1_cls, for_stage2_wh, for_stage2_offset, stage2_cls, stage2_wh, inds, value = outs
        gt_hms, gt_whs, gt_inds, gt_offsets, gt_reg_masks, gt_cls, gt_annos = targets

        gt_whs = gt_whs.view(-1, 4)
        gt_offsets = gt_offsets.unsqueeze(2).repeat(1, 1, 9, 1).view(-1, 2)
        gt_reg_masks = gt_reg_masks.view(-1, 1)

        hm = torch.clamp(torch.sigmoid(hm), min=1e-4, max=1-1e-4)
        # I. Heatmap Loss
        hm_loss = self.focal_loss(hm, gt_hms)
        # II. Stage1 WH Loss
        stage1_wh_loss = (F.l1_loss(stage1_wh, gt_whs, reduction='none') * gt_reg_masks).sum() / (gt_reg_masks.sum() + 1e-6)
        # III. Offset Loss
        stage1_offset_loss = (F.l1_loss(offset, gt_offsets, reduction='none') * gt_reg_masks).sum() / (gt_reg_masks.sum() + 1e-6)
        # IV. Stage2 Class Loss
        # Calculate IOU between prediction and bbox
        # 1. Transform bbox.
        bs, _, _, w = hm.size()
        inds = inds.view(bs, -1)
        for_stage2_wh = for_stage2_wh.view(bs, -1, 4)
        for_stage2_offset = for_stage2_offset.view(bs, -1, 2)
        stage2_cls = stage2_cls.view(bs, -1, stage2_cls.size(1))
        stage2_wh = stage2_wh.view(bs, -1, stage2_wh.size(1))
        stage2_cls_loss = 0
        stage2_wh_loss = 0
        for b in range(bs):
            gt_anno = gt_annos[b]
            pred_x, pred_y, pred_w, pred_h, point_x, point_y =\
                self.inds_to_xywh(inds[b], for_stage2_wh[b], for_stage2_offset[b], width=w)
            pred = torch.stack((pred_x, pred_y, pred_w, pred_h), dim=1) * self.cfg.Train.scale_factor
            iou = bbox_iou(pred, gt_anno[:, :4], x1y1x2y2=False)
            max_v, max_idx = torch.max(iou, dim=1)
            pos_idx = max_v > 0.5
            target_cls = (gt_cls[b][max_idx] * pos_idx.float()).long()
            pred_cls = stage2_cls[b]
            stage2_cls_loss += F.cross_entropy(pred_cls, target_cls)
            gt_anno = gt_anno / self.cfg.Train.scale_factor
            w1 = point_x - gt_anno[max_idx, 0] - for_stage2_wh[b][:, 0]
            h1 = point_y - gt_anno[max_idx, 1] - for_stage2_wh[b][:, 1]
            w2 = gt_anno[max_idx, 0] + gt_anno[max_idx, 2] - point_x - for_stage2_wh[b][:, 2]
            h2 = gt_anno[max_idx, 1] + gt_anno[max_idx, 3] - point_y - for_stage2_wh[b][:, 3]

            target_wh = torch.stack((w1, h1, w2, h2), dim=1) / self.cfg.Train.scale_factor
            pred_wh = stage2_wh[b]
            stage2_wh_loss += (F.smooth_l1_loss(pred_wh, target_wh, reduction='none') * pos_idx.float().unsqueeze(1)).sum() / (pos_idx.float().sum() + 1e-6)
        stage2_cls_loss /= bs
        stage2_wh_loss /= bs

        return hm_loss, stage1_wh_loss, stage1_offset_loss, stage2_cls_loss, stage2_wh_loss

    def training_process(self):
        if self.main_proc_flag:
            logger = Logger(self.cfg)

        self.model.train()

        total_loss = 0
        total_hm_loss = 0
        total_wh_loss = 0
        total_off_loss = 0
        total_cls_loss = 0
        total_stage2_wh_loss = 0

        epoch = 0
        self.training_loader.sampler.set_epoch(epoch)
        training_loader = iter(self.training_loader)

        for step in range(self.cfg.Train.iter_num):
            self.lr_sch.step()
            self.optimizer.zero_grad()

            try:
                imgs, annos, gt_hms, gt_whs, gt_inds, gt_offsets, gt_reg_masks, names = next(training_loader)
            except StopIteration:
                epoch += 1
                self.training_loader.sampler.set_epoch(epoch)
                training_loader = iter(self.training_loader)
                imgs, annos, gt_hms, gt_whs, gt_inds, gt_offsets, gt_reg_masks, names = next(training_loader)

            imgs = imgs.cuda(self.cfg.Distributed.gpu_id)
            annos = annos.cuda(self.cfg.Distributed.gpu_id)
            gt_hms = gt_hms.cuda(self.cfg.Distributed.gpu_id)
            gt_whs = gt_whs.cuda(self.cfg.Distributed.gpu_id)
            gt_inds = gt_inds.cuda(self.cfg.Distributed.gpu_id)
            gt_offsets = gt_offsets.cuda(self.cfg.Distributed.gpu_id)
            gt_reg_masks = gt_reg_masks.cuda(self.cfg.Distributed.gpu_id)
            gt_stage2_cls = annos[:, :, 5]

            outs = self.model(imgs, gt_inds, k=500)
            targets = gt_hms, gt_whs, gt_inds, gt_offsets, gt_reg_masks, gt_stage2_cls, annos

            hm_loss, stage1_wh_loss, stage1_offset_loss, stage2_cls_loss, stage2_wh_loss = self.criterion(outs, targets)

            loss = hm_loss + (0.1 * stage1_wh_loss) + stage1_offset_loss + stage2_cls_loss + stage2_wh_loss
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            total_hm_loss += hm_loss.item()
            total_wh_loss += stage1_wh_loss.item()
            total_off_loss += stage1_offset_loss.item()
            total_cls_loss += stage2_cls_loss.item()
            total_stage2_wh_loss += stage2_wh_loss.item()

            if self.main_proc_flag:
                if step % self.cfg.Train.print_interval == self.cfg.Train.print_interval - 1:
                    # Loss
                    for param_group in self.optimizer.param_groups:
                        lr = param_group['lr']
                    log_data = {'scalar': {
                        'train/total_loss': total_loss / self.cfg.Train.print_interval,
                        'train/hm_loss': total_hm_loss / self.cfg.Train.print_interval,
                        'train/wh_loss': total_wh_loss / self.cfg.Train.print_interval,
                        'train/off_loss': total_off_loss / self.cfg.Train.print_interval,
                        'train/cls_loss': total_cls_loss / self.cfg.Train.print_interval,
                        'train/s2_wh_loss': total_stage2_wh_loss / self.cfg.Train.print_interval,
                        'train/lr': lr
                    }}

                    # Generate bboxs
                    s1_pred_bbox, s2_pred_bbox = self.generate_bbox(imgs[0:1], k=500)
                    gt_hms, gt_whs, gt_offsets, gt_stage2_cls, gt_inds = self.format_gt(gt_hms, gt_whs, gt_offsets, gt_stage2_cls, gt_inds)
                    s1_gt_bbox, s2_gt_bbox = self.transform_bbox(gt_hms, gt_whs, gt_offsets, gt_stage2_cls, None,
                                                                 gt_inds, scale_factor=4)

                    # Visualization
                    img = (denormalize(imgs[0].cpu()).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                    # Do nms
                    s1_pred_bbox = self._ext_nms(s1_pred_bbox, per_cls=False)
                    s2_pred_bbox = self._ext_nms(s2_pred_bbox)
                    #
                    s1_pred_on_img = visualize(img.copy(), s1_pred_bbox, xywh=False, with_score=True)
                    s2_pred_on_img = visualize(img.copy(), s2_pred_bbox, xywh=False, with_score=True)
                    s1_gt_on_img = visualize(img.copy(), s1_gt_bbox, xywh=True, with_score=True)
                    s2_gt_on_img = visualize(img.copy(), s2_gt_bbox, xywh=True, with_score=True)

                    s1_pred_on_img = torch.from_numpy(s1_pred_on_img).permute(2, 0, 1).unsqueeze(0).float() / 255.
                    s2_pred_on_img = torch.from_numpy(s2_pred_on_img).permute(2, 0, 1).unsqueeze(0).float() / 255.
                    s1_gt_on_img = torch.from_numpy(s1_gt_on_img).permute(2, 0, 1).unsqueeze(0).float() / 255.
                    s2_gt_on_img = torch.from_numpy(s2_gt_on_img).permute(2, 0, 1).unsqueeze(0).float() / 255.
                    log_data['imgs'] = {'stage1': [s1_gt_on_img, s1_pred_on_img],
                                        'stage2': [s2_gt_on_img, s2_pred_on_img]}
                    logger.log(log_data, step)

                    total_loss = 0
                    total_hm_loss = 0
                    total_wh_loss = 0
                    total_off_loss = 0
                    total_cls_loss = 0
                    total_stage2_wh_loss = 0

                if step % self.cfg.Train.checkpoint_interval == self.cfg.Train.checkpoint_interval - 1 or \
                        step == self.cfg.Train.iter_num - 1:
                    self.save_ckp(self.model.module, step, logger.log_dir)

    def generate_bbox(self, img, k=100, inds=None):
        with torch.no_grad():
            scale_factor = self.cfg.Train.scale_factor
            outs = self.model(img, inds, k=k)
            hm, stage1_wh, stage1_offset, stage1_cls, for_stage2_wh, for_stage2_offset, stage2_cls, stage2_wh, inds, stage1_score = outs
            hm = torch.sigmoid(hm)
            stage1_pred, stage2_pred =\
                self.transform_bbox(hm, for_stage2_wh, for_stage2_offset, stage2_cls, stage2_wh,
                                    inds, stage1_cls, stage1_score, scale_factor)
            return stage1_pred, stage2_pred

    @staticmethod
    def inds_to_xywh(inds, wh, offset, width):
        xs = (inds % width).view(-1).float()  # (b*n*9)
        ys = (inds // width).view(-1).float()  # (b*n*9)

        point_x = xs + offset[:, 0]  # (b*n*9)
        point_y = ys + offset[:, 1]  # (b*n*9)

        x = (point_x - wh[:, 0])
        y = (point_y - wh[:, 1])
        w = (wh[:, 0] + wh[:, 2])
        h = (wh[:, 1] + wh[:, 3])
        return x, y, w, h, point_x, point_y

    @staticmethod
    def format_gt(hms, whs, offsets, cls, inds):
        """

        :param hms: (b, 9, h, w)
        :param whs: (b, n, 9*4)
        :param offsets: (b, n, 2)
        :param cls: (b, n)
        :param inds: (b, n, 9)
        :return:
        """
        hm = hms[:, 4:5, :, :]
        whs = whs[:, :, 16:20].contiguous().view(-1, 4)
        offsets = offsets.view(-1, 2)
        inds = inds[:, :, 4].view(-1)
        return hm, whs, offsets, cls, inds

    def transform_bbox(self, hm, stage1_wh, stage1_offset, stage2_cls, stage2_wh,
                       inds, stage1_cls=None, stage1_score=None, scale_factor=4):
        bs, points_num, h, w = hm.size()

        # I. Stage 1.
        # a. Get ltx and lty of the predicted bboxes.
        s1_pred_x, s1_pred_y, s1_pred_w, s1_pred_h, _, _ = self.inds_to_xywh(inds, stage1_wh, stage1_offset, width=w)
        # b. Get score.
        if stage1_score is not None:
            stage1_score = stage1_score.view(-1).sigmoid()
        else:
            stage1_score = torch.ones_like(s1_pred_x)
        # c. Get class
        if stage1_cls is None:
            stage1_cls = torch.zeros_like(stage1_score)

        stage1_pred = torch.stack((s1_pred_x * scale_factor, s1_pred_y * scale_factor,
                                   s1_pred_w * scale_factor, s1_pred_h * scale_factor,
                                   stage1_score, stage1_cls.float())).permute(1, 0)
        stage1_pred = stage1_pred[stage1_pred[:, 4] > 0.1, :]
        _, indices = torch.sort(stage1_pred[:, 4])
        stage1_pred = stage1_pred[indices, :]

        # II. Stage2.
        # a. Get ltx and lty of the predicted bboxes.
        if stage2_wh is None:
            stage2_wh = torch.zeros_like(stage1_wh)
        s2_pred_x = s1_pred_x - stage2_wh[:, 0]
        s2_pred_y = s1_pred_y - stage2_wh[:, 1]
        s2_pred_w = s1_pred_w + stage2_wh[:, 0] + stage2_wh[:, 2]
        s2_pred_h = s1_pred_h + stage2_wh[:, 1] + stage2_wh[:, 3]
        # b. Get class and score.
        if stage2_cls.size(0) != bs:
            stage2_cls = torch.softmax(stage2_cls, dim=1)
            score, cls = torch.max(stage2_cls, dim=1)
        else:
            cls = stage2_cls.float().view(-1)
            score = torch.ones_like(cls).float()
        cls += 1
        # c. Cat
        stage2_pred = torch.stack((s2_pred_x * scale_factor, s2_pred_y * scale_factor,
                                   s2_pred_w * scale_factor, s2_pred_h * scale_factor,
                                   score, cls.float())).permute(1, 0)
        stage2_pred = stage2_pred[stage2_pred[:, 4] > 0.1, :]
        _, indices = torch.sort(stage2_pred[:, 4])
        stage2_pred = stage2_pred[indices, :]
        print(stage1_pred.size(), stage2_pred.size())
        return stage1_pred, stage2_pred

    def _ext_nms(self, pred_bbox, per_cls=True):
        if pred_bbox.size(0) == 0:
            return pred_bbox
        keep_bboxs = []
        if per_cls:
            cls_unique = pred_bbox[:, 5].unique()
            for cls in cls_unique:
                cls_idx = pred_bbox[:, 5] == cls
                bbox_for_nms = pred_bbox[cls_idx].detach().cpu().numpy()
                bbox_for_nms[:, 2] = bbox_for_nms[:, 0] + bbox_for_nms[:, 2]
                bbox_for_nms[:, 3] = bbox_for_nms[:, 1] + bbox_for_nms[:, 3]
                keep_idx = nms(bbox_for_nms[:, :5], thresh=0.3, gpu_id=self.cfg.Distributed.gpu_id)
                keep_bbox = bbox_for_nms[keep_idx]
                keep_bboxs.append(keep_bbox)
            keep_bboxs = np.concatenate(keep_bboxs, axis=0)
        else:
            bbox_for_nms = pred_bbox.detach().cpu().numpy()
            bbox_for_nms[:, 2] = bbox_for_nms[:, 0] + bbox_for_nms[:, 2]
            bbox_for_nms[:, 3] = bbox_for_nms[:, 1] + bbox_for_nms[:, 3]
            keep_idx = nms(bbox_for_nms[:, :5], thresh=0.3, gpu_id=self.cfg.Distributed.gpu_id)
            keep_bboxs = bbox_for_nms[keep_idx]
        return torch.from_numpy(keep_bboxs)

    @staticmethod
    def save_result(file_path, pred_bbox):
        pred_bbox = torch.clamp(pred_bbox, min=0.)
        with open(file_path, 'w') as f:
            for i in range(pred_bbox.size()[0]):
                bbox = pred_bbox[i]
                line = '%d,%d,%d,%d,%.4f,%d,-1,-1\n' % (
                    int(bbox[0]), int(bbox[1]), int(bbox[2])-int(bbox[0]), int(bbox[3])-int(bbox[1]),
                    float(bbox[4]), int(bbox[5])
                )
                f.write(line)

    def evaluation_process(self):
        self.model.eval()

        state_dict = torch.load(self.cfg.Val.model_path)
        self.model.module.load_state_dict(state_dict)
        step = 0
        all_step = len(self.validation_loader)

        with torch.no_grad():
            for data in self.validation_loader:
                step += 1
                imgs, annos, names = data
                imgs = imgs.cuda()

                outs = self.model(imgs)

                hm, wh, offset = outs[0][1], outs[1][1], outs[2][1]
                pred_bbox1 = self.transform_bbox(hm, wh, offset, scale_factor=self.cfg.Train.scale_factor).cpu()

                # Do nms
                pred_bbox1 = self._ext_nms(pred_bbox1, gpu_id=0)

                file_path = os.path.join(self.cfg.Val.result_dir, names[0] + '.txt')
                self.save_result(file_path, pred_bbox1)

                del outs
                del pred_bbox1
                print("\r[{}/{}]".format(step, all_step), end='', flush=True)
            print('=> Evaluation Done!')
