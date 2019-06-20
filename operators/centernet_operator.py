import os
from models.centernet import CenterNet
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


class CenterNetOperator(BaseOperator):
    def __init__(self, cfg):
        self.cfg = cfg

        model = CenterNet(cfg).cuda()
        self.optimizer = optim.Adam(model.parameters(), lr=cfg.Train.lr)

        self.lr_sch = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=cfg.Train.lr_milestones, gamma=0.1)

        self.training_loader, self.validation_loader = make_dataloader(cfg, collate_fn='ctnet')

        super(CenterNetOperator, self).__init__(cfg=self.cfg, model=model, lr_sch=self.lr_sch)

        # TODO: change it to our class
        self.focal_loss = FocalLossHM()
        self.l1_loss = RegL1Loss()

    def criterion(self, outs, annos):
        hms, whs, offsets = outs
        t_hms, t_whs, t_inds, t_offsets, t_reg_masks = annos
        hm_loss, wh_loss, off_loss = 0, 0, 0

        for s in range(self.cfg.Model.num_stacks):
            hm = hms[s]
            wh = whs[s]
            offset = offsets[s]
            hm = torch.clamp(torch.sigmoid(hm), min=1e-4, max=1-1e-4)
            # Heatmap Loss
            hm_loss += self.focal_loss(hm, t_hms) / self.cfg.Model.num_stacks
            # WH Loss
            wh_loss += self.l1_loss(wh, t_reg_masks, t_inds, t_whs) / self.cfg.Model.num_stacks
            # OffSet Loss
            off_loss += self.l1_loss(offset, t_reg_masks, t_inds, t_offsets) / self.cfg.Model.num_stacks
        return hm_loss, wh_loss, off_loss

    def training_process(self):
        logger = Logger(self.cfg)

        self.model.train()

        total_loss = 0
        total_hm_loss = 0
        total_wh_loss = 0
        total_off_loss = 0

        training_loader = iter(self.training_loader)

        for step in range(self.cfg.Train.iter_num):
            self.lr_sch.step()
            self.optimizer.zero_grad()

            try:
                imgs, annos, hms, whs, inds, offsets, reg_masks, names = next(training_loader)
            except StopIteration:
                training_loader = iter(self.training_loader)
                imgs, annos, hms, whs, inds, offsets, reg_masks, names = next(training_loader)

            imgs = imgs.cuda()
            hms = hms.cuda()
            whs = whs.cuda()
            inds = inds.cuda()
            offsets = offsets.cuda()
            reg_masks = reg_masks.cuda()

            targets = hms, whs, inds, offsets, reg_masks

            outs = self.model(imgs)

            hm_loss, wh_loss, off_loss = self.criterion(outs, targets)

            loss = hm_loss + (0.1 * wh_loss) + off_loss
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            total_hm_loss += hm_loss.item()
            total_wh_loss += wh_loss.item()
            total_off_loss += off_loss.item()

            if step % self.cfg.Train.print_interval == self.cfg.Train.print_interval - 1:
                # Loss
                for param_group in self.optimizer.param_groups:
                    lr = param_group['lr']
                log_data = {'scalar': {
                    'train/total_loss': total_loss / self.cfg.Train.print_interval,
                    'train/hm_loss': total_hm_loss / self.cfg.Train.print_interval,
                    'train/wh_loss': total_wh_loss / self.cfg.Train.print_interval,
                    'train/off_loss': total_off_loss / self.cfg.Train.print_interval,
                    'train/lr': lr
                }}

                # Visualization
                img = (denormalize(imgs[0].cpu()).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

                hm, wh, offset = outs[0][0], outs[1][0], outs[2][0]
                pred_bbox0 = self.transform_bbox(hm, wh, offset, scale_factor=self.cfg.Train.scale_factor).cpu()

                hm, wh, offset = outs[0][1], outs[1][1], outs[2][1]
                pred_bbox1 = self.transform_bbox(hm, wh, offset, scale_factor=self.cfg.Train.scale_factor).cpu()

                # Do nms
                pred_bbox0 = self._ext_nms(pred_bbox0)
                pred_bbox1 = self._ext_nms(pred_bbox1)

                pred0_on_img = visualize(img.copy(), pred_bbox0, xywh=False, with_score=True)
                pred1_on_img = visualize(img.copy(), pred_bbox1, xywh=False, with_score=True)
                gt_on_img = visualize(img, annos[0])
                pred0_on_img = torch.from_numpy(pred0_on_img).permute(2, 0, 1).unsqueeze(0).float() / 255.
                pred1_on_img = torch.from_numpy(pred1_on_img).permute(2, 0, 1).unsqueeze(0).float() / 255.
                gt_on_img = torch.from_numpy(gt_on_img).permute(2, 0, 1).unsqueeze(0).float() / 255.

                log_data['imgs'] = {'train': [pred0_on_img, pred1_on_img, gt_on_img]}
                logger.log(log_data, step)

                total_loss = 0
                total_hm_loss = 0
                total_wh_loss = 0
                total_off_loss = 0

            if step % self.cfg.Train.checkpoint_interval == self.cfg.Train.checkpoint_interval - 1 or \
                    step == self.cfg.Train.iter_num - 1:
                self.save_ckp(self.model.module, step, logger.log_dir)

    def transform_bbox(self, hm, wh, offset, k=850, scale_factor=4):
        batchsize, cls_num, h, w = hm.size()
        hm = torch.sigmoid(hm)

        scores, inds, clses, ys, xs = self._topk(hm, k)

        if offset is not None:
            offset = self._transpose_and_gather_feat(offset, inds)
            offset = offset.view(batchsize, k, 2)
            xs = xs.view(batchsize, k, 1) + offset[:, :, 0:1]
            ys = ys.view(batchsize, k, 1) + offset[:, :, 1:2]
        else:
            xs = xs.view(batchsize, k, 1) + 0.5
            ys = ys.view(batchsize, k, 1) + 0.5
        wh = self._transpose_and_gather_feat(wh, inds)

        wh = wh.view(batchsize, k, 2)
        clses = clses.view(batchsize, k, 1).float() + 1
        scores = scores.view(batchsize, k, 1)

        pred_x = (xs - wh[..., 0:1] / 2) * scale_factor
        pred_y = (ys - wh[..., 1:2] / 2) * scale_factor
        pred_w = wh[..., 0:1] * scale_factor
        pred_h = wh[..., 1:2] * scale_factor
        pred = torch.cat([pred_x[0], pred_y[0], pred_w[0], pred_h[0], scores[0], clses[0]], dim=1)
        pred = pred[pred[:, 4] > 0.1, :]
        return pred

    def _transpose_and_gather_feat(self, feat, ind):
        feat = feat.permute(0, 2, 3, 1).contiguous()
        feat = feat.view(feat.size(0), -1, feat.size(3))
        feat = self._gather_feat(feat, ind)
        return feat

    def _topk(self, scores, k=40):
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

    @staticmethod
    def _ctnet_nms(heat, kernel=3):
        pad = (kernel - 1) // 2

        hmax = nn.functional.max_pool2d(
            heat, (kernel, kernel), stride=1, padding=pad)
        keep = (hmax == heat).float()
        return heat * keep

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

    @staticmethod
    def _ext_nms(pred_bbox, gpu_id=None):
        if pred_bbox.size(0) == 0:
            return pred_bbox
        cls_unique = pred_bbox[:, 5].unique()
        keep_bboxs = []
        for cls in cls_unique:
            cls_idx = pred_bbox[:, 5] == cls
            bbox_for_nms = pred_bbox[cls_idx].detach().cpu().numpy()
            bbox_for_nms[:, 2] = bbox_for_nms[:, 0] + bbox_for_nms[:, 2]
            bbox_for_nms[:, 3] = bbox_for_nms[:, 1] + bbox_for_nms[:, 3]
            keep_idx = nms(bbox_for_nms[:, :5], thresh=0.7, gpu_id=gpu_id)
            keep_bbox = bbox_for_nms[keep_idx]
            keep_bboxs.append(keep_bbox)
        keep_bboxs = np.concatenate(keep_bboxs, axis=0)
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
