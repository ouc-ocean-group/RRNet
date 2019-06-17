import cv2
import os
from models.centernet import CenterNet
from utils.metrics.metrics import evaluate_results
from modules.loss.focalloss2 import FocalLoss
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

        model = CenterNet(cfg).cuda(cfg.Distributed.gpu_id)

        self.optimizer = optim.Adam(model.parameters(), lr=cfg.Train.lr)

        self.lr_sch = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=cfg.Train.lr_milestones, gamma=0.1)

        self.training_loader, self.validation_loader = make_dataloader(cfg, collate_fn='ctnet')

        super(CenterNetOperator, self).__init__(cfg=self.cfg, model=model, lr_sch=self.lr_sch)

        # TODO: change it to our class
        self.focal_loss = FocalLoss()
        self.l1_loss = RegL1Loss()

        self.main_proc_flag = cfg.Distributed.gpu_id == 0

    def criterion(self, outs, annos):
        hms, whs, regs = outs
        t_hms, t_whs, t_regs, t_inds, t_reg_masks = annos
        hm_loss, wh_loss, off_loss = 0, 0, 0
        for s in range(self.cfg.Model.num_stacks):
            hm = hms[s]
            wh = whs[s]
            reg = regs[s]
            hm = torch.clamp(hm.sigmoid_(), min=1e-4, max=1 - 1e-4)
            # Heatmap Loss
            hm_loss += self.focal_loss(hm, t_hms) / self.cfg.Model.num_stacks
            # WH Loss
            wh_loss += self.l1_loss(wh, t_reg_masks, t_inds, t_whs) / self.cfg.Model.num_stacks
            # OffSet Loss
            off_loss += self.l1_loss(reg, t_reg_masks, t_inds, t_regs) / self.cfg.Model.num_stacks
        return hm_loss, wh_loss, off_loss

    def training_process(self):
        if self.main_proc_flag:
            logger = Logger(self.cfg)

        self.model.train()

        total_loss = 0
        total_hm_loss = 0
        total_wh_loss = 0
        total_off_loss = 0

        epoch = 0
        self.training_loader.sampler.set_epoch(epoch)
        training_loader = iter(self.training_loader)

        for step in range(self.cfg.Train.iter_num):
            self.lr_sch.step()
            self.optimizer.zero_grad()

            try:
                imgs, annos, hms, whs, regs, inds, reg_masks, names = next(training_loader)
            except StopIteration:
                epoch += 1
                self.training_loader.sampler.set_epoch(epoch)
                training_loader = iter(self.training_loader)
                imgs, annos, hms, whs, regs, inds, reg_masks, names = next(training_loader)

            imgs = imgs.cuda(self.cfg.Distributed.gpu_id)
            hms = hms.cuda(self.cfg.Distributed.gpu_id)
            whs = whs.cuda(self.cfg.Distributed.gpu_id)
            regs = regs.cuda(self.cfg.Distributed.gpu_id)
            inds = inds.cuda(self.cfg.Distributed.gpu_id)
            reg_masks = reg_masks.cuda(self.cfg.Distributed.gpu_id)

            annos = hms, whs, regs, inds, reg_masks

            outs = self.model(imgs)

            hm_loss, wh_loss, off_loss = self.criterion(outs, annos)

            loss = hm_loss + (0.1 * wh_loss) + off_loss
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            total_hm_loss += hm_loss.item()
            total_wh_loss += wh_loss.item()
            total_off_loss += off_loss.item()

            if self.main_proc_flag:
                if step % self.cfg.Train.print_interval == self.cfg.Train.print_interval - 1:
                    # Loss
                    log_data = {'scalar': {
                        'train/total_loss': total_loss / self.cfg.Train.print_interval,
                        'train/hm_loss': total_hm_loss / self.cfg.Train.print_interval,
                        'train/wh_loss': total_wh_loss / self.cfg.Train.print_interval,
                        'train/off_loss': total_off_loss / self.cfg.Train.print_interval,
                    }}

                    img = (denormalize(imgs[0].cpu()).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                    pred_bbox = self.ctnet_transform_bbox(outs).cpu()

                    nms_bbox = pred_bbox[:, :5].detach().clone().numpy()
                    nms_bbox[:, 2] = nms_bbox[:, 0] + nms_bbox[:, 2]
                    nms_bbox[:, 3] = nms_bbox[:, 1] + nms_bbox[:, 3]
                    keep_idx = nms(nms_bbox, thresh=0.3, gpu_id=self.cfg.Distributed.gpu_id)
                    pred_bbox = pred_bbox[keep_idx]

                    vis_img = visualize_ctnet(img, pred_bbox)
                    vis_gt_img = visualize(img, gt[0])
                    vis_img = torch.from_numpy(vis_img).permute(2, 0, 1).unsqueeze(0).float() / 255.
                    vis_gt_img = torch.from_numpy(vis_gt_img).permute(2, 0, 1).unsqueeze(0).float() / 255.

                    log_data['imgs'] = {'train': [vis_img, vis_gt_img]}
                    logger.log(log_data, step)

                    total_loss = 0
                    total_hm_loss = 0
                    total_wh_loss = 0
                    total_off_loss = 0

                    print('lr: %g' % self.lr_sch.get_lr()[0])

                if step % self.cfg.Train.checkpoint_interval == self.cfg.Train.checkpoint_interval - 1 or \
                        step == self.cfg.Train.iter_num - 1:
                    self.save_ckp(self.model.module, step, logger.log_dir)

    def ctnet_transform_bbox(self, outs, K=850):
        heat = outs[0][1]
        wh = outs[1][1]
        reg = outs[2][1]
        batch, cat, height, width = heat.size()

        heat = torch.sigmoid(heat)
        # perform nms on heatmaps
        # heat = self._nms(heat)

        scores, inds, clses, ys, xs = self._topk(heat, K=K)
        if reg is not None:
            reg = self._tranpose_and_gather_feat(reg, inds)
            reg = reg.view(batch, K, 2)
            xs = xs.view(batch, K, 1) + reg[:, :, 0:1]
            ys = ys.view(batch, K, 1) + reg[:, :, 1:2]
        else:
            xs = xs.view(batch, K, 1) + 0.5
            ys = ys.view(batch, K, 1) + 0.5
        wh = self._tranpose_and_gather_feat(wh, inds)

        wh = wh.view(batch, K, 2)
        clses = clses.view(batch, K, 1).float()
        scores = scores.view(batch, K, 1)

        # pred_x = (xs - wh[..., 0:1] / 2) * 2
        pred_x = (xs - wh[..., 0:1] / 2) * 4
        # pred_y = (ys - wh[..., 1:2] / 2) * 2
        pred_y = (ys - wh[..., 1:2] / 2) * 4
        # pred_w = wh[..., 0:1] * 2
        pred_w = wh[..., 0:1] * 4
        # pred_h = wh[..., 1:2] * 2
        pred_h = wh[..., 1:2] * 4
        pred = torch.cat([pred_x[0], pred_y[0], pred_w[0], pred_h[0], scores[0], clses[0]], dim=1)
        # pred1 = pred[:, 4] >= 0.5 #Score Threshhold
        # pred = pred[pred1]
        return pred

    def _tranpose_and_gather_feat(self, feat, ind):
        feat = feat.permute(0, 2, 3, 1).contiguous()
        feat = feat.view(feat.size(0), -1, feat.size(3))
        feat = self._gather_feat(feat, ind)
        return feat

    def _topk(self, scores, K=40):
        batch, cat, height, width = scores.size()

        topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

        topk_inds = topk_inds % (height * width)
        topk_ys = (topk_inds / width).int().float()
        topk_xs = (topk_inds % width).int().float()

        topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
        topk_clses = (topk_ind / K).int()
        topk_inds = self._gather_feat(
            topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)
        topk_ys = self._gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
        topk_xs = self._gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)

        return topk_score, topk_inds, topk_clses, topk_ys, topk_xs

    def _nms(self, heat, kernel=3):
        pad = (kernel - 1) // 2

        hmax = nn.functional.max_pool2d(
            heat, (kernel, kernel), stride=1, padding=pad)
        keep = (hmax == heat).float()
        return heat * keep

    def _gather_feat(self, feat, ind, mask=None):
        dim = feat.size(2)
        ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
        feat = feat.gather(1, ind)
        if mask is not None:
            mask = mask.unsqueeze(2).expand_as(feat)
            feat = feat[mask]
            feat = feat.view(-1, dim)
        return feat

    @staticmethod
    def save_result(file_path, pred_bbox):
        pred_bbox = torch.clamp(pred_bbox, min=0.)
        with open(file_path, 'w') as f:
            for i in range(pred_bbox.size()[0]):
                bbox = pred_bbox[i]
                line = '%d,%d,%d,%d,%.4f,%d,-1,-1\n' % (
                    int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]),
                    float(bbox[4]), int(bbox[5] + 1)
                )
                f.write(line)

    def evaluation_process(self):
        self.model.eval()

        state_dict = torch.load(self.cfg.Val.model_path)
        self.model.module.load_state_dict(state_dict)
        epoch = 0
        step = 0

        self.validation_loader.sampler.set_epoch(epoch)
        gt_dir = os.path.join(self.cfg.data_root, 'val', 'annotations')
        with torch.no_grad():
            for data in self.validation_loader:
                step += 1
                imgs, hms, whs, regs, inds, reg_masks, gt, names = data
                imgs = imgs.cuda(self.cfg.Distributed.gpu_id)

                outs = self.model(imgs)
                pred_bbox = self.ctnet_transform_bbox(outs).cpu()

                # NMS

                nms_bbox = pred_bbox[:, :5].detach().clone().numpy()
                nms_bbox[:, 2] = nms_bbox[:, 0] + nms_bbox[:, 2]
                nms_bbox[:, 3] = nms_bbox[:, 1] + nms_bbox[:, 3]
                keep_idx = nms(nms_bbox, thresh=0.8, gpu_id=self.cfg.Distributed.gpu_id)
                pred_bbox = pred_bbox[keep_idx]

                file_path = os.path.join(self.cfg.Val.result_dir, names[0] + '.txt')
                self.save_result(file_path, pred_bbox)
                file_path = os.path.join('./result/', names[0] + '.txt')
                self.save_result(file_path, pred_bbox)

                img = (denormalize(imgs[0].cpu()).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                vis_img = visualize_ctnet(img, pred_bbox)
                cv2.imwrite('./' + names[0] + '.jpg', vis_img)
                evaluate_results('./result/', gt_dir)
                os.remove(file_path)
                print(names[0])

                del imgs
                del outs
                del pred_bbox
                if self.main_proc_flag:
                    print('Step : %d / %d' % (step, len(self.validation_loader)))
            print('Done !!!')
