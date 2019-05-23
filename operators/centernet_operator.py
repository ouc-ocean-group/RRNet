
from models.centernet import CenterNet
from modules.loss.focalloss2 import FocalLoss
import numpy as np
import math
from modules.loss.regl1loss import RegL1Loss
import torch
import torch.optim as optim
from .base_operator import BaseOperator

from datasets import make_dataloader
from utils.vis.logger import Logger

from datasets.transforms.functional import denormalize, gaussian_radius, draw_umich_gaussian
from utils.vis.annotations import visualize



class CenterNetOperator(BaseOperator):
    def __init__(self, cfg):
        self.cfg = cfg

        model = CenterNet(cfg).cuda(cfg.Distributed.gpu_id)

        self.optimizer = optim.SGD(model.parameters(),
                                   lr=cfg.Train.lr, momentum=cfg.Train.momentum, weight_decay=cfg.Train.weight_decay)

        self.lr_sch = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=cfg.Train.lr_milestones, gamma=0.1)

        self.training_loader, self.validation_loader = make_dataloader(cfg)

        super(CenterNetOperator, self).__init__(
            cfg=self.cfg, model=model, lr_sch=self.lr_sch)

        # TODO change it to our class
        self.focal_loss = FocalLoss()
        self.l1_loss = RegL1Loss()

        self.main_proc_flag = cfg.Distributed.gpu_id == 0

    def criterion(self, outs, annos):
        hms, whs, regs = outs
        t_hms, t_whs, t_regs, t_inds, t_reg_masks=annos
        hm_loss, wh_loss, off_loss = 0, 0, 0
        for s in range(self.cfg.Model.num_stacks):
            hm = hms[s]
            wh = whs[s]
            reg = regs[s]
            hm = torch.clamp(hm.sigmoid_(), min=1e-4, max=1 - 1e-4)
            # Heatmap Loss
            hm_loss += self.focal_loss(hm, t_hms) / self.cfg.num_stacks
            # WH Loss
            wh_loss += self.l1_loss(wh, t_reg_masks, t_inds, t_whs) / self.cfg.num_stacks
            # OffSet Loss
            off_loss += self.l1_loss(reg, t_reg_masks, t_inds, t_regs) / self.cfg.num_stacks
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
                # imgs, annos = next(training_loader)
                imgs, hms, whs, regs, inds, reg_masks = next(training_loader)
            except StopIteration:
                epoch += 1
                self.training_loader.sampler.set_epoch(epoch)
                training_loader = iter(self.training_loader)
                # imgs, annos = next(training_loader)
                imgs, hms, whs, regs, inds, reg_masks = next(training_loader)

            imgs = imgs.cuda(self.cfg.Distributed.gpu_id)
            hms = hms.cuda(self.cfg.Distributed.gpu_id)
            whs = whs.cuda(self.cfg.Distributed.gpu_id)
            regs = regs.cuda(self.cfg.Distributed.gpu_id)
            inds = inds.cuda(self.cfg.Distributed.gpu_id)
            reg_masks = reg_masks.cuda(self.cfg.Distributed.gpu_id)
            annos = hms, whs, regs, inds, reg_masks
            outs = self.model(imgs)
            # annos= self.trans_anns(imgs,annos)
            hm_loss, wh_loss, off_loss = self.criterion(outs, annos)
            loss = hm_loss + wh_loss + off_loss
            # TODO if here use loss.mean()
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
                    pred_bbox = self.transform_bbox(outs[1][0], outs[0][0]).cpu()
                    vis_img = visualize(img, pred_bbox)
                    vis_gt_img = visualize(img, annos[0])
                    vis_img = torch.from_numpy(vis_img).permute(2, 0, 1).unsqueeze(0).float() / 255.
                    vis_gt_img = torch.from_numpy(vis_gt_img).permute(2, 0, 1).unsqueeze(0).float() / 255.

                    log_data['imgs'] = {'train': [vis_img, vis_gt_img]}
                    logger.log(log_data, step)

                    total_loss = 0
                    total_hm_loss = 0
                    total_wh_loss = 0
                    total_off_loss = 0

                if step % self.cfg.Train.checkpoint_interval == self.cfg.Train.checkpoint_interval - 1 or \
                        step == self.cfg.Train.iter_num - 1:
                    self.save_ckp(self.model.module, step, logger.log_dir)
    def transform_bbox(self, cls_pred, loc_pred):
        """
        Transform prediction class and location into bbox coordinate.
        :param cls_pred: (AnchorN, ClassN)
        :param loc_pred: (AnchorsN, 4)
        :return: BBox, (N, 8)
        """
        cls_pred_prob, cls_pred_idx = cls_pred.max(dim=1)
        object_idx = cls_pred_prob > 0.05
        cls_prob = cls_pred_prob[object_idx]
        cls = cls_pred_idx[object_idx] + 1
        boxes = self.anchors[object_idx, :]
        deltas = loc_pred[object_idx, :]
        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        ctr_x = boxes[:, 0] + 0.5 * widths
        ctr_y = boxes[:, 1] + 0.5 * heights
        mean = torch.from_numpy(np.array([0, 0, 0, 0]).astype(np.float32)).cuda()
        std = torch.from_numpy(np.array([0.1, 0.1, 0.2, 0.2]).astype(np.float32)).cuda()
        dx = deltas[:, 0] * std[0] + mean[0]
        dy = deltas[:, 1] * std[1] + mean[1]
        dw = deltas[:, 2] * std[2] + mean[2]
        dh = deltas[:, 3] * std[3] + mean[3]

        pred_ctr_x = ctr_x + dx * widths
        pred_ctr_y = ctr_y + dy * heights
        pred_w = torch.exp(dw) * widths
        pred_h = torch.exp(dh) * heights

        pred_x = pred_ctr_x - 0.5 * pred_w
        pred_y = pred_ctr_y - 0.5 * pred_h

        pred = torch.stack([pred_x, pred_y, pred_w, pred_h, cls_prob, cls.float()]).t()
        return pred

    def evaluation_process(self):
        pass


    # TODO these function is np, it may cause bugs
    def trans_anns(self, images, anns):
        # trans anns to (hm,wh,reg) format
        batch = len(anns)
        hms = []
        whs = []
        regs = []
        inds = []
        reg_masks = []
        for i in range(batch):
            ann = anns[i]
            max_objs = len(ann)
            height, width = images[i].shape[1], images[i].shape[2]
            # init var
            hm = np.zeros((self.cfg.num_classes, height, width), dtype=np.float32)
            wh = np.zeros((max_objs, 2), dtype=np.float32)
            reg = np.zeros((max_objs, 2), dtype=np.float32)
            ind = np.zeros((max_objs), dtype=np.int64)
            reg_mask = np.zeros((max_objs), dtype=np.uint8)
            for k in range(max_objs):
                an = ann[k]
                # select bbox change box(x,y,h,w) to (x1,y1,x2,y2)
                bbox = an[0:4]
                bbox[2] += bbox[0]
                bbox[3] += bbox[1]
                # box class (object_category)
                cls_id = an[5]

                # cal and draw heatmap by gaussian
                h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
                if h > 0 and w > 0:
                    # draw heatmap
                    radius = gaussian_radius((math.ceil(h), math.ceil(w)))
                    radius = max(0, int(radius))
                    ct = np.array(
                        [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
                    ct_int = ct.astype(np.int32)
                    draw_umich_gaussian(hm[cls_id], ct_int, radius)
                    # cal wh
                    wh[k] = 1. * w, 1. * h
                    ind[k] = ct_int[1] * width + ct_int[0]
                    reg[k] = ct - ct_int
                    reg_mask[k] = 1
            hms.append(hm)
            whs.append(wh)
            inds.append(ind)
            regs.append(reg)
            reg_masks.append(reg_mask)
        return hms, whs, inds, regs, reg_masks
