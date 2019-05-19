import torch
import torch.nn.functional as F
import torch.optim as optim
from .base_operator import BaseOperator
from models.retinanet import RetinaNet
from datasets import make_dataloader
from utils.vis.logger import Logger
from modules.loss.focalloss import FocalLoss
from modules.anchor import Anchors
from utils.metrics.metrics import bbox_iou


class RetinaNetOperator(BaseOperator):
    def __init__(self, cfg):
        self.cfg = cfg

        model = RetinaNet(cfg).cuda(cfg.Distributed.gpu_id)

        self.optimizer = optim.SGD(model.parameters(),
                                   lr=cfg.Train.lr, momentum=cfg.Train.momentum, weight_decay=cfg.Train.weight_decay)

        self.lr_sch = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=cfg.Train.lr_milestones, gamma=0.1)

        self.training_loader, self.validation_loader = make_dataloader(cfg)

        super(RetinaNetOperator, self).__init__(
            cfg=self.cfg, model=model, optimizer=self.optimizer, lr_sch=self.lr_sch)

        anchor_maker = Anchors()
        # Make the default anchors for training.
        # During the inference phase, we should create different default anchors for images in different size.
        self.anchors = anchor_maker(cfg.Train.crop_size)

        self.focal_loss = FocalLoss(class_num=cfg.num_classes, ignore=-1)

        self.main_proc_flag = cfg.Distributed.gpu_id == 0

    def criterion(self, outs, annos):
        annos[:, :, 2] += annos[:, :, 0]
        annos[:, :, 3] += annos[:, :, 1]

        anchors = self.anchors

        anchors_widths = anchors[:, 2] - anchors[:, 0]
        anchors_heights = anchors[:, 3] - anchors[:, 1]
        anchors_ctr_x = anchors[:, 0] + 0.5 * anchors_widths
        anchors_ctr_y = anchors[:, 1] + 0.5 * anchors_heights

        cls_targets = []
        reg_targets = []
        reg_preds = []

        for n in range(outs.size(0)):
            anno = annos[n]
            iou = bbox_iou(anno[:, :4], anchors)  # bbox num * anchor num  e.g., (20, 49104)
            max_iou, max_idx = torch.max(iou, dim=0)
            pos_idx = torch.ge(max_iou, 0.5)
            neg_idx = torch.lt(max_iou, 0.4)

            assigned_anno = anno[max_idx[pos_idx], :]

            # I. Classification loss
            pos_cls = assigned_anno[:, 5]
            cls_target = (torch.ones(anchors.size(0)) * -1).cuda()
            cls_target[pos_idx] = pos_cls
            cls_target[cls_target == 0] = -1
            cls_target[neg_idx] = 0
            cls_target = cls_target.long()
            cls_targets.append(cls_target.unsqueeze(0))

            # II. Regression loss

            if pos_idx.sum() > 0:
                anchor_widths_pi = anchors_widths[pos_idx]
                anchor_heights_pi = anchors_heights[pos_idx]
                anchor_ctr_x_pi = anchors_ctr_x[pos_idx]
                anchor_ctr_y_pi = anchors_ctr_y[pos_idx]

                gt_widths = assigned_anno[:, 2]
                gt_heights = assigned_anno[:, 3]
                gt_ctr_x = assigned_anno[:, 0] + 0.5 * gt_widths
                gt_ctr_y = assigned_anno[:, 1] + 0.5 * gt_heights

                gt_widths = torch.clamp(gt_widths, min=1)
                gt_heights = torch.clamp(gt_heights, min=1)

                target_dx = (gt_ctr_x - anchor_ctr_x_pi) / anchor_widths_pi
                target_dy = (gt_ctr_y - anchor_ctr_y_pi) / anchor_heights_pi
                target_dw = torch.log(gt_widths / anchor_widths_pi)
                target_dh = torch.log(gt_heights / anchor_heights_pi)

                reg_target = torch.stack((target_dx, target_dy, target_dw, target_dh))
                reg_target = reg_target.t()

                # TODO: Why should we use this normalization?
                reg_target = reg_target / torch.Tensor([[0.1, 0.1, 0.2, 0.2]]).cuda()

                reg_pred = outs[0][n, pos_idx, :]

                reg_preds.append(reg_pred)
                reg_targets.append(reg_target)

        reg_preds = torch.cat(reg_preds)
        reg_targets = torch.cat(reg_targets)
        reg_loss = F.smooth_l1_loss(reg_preds, reg_targets)

        cls_targets = torch.cat(cls_targets)
        cls_loss = self.focal_loss(outs[1], cls_targets) / reg_targets.size(0)
        return cls_loss, reg_loss

    def training_process(self):
        if self.main_proc_flag:
            logger = Logger(self.cfg)

        self.model.train()

        total_loss = 0
        total_cls_loss = 0
        total_loc_loss = 0

        epoch = 0
        self.training_loader.sampler.set_epoch(epoch)
        training_loader = iter(self.training_loader)

        for step in range(self.cfg.Train.iter_num):
            self.lr_sch.step()
            self.optimizer.zero_grad()

            try:
                imgs, annos = next(training_loader)
            except StopIteration:
                epoch += 1
                self.training_loader.sampler.set_epoch(epoch)
                training_loader = iter(self.training_loader)
                imgs, annos = next(training_loader)

            imgs = imgs.cuda(self.cfg.Distributed.gpu_id)
            annos = annos.cuda(self.cfg.Distributed.gpu_id)

            outs = self.model(imgs)
            cls_loss, loc_loss = self.criterion(outs, annos)
            loss = cls_loss + loc_loss
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            total_cls_loss += cls_loss.item()
            total_loc_loss += loc_loss.item()

            if self.main_proc_flag:
                if step % self.cfg.Train.print_interval == self.cfg.Train.print_interval - 1:
                    # Loss
                    log_data = {'scalar': {
                        'train/total_loss': total_loss / self.cfg.Train.print_interval,
                        'train/cls_loss': total_cls_loss / self.cfg.Train.print_interval,
                        'train/loc_loss': total_loc_loss / self.cfg.Train.print_interval,
                    }}

                    logger.log(log_data, step)

                    total_loss = 0
                    total_cls_loss = 0
                    total_loc_loss = 0

                if step % self.cfg.Train.checkpoint_interval == self.cfg.Train.checkpoint_interval - 1 or \
                        step == self.cfg.Train.iter_num - 1:
                    self.save_ckp(self.model.module, step, logger.log_dir)

    def evaluation_process(self):
        pass
