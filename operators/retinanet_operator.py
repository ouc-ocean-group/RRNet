import os
import torch
import torch.optim as optim
from .base_operator import BaseOperator
from models.retinanet import RetinaNet
from datasets import make_dataloader
from utils.vis.logger import Logger
from modules.loss.focalloss import FocalLoss
from modules.anchor import Anchors
from utils.metrics.metrics import bbox_iou
from datasets.transforms.functional import denormalize
from utils.vis.annotations import visualize
import numpy as np
from ext.nms.nms_wrapper import nms

class RetinaNetOperator(BaseOperator):
    def __init__(self, cfg):
        self.cfg = cfg

        model = RetinaNet(cfg).cuda(cfg.Distributed.gpu_id)

        self.optimizer = optim.Adam(model.parameters(), lr=cfg.Train.lr)

        self.lr_sch = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=cfg.Train.lr_milestones, gamma=0.1)

        self.training_loader, self.validation_loader = make_dataloader(cfg)

        super(RetinaNetOperator, self).__init__(cfg=self.cfg, model=model, lr_sch=self.lr_sch)

        self.anchor_maker = Anchors(sizes=(16, 64, 128))

        self.anchors, self.anchors_widths, self.anchors_heights, self.anchors_ctr_x, self.anchors_ctr_y = \
            self.make_anchor(cfg.Train.crop_size)

        self.focal_loss = FocalLoss()

        self.main_proc_flag = cfg.Distributed.gpu_id == 0

    def make_anchor(self, size):
        anchors = self.anchor_maker(size).cuda()
        anchors_widths = anchors[:, 2] - anchors[:, 0]
        anchors_heights = anchors[:, 3] - anchors[:, 1]
        anchors_ctr_x = anchors[:, 0] + 0.5 * anchors_widths
        anchors_ctr_y = anchors[:, 1] + 0.5 * anchors_heights
        return anchors, anchors_widths, anchors_heights, anchors_ctr_x, anchors_ctr_y

    def criterion(self, outs, annos):
        loc_preds, cls_preds = outs  # (bs, AnchorN, ClassN), (bs, AnchorN, 4), e.g., (4, 97965, 4)
        bs_num = cls_preds.size(0)

        annos[:, :, 2] += annos[:, :, 0]  # (bs, AnnoN, 8)
        annos[:, :, 3] += annos[:, :, 1]

        cls_losses = []
        reg_losses = []
        for n in range(bs_num):
            anno = annos[n]  # (AnnoN, 8), e.g., (97965, 8)
            iou = bbox_iou(anno[:, :4], self.anchors)  # (AnnoN, AnchorN)  e.g., (101, 97965)
            max_iou, max_idx = torch.max(iou, dim=0)  # (AnchorN)
            pos_idx = torch.ge(max_iou, 0.5)
            neg_idx = torch.lt(max_iou, 0.4)
            cls_idx = pos_idx + neg_idx
            # I. Classification loss
            cls_target = torch.zeros_like(cls_preds[n])
            cls_pred = cls_preds[n][cls_idx, :]

            assigned_anno = anno[max_idx[pos_idx], :]
            cls_target[pos_idx, assigned_anno[:, 5].long() - 1] = 1
            cls_target = cls_target[cls_idx]

            cls_loss = self.focal_loss(cls_pred, cls_target) / max(1., pos_idx.sum().float())
            cls_losses.append(cls_loss)
            # II. Regression loss

            if pos_idx.sum() > 0:
                anchor_widths_pi = self.anchors_widths[pos_idx]
                anchor_heights_pi = self.anchors_heights[pos_idx]
                anchor_ctr_x_pi = self.anchors_ctr_x[pos_idx]
                anchor_ctr_y_pi = self.anchors_ctr_y[pos_idx]

                gt_widths = assigned_anno[:, 2] - assigned_anno[:, 0]
                gt_heights = assigned_anno[:, 3] - assigned_anno[:, 1]
                gt_ctr_x = assigned_anno[:, 0] + 0.5 * gt_widths
                gt_ctr_y = assigned_anno[:, 1] + 0.5 * gt_heights

                gt_widths = torch.clamp(gt_widths, min=1)
                gt_heights = torch.clamp(gt_heights, min=1)

                with torch.no_grad():
                    target_dx = (gt_ctr_x - anchor_ctr_x_pi) / anchor_widths_pi
                    target_dy = (gt_ctr_y - anchor_ctr_y_pi) / anchor_heights_pi
                    target_dw = torch.log(gt_widths / anchor_widths_pi)
                    target_dh = torch.log(gt_heights / anchor_heights_pi)

                    reg_target = torch.stack((target_dx, target_dy, target_dw, target_dh))
                    reg_target = reg_target.t()
                    reg_target = reg_target / torch.tensor([[0.1, 0.1, 0.2, 0.2]]).cuda()

                reg_pred = loc_preds[n, pos_idx, :]

                # reg_loss = F.smooth_l1_loss(reg_pred, reg_target.detach())
                regression_diff = torch.abs(reg_target.detach() - reg_pred)

                reg_loss = torch.where(
                    torch.le(regression_diff, 1.0 / 9.0),
                    0.5 * 9.0 * torch.pow(regression_diff, 2),
                    regression_diff - 0.5 / 9.0
                )
                reg_losses.append(reg_loss.mean())
            else:
                reg_losses.append(torch.zeros(1).to(loc_preds.device))

        return sum(cls_losses) / bs_num, sum(reg_losses) / bs_num

    def training_process(self):
        logger = Logger(self.cfg, self.main_proc_flag)

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
                imgs, annos, names = next(training_loader)
            except StopIteration:
                epoch += 1
                self.training_loader.sampler.set_epoch(epoch)
                training_loader = iter(self.training_loader)
                imgs, annos, names = next(training_loader)
            imgs = imgs.cuda(self.cfg.Distributed.gpu_id)
            annos = annos.cuda(self.cfg.Distributed.gpu_id)
            outs = self.model(imgs)
            cls_loss, loc_loss = self.criterion(outs, annos.clone())
            loss = cls_loss + loc_loss
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            total_cls_loss += cls_loss.item()
            total_loc_loss += loc_loss.item()

            if step % self.cfg.Train.print_interval == self.cfg.Train.print_interval - 1:
                # Loss
                log_data = {'scalar': {
                    'train/total_loss': total_loss / self.cfg.Train.print_interval,
                    'train/cls_loss': total_cls_loss / self.cfg.Train.print_interval,
                    'train/loc_loss': total_loc_loss / self.cfg.Train.print_interval,
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
                total_cls_loss = 0
                total_loc_loss = 0

            if self.main_proc_flag and (
                    step % self.cfg.Train.checkpoint_interval == self.cfg.Train.checkpoint_interval - 1 or\
                    step == self.cfg.Train.iter_num - 1):
                self.save_ckp(self.model.module, step, logger.log_dir)

    def transform_bbox(self, cls_pred, loc_pred):
        """
        Transform prediction class and location into bbox coordinate.
        :param cls_pred: (AnchorN, ClassN)
        :param loc_pred: (AnchorsN, 4)
        :return: BBox, (N, 8)
        """
        cls_pred = cls_pred.sigmoid()
        cls_pred_prob, cls_pred_idx = cls_pred.max(dim=1)
        object_idx = cls_pred_prob > 0.1
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

    @staticmethod
    def save_result(file_path, pred_bbox):
        pred_bbox = torch.clamp(pred_bbox, min=0.)
        with open(file_path, 'w') as f:
            for i in range(pred_bbox.size()[0]):
                bbox = pred_bbox[i]
                line = '%d,%d,%d,%d,%.4f,%d,-1,-1\n' % (
                    int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]),
                    float(bbox[4]), int(bbox[5])
                )
                f.write(line)

    def evaluation_process(self):
        self.model.eval()

        state_dict = torch.load(self.cfg.Val.model_path)
        self.model.module.load_state_dict(state_dict)
        epoch = 0
        step = 0

        self.validation_loader.sampler.set_epoch(epoch)

        with torch.no_grad():
            for data in self.validation_loader:
                step += 1
                imgs, annos, names = data
                imgs = imgs.cuda(self.cfg.Distributed.gpu_id)

                img_size = (imgs.size()[2], imgs.size()[3])
                self.anchors, self.anchors_widths, self.anchors_heights, self.anchors_ctr_x, self.anchors_ctr_y = \
                    self.make_anchor(img_size)

                outs = self.model(imgs)
                pred_bbox = self.transform_bbox(outs[1][0], outs[0][0]).cpu()

                # NMS
                nms_bbox = pred_bbox[:, :5].detach().clone().numpy()
                nms_bbox[:, 2] = nms_bbox[:, 0] + nms_bbox[:, 2]
                nms_bbox[:, 3] = nms_bbox[:, 1] + nms_bbox[:, 3]
                keep_idx = nms(nms_bbox, thresh=0.3, gpu_id=self.cfg.Distributed.gpu_id)
                pred_bbox = pred_bbox[keep_idx]

                file_path = os.path.join(self.cfg.Val.result_dir, names[0] + '.txt')
                self.save_result(file_path, pred_bbox)

                del imgs
                del outs
                del pred_bbox
                if self.main_proc_flag:
                    print('Step : %d / %d' % (step, len(self.validation_loader)))
            print('Done !!!')

