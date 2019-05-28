import torch
import torch.optim as optim
from models.nas_retinanet import NASRetinaNet
from modules.nas.controller import Controller
from datasets import make_nas_dataloader
from utils.vis.logger import Logger
from modules.loss.focalloss import FocalLoss
from modules.anchor import Anchors
from utils.metrics.metrics import bbox_iou


class NASFPNOperator(object):
    def __init__(self, cfg):
        self.cfg = cfg

        self.supernet = NASRetinaNet(cfg).cuda()
        self.controller = Controller(cfg).cuda()

        self.supernet_optimizer = optim.Adam(self.supernet.parameters(), lr=cfg.Train.lr)
        self.controller_optimizer = optim.Adam(self.controller.parameters(), lr=cfg.Train.lr)

        self.supernet_loader, self.controller_loader, self.validation_loader = make_nas_dataloader(cfg)

        self.anchor_maker = Anchors(sizes=(16, 64, 128))

        self.anchors, self.anchors_widths, self.anchors_heights, self.anchors_ctr_x, self.anchors_ctr_y = \
            self.make_anchor(cfg.Train.crop_size)

        self.focal_loss = FocalLoss()

        self.main_proc_flag = True

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
            cls_target[pos_idx, assigned_anno[:, 5].long()-1] = 1
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
        baseline = None
        for epoch in range(self.cfg.NAS.epoch):
            self.supernet.train()
            self.controller.eval()

            total_sn_loss = 0
            logger.print("=> Epoch: {}".format(epoch))
            logger.print("=> Training SuperNet...")
            # Train super net.
            logger.init_timer(len(self.supernet_loader))
            for step, (imgs, annos, _) in enumerate(self.supernet_loader):
                self.supernet_optimizer.zero_grad()
                imgs = imgs.cuda()
                annos = annos.cuda()

                p_seq, l_seq, _, _ = self.controller()

                outs = self.supernet(imgs, p_seq, l_seq)
                cls_loss, loc_loss = self.criterion(outs, annos.clone())
                loss = cls_loss + loc_loss
                loss.backward()
                self.supernet_optimizer.step()

                total_sn_loss += loss.item()

                if step % self.cfg.Train.print_interval == self.cfg.Train.print_interval - 1:
                    log_str = "{} {}/{} | SNLoss: {:.4}".format(
                        logger.stamp_timer(step), step, len(self.supernet_loader),
                        total_sn_loss/self.cfg.Train.print_interval)
                    logger.print(log_str)
                    total_sn_loss = 0

            logger.print("=> Training Controller...")
            self.supernet.eval()
            self.controller.train()

            total_ctl_loss = 0
            total_ctl_reward = 0

            controller_loader = iter(self.controller_loader)
            logger.init_timer(300)
            for step in range(300):
                try:
                    imgs, annos, _ = next(controller_loader)
                except:
                    controller_loader = iter(self.controller_loader)
                    imgs, annos, _ = next(controller_loader)
                imgs, annos = imgs.cuda(), annos.cuda()

                self.controller_optimizer.zero_grad()

                p_seq, l_seq, entropy, log_prob = self.controller()

                with torch.no_grad():
                    outs = self.supernet(imgs, p_seq, l_seq)
                    cls_loss, loc_loss = self.criterion(outs, annos.clone())
                    reward = (-1 * (cls_loss + loc_loss)).exp()

                if self.cfg.NAS.entropy_weight is not None:
                    reward += self.cfg.NAS.entropy_weight * entropy

                if baseline is None:
                    baseline = reward
                baseline -= (1 - self.cfg.NAS.baseline_decrease) * (baseline.item() - reward.item())

                loss = log_prob * (reward.detach() - baseline)
                loss = loss.sum()

                loss.backward()
                self.controller_optimizer.step()

                total_ctl_loss += loss.item()
                total_ctl_reward += reward.item()

                if step % self.cfg.Train.print_interval == self.cfg.Train.print_interval - 1:
                    log_str = "{} {}/{} | CTLLoss: {:.4}, CTLReward: {:.4}".format(
                        logger.stamp_timer(step), step, len(self.supernet_loader),
                        total_ctl_loss / self.cfg.Train.print_interval,
                        total_ctl_reward / self.cfg.Train.print_interval
                    )
                    logger.print(log_str)
                    total_ctl_loss, total_ctl_reward = 0, 0

            logger.print("=> Validation...")
            self.supernet.eval()
            self.controller.eval()
            validation_loader = iter(self.validation_loader)
            for i in range(10):
                try:
                    imgs, annos, _ = next(validation_loader)
                except:
                    validation_loader = iter(self.validation_loader)
                    imgs, annos, _ = next(validation_loader)
                imgs, annos = imgs.cuda(), annos.cuda()

                self.controller_optimizer.zero_grad()

                p_seq, l_seq, entropy, log_prob = self.controller()

                with torch.no_grad():
                    outs = self.supernet(imgs, p_seq, l_seq)
                    cls_loss, loc_loss = self.criterion(outs, annos.clone())
                    reward = (-1 * (cls_loss + loc_loss)).exp()

                logger.print("   PArch: {}".format(str(p_seq.cpu().numpy())))
                logger.print("   LArch: {}".format(str(l_seq.cpu().numpy())))
                logger.print("   Reward: {:.6}".format(float(reward)))
                logger.print("-----------------------------------------------")

