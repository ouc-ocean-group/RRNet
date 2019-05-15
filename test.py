from configs.retinanet_config import Config
from datasets import make_dataloader
from modules.anchor import Anchors
from utils.metrics.metrics import bbox_iou
import torch
from models.retinanet import build_net
from modules.loss.functional import focal_loss
import torch.nn.functional as F


if __name__ == '__main__':
    train_loader, val_loader = make_dataloader(Config)

    anchor_maker = Anchors()

    retina_net = build_net(Config).cuda()

    for i, batch in enumerate(train_loader):
        imgs, annos = batch
        imgs = imgs.cuda()
        annos = annos.cuda()
        outs = retina_net(imgs)

        annos[:, :, 2] += annos[:, :, 0]
        annos[:, :, 3] += annos[:, :, 1]

        anchors = anchor_maker(imgs.size()[-2:]).cuda()
        anchors_widths  = anchors[:, 2] - anchors[:, 0]
        anchors_heights = anchors[:, 3] - anchors[:, 1]
        anchors_ctr_x   = anchors[:, 0] + 0.5 * anchors_widths
        anchors_ctr_y   = anchors[:, 1] + 0.5 * anchors_heights

        cls_targets = []
        reg_targets = []
        reg_preds = []

        for n in range(Config.Train.batch_size):
            anno = annos[n]
            iou = bbox_iou(anno[:, :4], anchors)  # bbox num * anchor num  e.g., (20, 49104)
            max_iou, max_idx = torch.max(iou, dim=0)
            pos_idx = torch.ge(max_iou, 0.5)
            neg_idx = torch.lt(max_iou, 0.4)

            assigned_anno = anno[max_idx[pos_idx], :]

            # Classification loss
            pos_cls = assigned_anno[:, 5]
            cls_target = (torch.ones(anchors.size(0)) * -1).cuda()
            cls_target[pos_idx] = pos_cls
            cls_target[cls_target == 0] = -1
            cls_target[neg_idx] = 0
            cls_target = cls_target.long()
            cls_targets.append(cls_target.unsqueeze(0))

            # Regression loss

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

        cls_targets = torch.cat(cls_targets)
        cls_loss = focal_loss(outs[1], cls_targets, num_classes=Config.num_classes)

        reg_preds = torch.cat(reg_preds)
        reg_targets = torch.cat(reg_targets)
        reg_loss = F.smooth_l1_loss(reg_preds, reg_targets)

