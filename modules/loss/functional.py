import torch
import torch.nn.functional as F
import math


def focal_loss(cls_preds, cls_targets, gamma=2.0, alpha=0.75):
    """
    cal focal loss for retinanet
    :param cls_preds: [N,H*W*anchors,classes]
    :param cls_targets: [N,H*W*anchors]
    :param alpha: float32
    :param gamma: float32
    :return: tensor focal loss,not mean
    """
    cls_preds = torch.sigmoid(cls_preds).clamp(1e-7, 1. - 1e-7)
    alpha_factor = torch.ones(cls_targets.shape, device=cls_preds.device) * alpha
    alpha_factor = torch.where(torch.eq(cls_targets, 1.), alpha_factor, 1. - alpha_factor)
    focal_weight = torch.where(torch.eq(cls_targets, 1.), 1. - cls_preds, cls_preds)
    focal_weight = alpha_factor * torch.pow(focal_weight, gamma)
    bce = -(cls_targets * torch.log(cls_preds) + (1.0 - cls_targets) * torch.log(1.0 - cls_preds))
    cls_loss = focal_weight * bce
    return cls_loss.sum()


def focal_loss_for_hm(pred, gt):
    """
    Modified focal loss. Exactly the same as CornerNet.
        Runs faster and costs a little bit more memory
    :param pred: (batch x c x h x w)
    :param gt: (batch x c x h x w)
    """

    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()

    neg_weights = torch.pow(1 - gt, 4)

    loss = 0

    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

    num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos
    return loss


def flat_tensor(tensor):
    c = tensor.size(1)
    tensor = tensor.permute(0, 2, 3, 1).contiguous().view(-1, c)
    return tensor


def kl_loss(ori_feats, projected_feats, hms, whs, inds, factor=0.1):
    """
    :param ori_feats: (batch x c x h x w)
    :param projected_feats: (batch x c x h x w)
    :param hms: (batch x cls x h x w)
    :param whs: (batch x n x 2)
    :param inds: (batch x n x 1)
    """
    ori_feats = flat_tensor(ori_feats)
    projected_feats = flat_tensor(projected_feats)
    whs = whs.view(-1, 2)
    bias = torch.arange(0, hms.size(0), dtype=torch.float, device=ori_feats.device).unsqueeze(1).unsqueeze(1) * \
           (hms.size(2) * hms.size(3))
    pos_inds = inds.permute(2, 0, 1).contiguous().view(-1).long() > 0
    inds = inds + bias
    inds = inds.permute(2, 0, 1).contiguous().view(-1).long()
    inds = inds[pos_inds]
    cls = ((hms == 1).float() * torch.arange(0, hms.size(1), dtype=torch.float, device=ori_feats.device).unsqueeze(0).unsqueeze(2).unsqueeze(2)).sum(dim=1, keepdim=True)
    cls = flat_tensor(cls)
    diagonals = (whs[:, 0] ** 2 + whs[:, 1] ** 2)[pos_inds]
    cls = cls[inds, :].squeeze()

    small_idx, large_idx = [], []

    for c in range(hms.size(1)):
        cls_flag = (cls == c)
        if cls_flag.sum() == 0:
            continue
        diagonal = diagonals[cls_flag]

        k = math.ceil(diagonal.size(0) * factor)

        top_v, top_idx = torch.topk(diagonal, k=k, largest=True)
        down_v, down_idx = torch.topk(diagonal, k=k, largest=False)

        small_i = inds[cls_flag][down_idx]
        large_i = inds[cls_flag][top_idx]
        small_idx.append(small_i)
        large_idx.append(large_i)
    small_idx = torch.cat(small_idx)
    large_idx = torch.cat(large_idx)

    small_alpha = projected_feats[small_idx, :]
    large_alpha = projected_feats[large_idx, :].detach()
    small_feats = ori_feats[small_idx, :]
    large_feats = ori_feats[large_idx, :].detach()
    loss = 0.5 * (small_alpha - large_alpha) + (large_alpha.exp() + F.smooth_l1_loss(small_feats, large_feats, reduction='none')) / (2 * small_alpha.exp())
    loss = loss.mean()
    return loss


def giou_loss(bbox, reg, gt, scale_factor):
    bbox[:, :4] = bbox[:, :4] * scale_factor
    s1_xywh = bbox[:, :4]
    s1_xywh[:, 2:4] -= s1_xywh[:, 0:2]

    s2_xywh = s1_xywh.detach()
    s2_xywh[:, 2:4] += 1
    out_ctr_x = reg[:, 0] * s2_xywh[:, 2] + s2_xywh[:, 0] + s2_xywh[:, 2] / 2
    out_ctr_y = reg[:, 1] * s2_xywh[:, 3] + s2_xywh[:, 1] + s2_xywh[:, 3] / 2
    out_w = reg[:, 2].exp() * s2_xywh[:, 2]
    out_h = reg[:, 3].exp() * s2_xywh[:, 3]
    out_x1 = out_ctr_x - out_w / 2.
    out_y1 = out_ctr_y - out_h / 2.
    out_x2 = out_ctr_x + out_w / 2.
    out_y2 = out_ctr_y + out_h / 2.

    s2_bboxes = torch.stack((out_x1, out_y1, out_x2, out_y2), dim=1)

    return _giou_loss(s2_bboxes, gt)


def _giou_loss(output, target):
    x1, y1, x2, y2 = output[:, 0], output[:, 1], output[:, 2], output[:, 3]
    x1g, y1g, x2g, y2g = target[:, 0], target[:, 1], target[:, 2], target[:, 3]

    x2 = torch.max(x1, x2)
    y2 = torch.max(y1, y2)

    xkis1 = torch.max(x1, x1g)
    ykis1 = torch.max(y1, y1g)
    xkis2 = torch.min(x2, x2g)
    ykis2 = torch.min(y2, y2g)

    xc1 = torch.min(x1, x1g)
    yc1 = torch.min(y1, y1g)
    xc2 = torch.max(x2, x2g)
    yc2 = torch.max(y2, y2g)

    intsctk = torch.zeros(x1.size(), device=output.device)
    mask = (ykis2 > ykis1) * (xkis2 > xkis1)
    intsctk[mask] = (xkis2[mask] - xkis1[mask]) * (ykis2[mask] - ykis1[mask])
    unionk = (x2 - x1) * (y2 - y1) + (x2g - x1g) * (y2g - y1g) - intsctk + 1e-7
    iouk = intsctk / unionk

    area_c = (xc2 - xc1) * (yc2 - yc1) + 1e-7
    miouk = iouk - ((area_c - unionk) / area_c)
    # iouk = (1 - iouk).mean(0)
    miouk = (1 - miouk).mean(0)

    return miouk