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
