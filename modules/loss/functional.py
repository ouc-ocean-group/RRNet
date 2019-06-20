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
    alpha_factor = torch.ones(cls_targets.shape).cuda() * alpha
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


def kl_loss(ori_feats, projected_feats, hms, whs, factor=0.1):
    """
    :param ori_feats: (batch x c x h x w)
    :param projected_feats: (batch x c x h x w)
    :param hms: (batch x cls x h x w)
    :param whs: (batch x 2 x h x w)
    """
    ori_feats = flat_tensor(ori_feats)
    projected_feats = flat_tensor(projected_feats)
    hms = flat_tensor(hms)
    whs = flat_tensor(whs)

    diagonals = whs[:, 0] ** 2 + whs[:, 1] ** 2

    small_feats, large_feats = [], []

    for c in range(hms.size(1)):
        center_flag = (hms[:, c] == 1)
        if center_flag.sum() == 0:
            continue
        o_feat = ori_feats[center_flag, :]
        p_feat = projected_feats[center_flag, :]
        diagonal = diagonals[center_flag]

        k = math.ceil(diagonal.size(0) * factor)

        top_v, top_idx = torch.topk(diagonal, k=k, largest=True)
        down_v, down_idx = torch.topk(diagonal, k=k, largest=False)

        small_feat = p_feat[down_idx, :]
        large_feat = o_feat[top_idx, :]
        small_feats.append(small_feat)
        large_feats.append(large_feat)

    small_feats = torch.cat(small_feats, dim=0)
    large_feats = torch.cat(large_feats, dim=0)
    loss = F.kl_div(small_feats, large_feats.detach(), reduction='batchmean')
    return loss


if __name__ == '__main__':
    ori_feats = torch.randn(2, 3, 4, 4)
    projected_feats = torch.randn(2, 3, 4, 4)
    hms = torch.zeros(2, 5, 4, 4)
    whs = torch.zeros(2, 2, 4, 4)

    hms[:, 1, :, :] = 1
    hms[1, 1, 0, 0] = 0

    whs[:, 0, 0, 0] = 5
    whs[:, 1, 0, 0] = 4

    whs[:, 0, 0, 1] = 15
    whs[:, 1, 0, 1] = 14

    whs[:, 0, 1, 0] = 25
    whs[:, 1, 1, 0] = 24

    whs[:, 0, 1, 1] = 35
    whs[:, 1, 1, 1] = 34

    loss = kl_loss(ori_feats, projected_feats, hms, whs)
    print(loss)
