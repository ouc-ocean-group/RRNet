import torch


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