import numpy as np
import torch
import pandas as pd
import glob
import os
import time
from ext.nms.nms_wrapper import nms, soft_nms


def bbox_iou(a, b, x1y1x2y2=True, overlap=False):
    """
    Calculate IoU and overlap between bbox group a and bbox group b.
    :param a: Tensor, m*4
    :param b: Tensor, n*4
    :param x1y1x2y2: if use (x1x2y1y2) format.
    :param overlap: If True, return overlap of a in b
    :return: IoU, m*n, (Overlap, m*n)
    """
    assert isinstance(a, torch.Tensor)
    assert isinstance(b, torch.Tensor)

    a, b = a.clone().float(), b.clone().float()
    if not x1y1x2y2:
        a[:, 2] += a[:, 0]
        a[:, 3] += a[:, 1]
        b[:, 2] += b[:, 0]
        b[:, 3] += b[:, 1]

    a_area = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    b_area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

    iw = torch.min(torch.unsqueeze(a[:, 2], dim=1), b[:, 2]) - torch.max(torch.unsqueeze(a[:, 0], 1), b[:, 0])
    ih = torch.min(torch.unsqueeze(a[:, 3], dim=1), b[:, 3]) - torch.max(torch.unsqueeze(a[:, 1], 1), b[:, 1])

    iw = torch.clamp(iw, min=0)
    ih = torch.clamp(ih, min=0)

    ua = torch.unsqueeze((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), dim=1) + b_area - iw * ih

    ua = torch.clamp(ua, min=1e-8)

    intersection = iw * ih

    IoU = intersection / ua
    if overlap:
        return IoU, intersection / a_area.unsqueeze(1)
    else:
        return IoU


def get_tp(pred, target,
           cls_tp_flags, cls_tp_confs, cls_target_count, cls_in_img_count,
           thresholds=torch.arange(0.5, 1.0, 0.05), cls_num=11):
    """
    Get the true positive flag in the prediction.
    :param pred: Tensor, m*6
    :param target: Tensor, n*6
    :param cls_tp_flags: base true positive flags of all the classes.
    :param cls_tp_confs: base true positive confidence of all the classes.
    :param cls_target_count: number of the gt bounding box of each class.
    :param cls_in_img_count: if ClassC in this image, cls_in_img_count[ClassC] += 1.
    :param thresholds: Tensor, IoU thresholds.
    :param cls_num: number of class.
    :return: cls_tp_flags, cls_tp_confs, cls_target_count, cls_in_img_count
    """

    threshold_num = thresholds.size(0)

    sort_idx = torch.sort(pred[:, 4], descending=True)[1]
    pred = pred[sort_idx, :]

    # Remove gt box in ignore region.
    ignore_idx = target[:, 5] == 0
    _, gt_overlap = bbox_iou(target[:, :4], target[:, :4], x1y1x2y2=False, overlap=True)
    if ignore_idx.sum() != 0:
        ignore_overlap = gt_overlap[:, ignore_idx].max(dim=1)[0]
        keep_idx = (ignore_overlap < 0.5) + ignore_idx
        target = target[keep_idx, :]

    # Remove prediction box in ignore region.
    ignore_idx = target[:, 5] == 0
    iou, overlap = bbox_iou(pred[:, :4], target[:, :4], x1y1x2y2=False, overlap=True)
    if ignore_idx.sum() != 0:
        ignore_overlap = overlap[:, ignore_idx].max(dim=1)[0]
        keep_idx = ignore_overlap < 0.5
        pred = pred[keep_idx, :]
        iou = iou[keep_idx, :]

    pred_cls = pred[:, 5].long()
    target_cls = target[:, 5].long()

    cls_grid = torch.meshgrid(pred_cls, target_cls)

    tp = cls_grid[0] == cls_grid[1]

    iou_flag = (iou.unsqueeze(2).repeat(1, 1, threshold_num) - thresholds) >= 0

    tp = tp.unsqueeze(2).repeat(1, 1, threshold_num) * iou_flag

    tp_iou = iou.unsqueeze(2).repeat(1, 1, threshold_num) * tp.float()

    cls_flag = torch.zeros(cls_num - 1)

    for cls in range(1, cls_num):
        cls_dt_tp_iou = tp_iou[pred_cls == cls, :, :]
        target_cls_flag = target_cls == cls
        cls_tp_iou = cls_dt_tp_iou[:, target_cls_flag, :]
        cls_flag[cls - 1] = 1 if target_cls_flag.sum() != 0 else 0
        cls_target_count[cls - 1] += cls_tp_iou.size(1)
        cls_in_img_count[cls - 1] += 1 if cls_tp_iou.size(1) != 0 else 0

        if cls_tp_iou.size(0) == 0 or cls_tp_iou.size(1) == 0:
            continue

        cls_tp_flag = torch.zeros_like(cls_tp_iou)
        for dt_i in range(cls_tp_iou.size(0)):
            dt_iou = cls_tp_iou[dt_i, :]
            max_iou, max_idx = dt_iou.max(dim=0)
            threshold_idx = max_iou.nonzero()
            if threshold_idx.size(0) != 0:
                threshold_idx = threshold_idx.squeeze().long()
                target_idx = max_idx[threshold_idx]
                cls_tp_iou[:, target_idx, threshold_idx] = 0
                cls_tp_flag[dt_i, target_idx, threshold_idx] = 1
        cls_tp_flag = cls_tp_flag.sum(1)
        cls_tp_conf = pred[pred_cls == cls, 4]
        cls_tp_flags[cls - 1] = torch.cat((cls_tp_flags[cls - 1], cls_tp_flag))
        cls_tp_confs[cls - 1] = torch.cat((cls_tp_confs[cls - 1], cls_tp_conf))

    return cls_tp_flags, cls_tp_confs, cls_target_count, cls_in_img_count


def calculate_ap_rc(cls_tp_flags, cls_tp_confs, cls_target_count, cls_in_img_count):
    """
    Calculate AP and max Recall.
    :return: Tensor, AP of all the thresholds, Tensor, max Recall.
    """
    cls_num = cls_target_count.size(0)
    threshold_num = cls_tp_flags[0].size(1)

    total_ap = torch.zeros(cls_num)
    total_rc = torch.zeros(threshold_num)
    eval_cls = cls_target_count != 0

    for cls in range(cls_num):
        if eval_cls[cls] == 0:
            continue
        cls_tp_flag = cls_tp_flags[cls]
        cls_tp_conf = cls_tp_confs[cls]

        sort_idx = torch.sort(cls_tp_conf, descending=True)[1]
        cls_tp_flag = cls_tp_flag[sort_idx, :]
        cls_tp_cumsum = cls_tp_flag.cumsum(dim=0)

        cls_prec = \
            cls_tp_cumsum / \
            torch.arange(1., cls_tp_cumsum.size(0) + 1, step=1.).unsqueeze(1).repeat(1, cls_tp_cumsum.size(1))
        cls_rec = cls_tp_cumsum / cls_target_count[cls].clamp(min=1)

        mrec = torch.cat((torch.zeros(1, cls_rec.size(1)), cls_rec, torch.ones(1, cls_rec.size(1))))
        mpre = torch.cat((torch.zeros(1, cls_prec.size(1)), cls_prec, torch.zeros(1, cls_prec.size(1))))

        for i in range(mpre.size(0) - 1, 0, -1):
            mpre[i - 1] = torch.max(mpre[i - 1], mpre[i])

        cum_idx = ((mrec[1:] - mrec[:-1]) > 0).float()
        total_ap += \
            torch.sum((mrec[1:, :] * cum_idx - mrec[:-1, :] * cum_idx) * mpre[1:, :] * cum_idx, dim=0) * \
            cls_in_img_count[cls]
        total_rc += mrec[:-1, :].max(dim=0)[0] * cls_in_img_count[cls]

    ap = total_ap / cls_in_img_count.sum()
    rc = (total_rc / cls_in_img_count.sum()).mean()
    return ap, rc


def evaluate_once(pred, target, thresholds=torch.arange(0.5, 1.0, 0.05), cls_num=11, max_det_num=500):
    """
    Evaluate AP and Recall between one prediction and one target.
    :param pred: Tensor, m*6.
    :param target: Tensor, n*6.
    :param thresholds: Tensor, IoU Thresholds.
    :param cls_num: Int, Class number.
    :param max_det_num: Int Max number of the prediction bbox.
    :return: AP Tensor and max recall.
    """
    assert isinstance(pred, torch.Tensor)
    assert isinstance(target, torch.Tensor)

    pred = pred[:max_det_num]
    threshold_num = thresholds.size(0)

    cls_target_count = torch.zeros(cls_num - 1)
    cls_in_img_count = torch.zeros(cls_num - 1)

    cls_tp_flags = [torch.zeros(0, threshold_num) for _ in range(1, cls_num)]  # Except ignore class
    cls_tp_confs = [torch.zeros(0) for _ in range(1, cls_num)]

    cls_tp_flags, cls_tp_confs, cls_target_count, cls_in_img_count = \
        get_tp(pred, target,
               cls_tp_flags, cls_tp_confs,
               cls_target_count, cls_in_img_count,
               thresholds, cls_num)
    ap, rc = calculate_ap_rc(cls_tp_flags, cls_tp_confs, cls_target_count, cls_in_img_count)
    print(ap)
    return ap, rc


def evaluate_results(pred_dir, target_dir, thresholds=torch.arange(0.5, 1.0, 0.05), cls_num=11, max_det_num=500):
    """
    Evaluate AP and Recall between many prediction files and ground truth files.
    :param pred_dir: String, prediction dir.
    :param target_dir: String target annotation dir.
    :param thresholds: Tensor, IoU Thresholds.
    :param cls_num: Int, Class number.
    :param max_det_num: Int Max number of the prediction bbox.
    :return: None
    """
    threshold_num = thresholds.size(0)
    st = time.time()
    pred_list = [x.split('/')[-1].split('.')[0] for x in glob.glob(os.path.join(pred_dir, '*.txt'))]

    cls_target_count = torch.zeros(cls_num - 1)
    cls_in_img_count = torch.zeros(cls_num - 1)

    cls_tp_flags = [torch.zeros(0, threshold_num) for _ in range(1, cls_num)]  # Except ignore class
    cls_tp_confs = [torch.zeros(0) for _ in range(1, cls_num)]

    for name in pred_list:
        pred = pd.read_csv(os.path.join(pred_dir, "{}.txt".format(name)), header=None, float_precision='high')
        target = pd.read_csv(os.path.join(target_dir, "{}.txt".format(name)), header=None, float_precision='high')
        pred = np.array(pred)
        pred[:, 2:4] += pred[:, 0:2]
        pred[:, :4] = pred[:, :4].astype(np.int).astype(np.float)
        pred[:, 2:4] -= pred[:, 0:2]
        pred = torch.from_numpy(pred).float()[:max_det_num]
        target = np.array(target)
        target = torch.from_numpy(target).float()[:max_det_num]

        cls_tp_flags, cls_tp_confs, cls_target_count, cls_in_img_count = \
            get_tp(pred, target,
                   cls_tp_flags, cls_tp_confs,
                   cls_target_count, cls_in_img_count,
                   thresholds, cls_num)
    ap, rc = calculate_ap_rc(cls_tp_flags, cls_tp_confs, cls_target_count, cls_in_img_count)

    print("Average Precision  (AP) @[ IoU=0.50:0.95] = {:.4}.".format(ap.mean().item()))
    print("Average Precision  (AP) @[ IoU=0.50     ] = {:.4}.".format(ap[0].item()))
    print("Average Precision  (AP) @[ IoU=0.75     ] = {:.4}.".format(ap[5].item()))
    print("Average Recall     (AR) @[ IoU=0.50:0.95] = {:.4}.".format(rc.item()))
    print("Cost Time: {}s".format(time.time() - st))


def auto_evaluate_results(pred_dir, target_dir, ctnet_min_threshold, softnms_min_threshold, thresholds=torch.arange(0.5, 1.0, 0.05), cls_num=11, max_det_num=500):
    """
    Evaluate AP and Recall between many prediction files and ground truth files.
    :param pred_dir: String, prediction dir.
    :param target_dir: String target annotation dir.
    :param thresholds: Tensor, IoU Thresholds.
    :param cls_num: Int, Class number.
    :param max_det_num: Int Max number of the prediction bbox.
    :return: None
    """
    threshold_num = thresholds.size(0)
    st = time.time()
    pred_list = [x.split('/')[-1].split('.')[0] for x in glob.glob(os.path.join(pred_dir, '*.txt'))]

    cls_target_count = torch.zeros(cls_num - 1)
    cls_in_img_count = torch.zeros(cls_num - 1)

    cls_tp_flags = [torch.zeros(0, threshold_num) for _ in range(1, cls_num)]  # Except ignore class
    cls_tp_confs = [torch.zeros(0) for _ in range(1, cls_num)]

    for name in pred_list:
        pred = pd.read_csv(os.path.join(pred_dir, "{}.txt".format(name)), header=None, float_precision='high')
        target = pd.read_csv(os.path.join(target_dir, "{}.txt".format(name)), header=None, float_precision='high')
        pred = np.array(pred)
        target = np.array(target)

        # fist filter
        pred = torch.from_numpy(pred[pred[:, 4] > ctnet_min_threshold]).float()
        # second filter
        _, idx = torch.sort(pred[:, 4], descending=True)
        pred = pred[idx]
        pred = _ext_nms(pred, softnms_min_threshold)
        pred[:, 2:4] += pred[:, 0:2]
        pred[:, :4] = pred[:, :4].astype(np.int).astype(np.float)
        pred[:, 2:4] -= pred[:, 0:2]
        pred = torch.from_numpy(pred).float()
        _, idx = torch.sort(pred[:, 4], descending=True)
        pred = pred[idx][:max_det_num]
        target = torch.from_numpy(target).float()[:max_det_num]

        cls_tp_flags, cls_tp_confs, cls_target_count, cls_in_img_count = \
            get_tp(pred, target,
                   cls_tp_flags, cls_tp_confs,
                   cls_target_count, cls_in_img_count,
                   thresholds, cls_num)
    ap, rc = calculate_ap_rc(cls_tp_flags, cls_tp_confs, cls_target_count, cls_in_img_count)

    print("Average Precision  (AP) @[ IoU=0.50:0.95] = {:.4}.".format(ap.mean().item()))
    print("Average Precision  (AP) @[ IoU=0.50     ] = {:.4}.".format(ap[0].item()))
    print("Average Precision  (AP) @[ IoU=0.75     ] = {:.4}.".format(ap[5].item()))
    print("Average Recall     (AR) @[ IoU=0.50:0.95] = {:.4}.".format(rc.item()))
    print("Cost Time: {}s".format(time.time() - st))


def _ext_nms(pred_bbox, threshold):
    if pred_bbox.size(0) == 0:
        return pred_bbox
    cls_unique = pred_bbox[:, 5].unique()
    keep_bboxs = []
    for cls in cls_unique:
        cls_idx = pred_bbox[:, 5] == cls
        bbox_for_nms = pred_bbox[cls_idx].detach().cpu().numpy()
        bbox_for_nms[:, 2] = bbox_for_nms[:, 0] + bbox_for_nms[:, 2]
        bbox_for_nms[:, 3] = bbox_for_nms[:, 1] + bbox_for_nms[:, 3]
        # keep_bbox = nms(bbox_for_nms, thresh=0.7, gpu_id=self.cfg.Distributed.gpu_id)
        keep_bbox = soft_nms(bbox_for_nms, Nt=0.7, threshold=threshold, method=2)
        keep_bboxs.append(keep_bbox)
    keep_bboxs = np.concatenate(keep_bboxs, axis=0)
    keep_bboxs[:, 2] = keep_bboxs[:, 2] - keep_bboxs[:, 0]
    keep_bboxs[:, 3] = keep_bboxs[:, 3] - keep_bboxs[:, 1]
    return keep_bboxs