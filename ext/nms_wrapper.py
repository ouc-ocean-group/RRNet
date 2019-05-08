# ----------------------------------------------------------
# Soft-NMS: Improving Object Detection With One Line of Code
# Copyright (c) University of Maryland, College Park
# Licensed under The MIT License [see LICENSE for details]
# Written by Navaneeth Bodla and Bharat Singh
# ----------------------------------------------------------

from nms.gpu_nms import gpu_nms
from nms.cpu_nms import cpu_nms, cpu_soft_nms
import numpy as np


def soft_nms(dets, sigma=0.5, Nt=0.3, threshold=0.001, method=1):

    keep = cpu_soft_nms(np.ascontiguousarray(dets, dtype=np.float32),
                        np.float32(sigma), np.float32(Nt),
                        np.float32(threshold),
                        np.uint8(method))
    return keep


# Original NMS implementation
def nms(dets, thresh):
    if dets.shape[0] == 0:
        return []
    else:
        return cpu_nms(dets, thresh)
