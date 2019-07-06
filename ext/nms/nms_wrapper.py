# ----------------------------------------------------------
# Soft-NMS: Improving Object Detection With One Line of Code
# Copyright (c) University of Maryland, College Park
# Licensed under The MIT License [see LICENSE for details]
# Written by Navaneeth Bodla and Bharat Singh
# ----------------------------------------------------------

from ext.nms.nms.gpu_nms import gpu_nms
from ext.nms.nms.cpu_nms import cpu_nms, cpu_soft_nms
import numpy as np


def soft_nms(dets, sigma=0.5, Nt=0.3, threshold=0.001, method=1):
    keep = cpu_soft_nms(np.ascontiguousarray(dets, dtype=np.float32),
                        np.float32(sigma), np.float32(Nt),
                        np.float32(threshold),
                        np.uint8(method))
    results = dets[keep]
    return results


# Original NMS implementation
def nms(dets, thresh, gpu_id=0):
    if dets.shape[0] == 0:
        return []
    else:
        if gpu_id is not None:
            keep = gpu_nms(dets[:, :5], thresh, device_id=gpu_id)
            results = dets[keep]
            return results
        keep = cpu_nms(dets[:, :5], thresh)
        results = dets[keep]
        return results


if __name__ == '__main__':
    anchor = [
        [10, 9, 20, 19, 0.5],
        [10, 10, 15, 30, 0.45],
        [10, 10, 26, 26, 0.7],

        [8, 9, 14, 16, 0.3],
        [8, 8, 15, 15, 0.1],
    ]
    keep = soft_nms(anchor, Nt=0.4, sigma=0.3,)
    print(keep)
    """
    [0, 1, 2, 3, 4]
    # 需要看论文在仔细研究一下
    """
    anchor = np.array(anchor).astype(np.float32)
    keep = nms(anchor, thresh=0.3, gpu=True)
    print(keep)
    """
    [2, 3]
    """
