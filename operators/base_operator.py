import os
import torch
from torch.nn.parallel import DistributedDataParallel
import random
import torch.optim as optim


class BaseOperator(object):
    def __init__(self, cfg, model, optimizer=None, lr_sch=None):
        self.cfg = cfg

        random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        torch.cuda.manual_seed(cfg.seed)

        self.model = DistributedDataParallel(model, device_ids=[self.cfg.distributed.gpu_id])

        self.optimizer = optim.SGD(self.model.parameters(),
                                   lr=cfg.train.lr, momentum=cfg.train.momentum,
                                   weight_decay=cfg.train.weight_decay) if optimizer is None else optimizer
        self.lr_sch = lr_sch

    def criterion(self, outs, labels):
        raise NotImplementedError

    def training_process(self):
        raise NotImplementedError

    def evaluation_process(self):
        raise NotImplementedError

    @staticmethod
    def save_ckp(models, step, path):
        torch.save(models.state_dict(), os.path.join(path, 'ckp-{}.pth'.format(step)))
