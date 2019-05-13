import torch.optim as optim
from .base_operator import BaseOperator
from models.retinanet import RetinaNet
from datasets import make_dataloader
from utils.vis.logger import Logger


class RetinaNetOperator(BaseOperator):
    def __init__(self, cfg):
        self.cfg = cfg

        model = RetinaNet(cfg).cuda(cfg.Distributed.gpu_id)

        self.optimizer = optim.SGD(model.parameters(),
                                   lr=cfg.Train.lr, momentum=cfg.Train.momentum, weight_decay=cfg.Train.weight_decay)

        self.lr_sch = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=cfg.Train.lr_milestones, gamma=0.1)

        self.training_loader, self.validation_loader = make_dataloader(cfg)

        super(RetinaNetOperator, self).__init__(
            cfg=self.cfg, model=model, optimizer=self.optimizer, lr_sch=self.lr_sch)

        self.main_proc_flag = cfg.Distributed.gpu_id == 0

    def criterion(self, outs, labels):
        # TODO: waiting for focal loss
        return

    def training_process(self):
        if self.main_proc_flag:
            logger = Logger(self.cfg)

        self.model.train()

        total_loss = 0
        total_cls_loss = 0
        total_loc_loss = 0

        epoch = 0
        self.training_loader.sampler.set_epoch(epoch)
        training_loader = iter(self.training_loader)

        for step in range(self.cfg.Train.iter_num):
            self.lr_sch.step()
            self.optimizer.zero_grad()

            try:
                imgs, raw_labels, _ = next(training_loader)
            except StopIteration:
                epoch += 1
                self.training_loader.sampler.set_epoch(epoch)
                training_loader = iter(self.training_loader)
                imgs, raw_labels, _ = next(training_loader)

            imgs = imgs.cuda(self.cfg.Distributed.gpu_id)
            labels = raw_labels.cuda(self.cfg.Distributed.gpu_id)

            outs = self.model(imgs)
            cls_loss, loc_loss = self.criterion(outs, labels)
            loss = cls_loss + loc_loss
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            total_cls_loss += cls_loss.item()
            total_loc_loss += loc_loss.item()

            if self.main_proc_flag:
                if step % self.cfg.Train.print_interval == self.cfg.Train.print_interval - 1:
                    # Loss
                    log_data = {'scalar': {
                        'train/total_loss': total_loss / self.cfg.Train.print_interval,
                        'train/cls_loss': total_cls_loss / self.cfg.Train.print_interval,
                        'train/loc_loss': total_loc_loss / self.cfg.Train.print_interval,
                    }}

                    logger.log(log_data, step)

                    total_loss = 0
                    total_cls_loss = 0
                    total_loc_loss = 0

                if step % self.cfg.Train.checkpoint_interval == self.cfg.Train.checkpoint_interval - 1 or\
                        step == self.cfg.Train.iter_num - 1:
                    self.save_ckp(self.model.module, step, logger.log_dir)

    def evaluation_process(self):
        pass
