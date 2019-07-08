import os
from datetime import datetime, timezone, timedelta
import torch
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils
from .timer import Timer


class Logger(object):
    def __init__(self, cfg, main_process_flag=True):
        """
        :param cfg: Configuration
        :param main_process_flag: if this is a logger for main process.
        """
        self.main_process_flag = main_process_flag
        if main_process_flag:
            self.log_dir = self.create_log_dir(cfg.log_prefix)
            self.tensorboard = SummaryWriter(self.log_dir) if cfg.use_tensorboard else None
        self.timer = Timer()
        self.total_iter_num = cfg.Train.iter_num
        self.init_timer(cfg.Train.iter_num)

    def init_timer(self, iter_length):
        self.total_iter_num = iter_length
        self.timer.start(iter_length)

    def stamp_timer(self, step):
        self.timer.stamp(step)

    def add_scalar(self, data, tag, n_iter):
        self.tensorboard.add_scalar(tag, float(data), n_iter)

    def add_img(self, data, tag, n_iter):
        self.tensorboard.add_image(tag, data, n_iter)

    def write_log_file(self, text):
        with open(os.path.join(self.log_dir, 'log.txt'), 'a+') as writer:
            writer.write(text+'\n')

    def log(self, data, n_iter):
        """
        Print training log to terminal and save it to the log file.
        data is a dict like: {'scalar':[], 'imgs':[]}
        :param data: data to log.
        :param n_iter: current training step.
        :return: None
        """
        if self.main_process_flag:
            log_str = "{} Iter. {}/{} | ".format(self.timer.stamp(n_iter), n_iter, self.total_iter_num)
            for k, v in data['scalar'].items():
                log_str += "{}: {:.4} ".format(k, float(v))
                self.add_scalar(float(v), tag=k, n_iter=n_iter)
            self.write_log_file(log_str)
            print(log_str)

            if 'imgs' in data:
                for k, v in data['imgs'].items():
                    vis_img = torch.cat(v, dim=0)
                    vis_img = vutils.make_grid(vis_img, normalize=True, scale_each=True)
                    self.add_img(vis_img, tag=k, n_iter=n_iter)

    def print(self, data):
        if self.main_process_flag:
            print(data)
            self.write_log_file(data)

    @staticmethod
    def create_log_dir(log_prefix):
        """
        Make log dir.
        :param log_prefix: Prefix of the log dir.
        :return: True log path.
        """
        # utc_time = datetime.utcnow()
        # tz_cn = timezone(timedelta(hours=8))
        # local_datetime = utc_time.astimezone(tz_cn)
        log_dir = os.path.join('./log/{}'.format(log_prefix))
        os.makedirs(log_dir)
        return log_dir

# if __name__ == '__main__':
#     import torch
#     iter_num = 1000
#     logger = Logger('./log', iter_num=iter_num, use_tensorboard=True)
#
#     demo_img = (torch.randn(1, 3, 200, 200), torch.randn(1, 3, 200, 200))
#
#     for i in range(iter_num):
#         loss = torch.randn(1)
#         log_data = {'scalar': {'train/loss': loss, 'train/lr': 0.01},
#                     'imgs': {'train/out': demo_img}}
#         logger.log(log_data, n_iter=i)
