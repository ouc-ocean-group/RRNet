import os
import zipfile
import shutil
from datetime import datetime
import torch
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils
from timer import Timer


class Logger(object):
    def __init__(self, log_dir, iter_num, use_tensorboard=True):
        self.log_dir = self.create_log_dir(log_dir)
        self.tensorboard = SummaryWriter(self.log_dir) if use_tensorboard else None
        self.timer = Timer()
        self.total_iter_num = iter_num
        self.init_timer(iter_num)

    def init_timer(self, iter_length):
        self.timer.start(iter_length)

    def add_scalar(self, data, tag, n_iter):
        self.tensorboard.add_scalar(tag, float(data), n_iter)

    def add_img(self, data, tag, n_iter):
        self.tensorboard.add_image(tag, data, n_iter)

    def write_log_file(self, text):
        with open(os.path.join(self.log_dir, 'log.txt'), 'a+') as writer:
            writer.write(text)

    def log(self, data, n_iter):
        """
        Print training log to terminal and save it to the log file.
        data is a dict like: {'scalar':[], 'imgs':[]}
        :param data: data to log.
        :param n_iter: current training step.
        :return: None
        """
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

    def create_log_dir(self, log_dir):
        """
        Make log dir.
        :param log_dir: Path
        :return: True log path.
        """
        if log_dir is not None:
            if os.path.isdir(log_dir):
                self.zip_dir(log_dir)
                shutil.rmtree(log_dir)
        else:
            log_dir = os.path.join('./log/{}'.format(datetime.now().isoformat(timespec='seconds')))
        os.makedirs(log_dir)
        return log_dir

    @staticmethod
    def zip_dir(dir_path):
        tmp_path = dir_path.split('/')
        out_file_name = os.path.join(
            *tmp_path[:-1], tmp_path[-1] + '-{}.zip'.format(datetime.now().isoformat(timespec='seconds')))
        zip_archive = zipfile.ZipFile(out_file_name, "w", zipfile.ZIP_DEFLATED)
        for p, d, fs in os.walk(dir_path):
            fpath = p.replace(dir_path, '')
            for f in fs:
                zip_archive.write(os.path.join(p, f),
                          os.path.join(fpath, f))
        zip_archive.close()


if __name__ == '__main__':
    import torch
    iter_num = 1000
    logger = Logger('./log', iter_num=iter_num, use_tensorboard=True)

    demo_img = (torch.randn(1, 3, 200, 200), torch.randn(1, 3, 200, 200))

    for i in range(iter_num):
        loss = torch.randn(1)
        log_data = {'scalar': {'train/loss': loss, 'train/lr': 0.01},
                    'imgs': {'train/out': demo_img}}
        logger.log(log_data, n_iter=i)
