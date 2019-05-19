import torch
import torch.multiprocessing as mp
import torch.distributed as dist


class DistributedWrapper(object):
    def __init__(self, cfg, operator_class):
        """
        This is a wrapper class for distributed training.
        :param cfg: configuration.
        :param operator_class: We use this class to construct the operator for training and evaluating.
        """
        self.cfg = cfg
        self.operator_class = operator_class

    def setup_distributed_params(self):
        """
        Setup the world size and ngpus per node.
        :return:
        """
        try:
            ngpus_per_node = torch.cuda.device_count()
            self.cfg.Distributed.ngpus_per_node = ngpus_per_node
            self.cfg.Distributed.world_size = ngpus_per_node * self.cfg.Distributed.world_size
        except ValueError:
            raise ValueError('[x] Can not get gpu numbers!')

    def init_operator(self, gpu, ngpus_per_node, cfg):
        """
        Create distributed model operator.
        :param gpu: gpu id.
        :param ngpus_per_node: to calculate the real rank.
        :param cfg: configuration.
        :return: model operator.
        """
        cfg.Distributed.gpu_id = gpu
        print("=> Use GPU: {}".format(gpu))

        # I. Init distributed process group.
        cfg.Distributed.rank = cfg.Distributed.rank * ngpus_per_node + gpu
        dist.init_process_group(backend='nccl', init_method=cfg.Distributed.dist_url,
                                world_size=cfg.Distributed.world_size, rank=cfg.Distributed.rank)
        torch.cuda.set_device(gpu)
        # II. Init operator.
        return self.operator_class(cfg)

    def train(self):
        """
        Start multiprocessing training.
        """
        self.setup_distributed_params()
        mp.spawn(self.dist_training_process, nprocs=self.cfg.Distributed.ngpus_per_node,
                 args=(self.cfg.Distributed.ngpus_per_node, self.cfg))

    def eval(self):
        """
        Start multiprocessing evaluating.
        """
        self.setup_distributed_params()
        mp.spawn(self.dist_evaluation_process, nprocs=self.cfg.Distributed.ngpus_per_node,
                 args=(self.cfg.Distributed.ngpus_per_node, self.cfg))

    def dist_training_process(self, gpu, ngpus_per_node, cfg):
        operator = self.init_operator(gpu, ngpus_per_node, cfg)
        operator.training_process()

    def dist_evaluation_process(self, gpu, ngpus_per_node, cfg):
        operator = self.init_operator(gpu, ngpus_per_node, cfg)
        operator.evaluation_process()
