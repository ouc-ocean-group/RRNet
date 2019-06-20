from operators.distributed_wrapper import DistributedWrapper
from configs.centernet_config import Config
from operators.centernet_operator import CenterNetOperator


if __name__ == '__main__':
    dis_operator = DistributedWrapper(Config, CenterNetOperator)
    dis_operator.train()
    print('Training is Done!')
