from configs.rrnet_config import Config
from operators.distributed_wrapper import DistributedWrapper
from operators.rrnet_operator import RRNetOperator


if __name__ == '__main__':
    dis_operator = DistributedWrapper(Config, RRNetOperator)
    dis_operator.train()
    print('Training is Done!')
