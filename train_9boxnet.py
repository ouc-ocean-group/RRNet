from configs.box9_config import Config
from operators.distributed_wrapper import DistributedWrapper
from operators.box9net_operator import Box9NetOperator


if __name__ == '__main__':
    dis_operator = DistributedWrapper(Config, Box9NetOperator)
    dis_operator.train()
    print('Training is Done!')
