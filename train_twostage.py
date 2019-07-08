from configs.twostage_config import Config
from operators.distributed_wrapper import DistributedWrapper
from operators.twostage_operator import TwoStageOperator


if __name__ == '__main__':
    dis_operator = DistributedWrapper(Config, TwoStageOperator)
    dis_operator.train()
    print('Training is Done!')
