from configs.centernet_config import Config
from operators.centernet_operator import CenterNetOperator


if __name__ == '__main__':
    operator = CenterNetOperator(Config)
    operator.training_process()
    print('Training is Done!')
