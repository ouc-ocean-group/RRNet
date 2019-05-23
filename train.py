from operators.distributed_wrapper import DistributedWrapper

# ==== import your configuration here ===
from configs.retinanet_config import Config
# ==== import your model operator here ===
from operators.retinanet_operator import RetinaNetOperator


if __name__ == "__main__":
    dis_operator = DistributedWrapper(Config, RetinaNetOperator)

    dis_operator.train()

    print("Training is Done!")
