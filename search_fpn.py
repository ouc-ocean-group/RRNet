from operators.distributed_wrapper import DistributedWrapper

# ==== import your configuration here ===
from configs.nas_retinanet_config import Config
# ==== import your model operator here ===
from operators.nasfpn_operator import NASFPNOperator


if __name__ == "__main__":
    dis_operator = DistributedWrapper(Config, NASFPNOperator)
    dis_operator.train()
    print("Searching is Done!")
