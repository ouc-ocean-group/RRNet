
# ==== import your configuration here ===
from configs.nas_retinanet_config import Config
# ==== import your model operator here ===
from operators.nasfpn_operator import NASFPNOperator


if __name__ == "__main__":
    operator = NASFPNOperator(Config)
    operator.training_process()
    print("Searching is Done!")
