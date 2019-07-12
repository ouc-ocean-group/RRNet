import os
from operators.distributed_wrapper import DistributedWrapper

from utils.metrics.metrics import evaluate_results
# ==== import your configuration here ===
from configs.rrnet_config import Config
# ==== import your model operator here ===
from operators.rrnet_operator import RRNetOperator

if __name__ == "__main__":
    print("Start generating Txt file ...")
    dis_operator = DistributedWrapper(Config, RRNetOperator)
    dis_operator.eval()

    print('Start Evaluating ...')
    result_dir = Config.Val.result_dir
    gt_dir = os.path.join(Config.data_root, 'val', 'annotations')
    evaluate_results(result_dir, gt_dir)
