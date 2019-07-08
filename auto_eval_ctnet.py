import os
from operators.distributed_wrapper import DistributedWrapper

from utils.metrics.metrics import evaluate_results, auto_evaluate_results
# ==== import your configuration here ===
from configs.centernet_config import Config
# ==== import your model operator here ===
from operators.centernet_operator import CenterNetOperator

if __name__ == "__main__":
    """
        Need Check some setting
            1. Config.Val.model_path = './data/ckp-59999.pth' 
            2. Config.Val.is_eval = True  

        output:
            Start generate Txt file ...
            => Load pretrained model...
            Processing 1/548
                  ···
            Processing 548/548
            Done !!!
            Using 110.653337
            Start Eval ...
            Average Precision  (AP) @[ IoU=0.50:0.95] = 0.1594.
            Average Precision  (AP) @[ IoU=0.50     ] = 0.3193.
            Average Precision  (AP) @[ IoU=0.75     ] = 0.1431.
            Average Recall     (AR) @[ IoU=0.50:0.95] = 0.2228.
            Cost Time: 10.611038446426392s
    """
    model = ['ckp-89999.pth']
    step = 1
    for m in model:
        Config.Val.model_path = os.path.join('./log', m)
        print("Start generate Txt file ...")
        dis_operator = DistributedWrapper(Config, CenterNetOperator)
        dis_operator.eval()

        print('Start Eval ...')
        result_dir = Config.Val.result_dir
        gt_dir = os.path.join(Config.data_root, 'val', 'annotations')

        ctnet_min_thres = [0.05, 0.08, 0.10, 0.20]
        softnms_min_thres = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
        for ct in ctnet_min_thres:
            for snms in softnms_min_thres:
                print('----------------------------------------------------------------')
                print('Step %d Config :' % step)
                step += 1
                print('| Model : %s | Ctnet_Min : %f | Softnms_Min : %f' % (m, ct, snms))
                print('----------------------------------------------------------------')
                auto_evaluate_results(result_dir, gt_dir, ctnet_min_threshold=ct, softnms_min_threshold=snms)
