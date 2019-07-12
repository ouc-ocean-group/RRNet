import os
from operators.distributed_wrapper import DistributedWrapper

from utils.metrics.metrics import auto_evaluate_results
# ==== import your configuration here ===
from configs.rrnet_config import Config
# ==== import your model operator here ===
from operators.rrnet_operator import RRNetOperator

if __name__ == "__main__":
    model = ['ckp-59999.pth',
             'ckp-64999.pth', 'ckp-74999.pth', 'ckp-79999.pth',
             'ckp-84999.pth', 'ckp-89999.pth', 'ckp-99999.pth']
    step = 1
    for m in model:
        Config.Val.model_path = os.path.join('./log/{}'.format(Config.log_prefix), m)
        print("Start generate Txt file ...")
        dis_operator = DistributedWrapper(Config, RRNetOperator)
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
                auto_evaluate_results(result_dir, gt_dir, ctnet_min_threshold=ct, softnms_min_threshold=snms)
