from datasets.transforms import *
from torch.utils.data import DistributedSampler
from easydict import EasyDict as edict


# Base Config ============================================
Config = edict()
Config.seed = 219
Config.dataset = 'drones_det'
Config.data_root = './data/DronesDET'
Config.log_prefix = 'TwoStageNet'
Config.use_tensorboard = True
Config.num_classes = 10

# Training Config =========================================
Config.Train = edict()
# If use the pretrained backbone model.
Config.Train.pretrained = True

# Dataloader params.
Config.Train.batch_size = 4
Config.Train.num_workers = 4
Config.Train.sampler = DistributedSampler

# Optimizer params.
Config.Train.lr = 2.5e-4
Config.Train.momentum = 0.9
Config.Train.weight_decay = 0.0001
# Milestones for changing learning rage.
Config.Train.lr_milestones = [60000, 80000]

Config.Train.iter_num = 100000

# Transforms
Config.Train.crop_size = (512, 512)
Config.Train.mean = (0.485, 0.456, 0.406)
Config.Train.std = (0.229, 0.224, 0.225)
Config.Train.scale_factor = 4
Config.Train.with_road = True
Config.Train.transforms = Compose([
    MultiScale(scale=(1, 1.15, 1.25, 1.35, 1.5)),
    ToTensor(),
    MaskIgnore(Config.Train.mean),
    FillDuck(),
    HorizontalFlip(),
    RandomCrop(Config.Train.crop_size),
    Normalize(Config.Train.mean, Config.Train.std),
    ToHeatmap(scale_factor=Config.Train.scale_factor)
])

# Log params.
Config.Train.print_interval = 20
Config.Train.checkpoint_interval = 5000


# Validation Config =========================================
Config.Val = edict()
Config.Val.model_path = './log/{}/ckp-89999.pth'.format(Config.log_prefix)
Config.Val.is_eval = True
Config.Val.auto_test = True
# Dataloader params.
Config.Val.batch_size = 1
Config.Val.num_workers = 4
Config.Val.sampler = DistributedSampler

# Transforms
Config.Val.mean = (0.485, 0.456, 0.406)
Config.Val.std = (0.229, 0.224, 0.225)
Config.Val.scales = [1, 1.1, 1.2, 1.3, 1.4, 1.5]
Config.Val.transforms = Compose([
    ToTensor(),
    Normalize(Config.Val.mean, Config.Val.std)
])
Config.Val.result_dir = './results/'


# Model Config ===============================================
Config.Model = edict()

Config.Model.backbone = 'hourglass'
Config.Model.num_stacks = 2
Config.Model.nms_type_for_stage1 = 'nms'  # or 'soft_nms'
Config.Model.nms_per_class_for_stage1 = True

# Distributed Config =========================================
Config.Distributed = edict()
Config.Distributed.world_size = 1
Config.Distributed.gpu_id = -1
Config.Distributed.rank = 0
Config.Distributed.ngpus_per_node = 1
Config.Distributed.dist_url = 'tcp://127.0.0.1:34564'
