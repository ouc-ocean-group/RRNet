from datasets.transforms import *
from torch.utils.data import DistributedSampler
from easydict import EasyDict as edict


# Base Config ============================================
Config = edict()
Config.seed = 219
Config.dataset = 'drones_det'
Config.data_root = './data/DronesDET'

# Training Config =========================================
Config.Train = edict()
# If use the pretrained backbone model.
Config.Train.pretrained = True

# Dataloader params.
Config.Train.batch_size = 1
Config.Train.num_workers = 4
Config.Train.sampler = None

# Transforms
Config.Train.mean = (0.485, 0.456, 0.406)
Config.Train.transforms = Compose([
    ToTensor(),
    MaskIgnore(Config.Train.mean)
])

# Validation Config =========================================
Config.Val = edict()

# Dataloader params.
Config.Val.batch_size = 1
Config.Val.num_workers = 4
Config.Val.sampler = None

# Transforms
Config.Val.mean = (0.485, 0.456, 0.406)
Config.Val.transforms = Compose([
    ToTensor(),
    MaskIgnore(Config.Train.mean)
])
