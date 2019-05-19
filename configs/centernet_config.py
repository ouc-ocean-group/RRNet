from datasets.transforms import *
from torch.utils.data import DistributedSampler


class Config:
    dataset = 'drones_det'
    data_root = './DronesDET'
    log_prefix = 'CenterNet'
    use_tensorboard = True
    # index 0 is not for the ignore region. It is the background or negative region.
    num_classes = 12

    class Train:
        # If use the pretrained backbone model.
        pretrained = True

        # Dataloader params.
        batch_size = 8
        num_workers = 2
        sampler = DistributedSampler

        # Optimizer params.
        lr = 0.01
        momentum = 0.9
        weight_decay = 0.0001
        # Milestones for changing learning rage.
        lr_milestones = [60000, 80000]

        iter_num = 90000

        # Transforms
        crop_size = (512, 512)
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        transforms = Compose([
            ToTensor(),
            HorizontalFlip(),
            RandomCrop(crop_size),
            Normalize(mean, std)
        ])

        # Log params.
        print_interval = 20
        checkpoint_interval = 30000

    class Val:
        model_path = './log/model.pth'

        # Dataloader params.
        batch_size = 2
        num_workers = 4
        sampler = DistributedSampler

        # Transforms
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        transforms = Compose([
            ToTensor(),
            Normalize(mean, std)
        ])

    class Model:
        backbone = 'hourglass'
        hm_detector = 'centernet_detector'
        wh_detector = 'centernet_detector'
        reg_detector = 'centernet_detector'
        num_stacks = 2

    class Distributed:
        world_size = 1
        gpu_id = -1
        rank = 0
        ngpus_per_node = 1
        dist_url = 'tcp://127.0.0.1:34567'
