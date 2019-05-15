from datasets.transforms import *


class Config:
    dataset = 'drones_det'
    data_root = './DronesDET'
    log_prefix = 'RetinaNet'
    use_tensorboard = True
    num_classes = 12

    class Train:
        pretrained = False
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        batch_size = 4
        num_workers = 1
        lr = 0.01
        momentum = 0.9
        weight_decay = 0.0001
        iter_num = 90000
        lr_milestones = [60000, 80000]
        crop_size = (512, 512)
        transforms = Compose([
            ToTensor(),
            HorizontalFlip(),
            RandomCrop(crop_size)
        ])
        sampler = None
        print_interval = 20
        checkpoint_interval = 30000

    class Val:
        batch_size = 2
        num_workers = 4
        flip = True
        model_path = './log/model.pth'
        transforms = None
        sampler = None

    class Model:
        backbone = 'resnet50'
        fpn = 'fpn'
        cls_detector = 'retinanet_detector'
        loc_detector = 'retinanet_detector'
        num_anchors = 9

    class Distributed:
        world_size = 1
        gpu_id = -1
        rank = 0
        ngpus_per_node = 1
        dist_url = 'tcp://127.0.0.1:34567'
