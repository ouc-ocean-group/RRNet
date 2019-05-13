from datasets.transforms import *


class Config:
    dataset = 'drones_det'
    data_root = './VisDrone'
    log_prefix = 'RetinaNet'
    use_tensorboard = True

    class Train:
        crop_size = (768, 768)
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        batch_size = 8
        num_workers = 4
        lr = 0.01
        momentum = 0.9
        weight_decay = 0.0001
        iter_num = 90000
        lr_milestones = [60000, 80000]
        transforms = Compose([
            ToTensor(),
            HorizontalFlip(),
            RandomCrop((800, 800))
        ])
        print_interval = 20
        checkpoint_interval = 30000

    class Val:
        batch_size = 2
        flip = True
        model_path = './log/model.pth'

    class Model:
        pretrained = True
        backbone = 'resnet50'
        fpn = 'fpn'
        cls_detector = 'retinanet_detector'
        loc_detector = 'retinanet_detector'
        num_anchors = 9
        num_classes = 20

    class Distributed:
        world_size = 1
        gpu_id = -1
        rank = 0
        ngpus_per_node = 1
        dist_url = 'tcp://127.0.0.1:34567'
