class Config:
    data_root = '/root/data/datasets/VisDrones/original'
    log_dir = './log/exp1/'

    class Train:
        crop_size = (768, 768)
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        batch_size = 8
        num_workers = 4
        lr = 0.01
        momentum = 0.9
        weight_decay = 5e-4
        iter_num = 40000

    class Val:
        batch_size = 2
        flip = True
        model_path = './log/model.pth'

    class Model:
        pretrained = True
        backbone = 'resnet50'
        fpn = 'demofpn'
        cls_detector = 'demodetector'
        reg_detector = 'demodetector'
        num_anchors = 9
        num_classes = 20