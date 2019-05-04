class Config(object):
    def __init__(self):

        # training set
        self.ignore_label = 255
        self.CSroot = '/root/data/datasets/cityscapes'
        self.ADE20Kroot = '/root/data/datasets/ADEChallengeData2016'
        self.crop_size = (768, 768)
        self.mean = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)
        self.brightness = 0.5
        self.contrast = 0.5
        self.saturation = 0.5
        self.scales = (0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0)

        self.train_batch_size = 8
        self.train_gpu = [0, 1]
        self.lr = 0.01
        self.momentum = 0.9
        self.weight_decay = 5e-4
        self.train_iter_num = 41000
        self.epoch = 121

        self.ohem_thresh = 0.7
        self.lr_start = 1e-2
        self.warmup_steps = 1000
        self.warmup_start_lr = 5e-6
        self.lr_power = 0.9
        self.train_dataset = 'cityscapes'

        # trainval setting

        #Cityscapes
        self.n_classes = 19
        #ADE20K
        # self.n_classes = 150
        self.resnet101_path = '/root/data/models/zhangyu/resnet101-imagenet.pth'


        # valadation setting
        self.val_gpu = [0, 1]
        self.val_dataset = 'cityscapes'
        self.eval_batch_size = 2
        self.eval_scales = (0.5, 0.75, 1.0, 1.25, 1.5, 1.75)
        self.eval_flip = True
        self.model_path = '/root/data/models/zhangyu/dv3p16xcityscapes_41000.pth'


        self.log_dir = '/root/data/models/zhangyu/FFASPPlog/'


config = Config()

