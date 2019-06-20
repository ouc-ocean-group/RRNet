from .drones_det import DronesDET
import torch
from torch.utils.data import DataLoader

datasets = {
    'drones_det': DronesDET
}


def make_dataloader(cfg, collate_fn=None):
    if cfg.dataset not in datasets:
        raise NotImplementedError

    train_dataset = datasets[cfg.dataset](root_dir=cfg.data_root, transforms=cfg.Train.transforms, split='train')
    val_dataset = datasets[cfg.dataset](root_dir=cfg.data_root, transforms=cfg.Val.transforms, split='val')

    collate_fn = train_dataset.collate_fn_ctnet if collate_fn is 'ctnet' else train_dataset.collate_fn
    gpu_num = torch.cuda.device_count()
    train_loader = DataLoader(train_dataset,
                              batch_size=cfg.Train.batch_size*gpu_num, num_workers=cfg.Train.num_workers,
                              pin_memory=True, collate_fn=collate_fn,
                              shuffle=True)
    val_loader = DataLoader(val_dataset,
                            batch_size=cfg.Val.batch_size*gpu_num, num_workers=cfg.Val.num_workers,
                            pin_memory=True, collate_fn=train_dataset.collate_fn,
                            shuffle=False)

    return train_loader, val_loader


def make_nas_dataloader(cfg):
    if cfg.dataset not in datasets:
        raise NotImplementedError

    train_dataset = datasets[cfg.dataset](root_dir=cfg.data_root, transforms=cfg.Train.transforms, split='train')
    val_dataset = datasets[cfg.dataset](root_dir=cfg.data_root, transforms=cfg.Val.transforms, split='val')
    half_num = int(len(train_dataset) / 2)
    # TODO: Remove Distributed Sampler
    supernet_sampler = SubsetRandomSampler(range(half_num))
    controller_sampler = SubsetRandomSampler(range(half_num, len(train_dataset)))

    supernet_loader = DataLoader(train_dataset,
                                 batch_size=cfg.Train.batch_size, num_workers=cfg.Train.num_workers,
                                 sampler=supernet_sampler,
                                 pin_memory=True, collate_fn=train_dataset.collate_fn)

    controller_loader = DataLoader(train_dataset,
                                   batch_size=cfg.Train.batch_size, num_workers=cfg.Train.num_workers,
                                   sampler=controller_sampler,
                                   pin_memory=True, collate_fn=train_dataset.collate_fn)
    val_loader = DataLoader(val_dataset,
                            batch_size=cfg.Val.batch_size, num_workers=cfg.Val.num_workers,
                            pin_memory=True, collate_fn=val_dataset.collate_fn)
    return supernet_loader, controller_loader, val_loader
