from .drones_det import DronesDET
from torch.utils.data import DataLoader, SubsetRandomSampler

datasets = {
    'drones_det': DronesDET
}


def make_dataloader(cfg):
    if cfg.dataset not in datasets:
        raise NotImplementedError

    train_dataset = datasets[cfg.dataset](root_dir=cfg.data_root, transforms=cfg.Train.transforms, split='train')
    val_dataset = datasets[cfg.dataset](root_dir=cfg.data_root, transforms=cfg.Val.transforms, split='val')

    train_loader = DataLoader(train_dataset,
                              batch_size=cfg.Train.batch_size, num_workers=cfg.Train.num_workers,
                              sampler=cfg.Train.sampler(train_dataset) if cfg.Train.sampler else None,
                              pin_memory=True, collate_fn=train_dataset.collate_fn,
                              shuffle=True if cfg.Train.sampler is None else False)
    val_loader = DataLoader(val_dataset,
                            batch_size=cfg.Val.batch_size, num_workers=cfg.Val.num_workers,
                            sampler=cfg.Val.sampler(val_dataset) if cfg.Train.sampler else None,
                            pin_memory=True, collate_fn=val_dataset.collate_fn,
                            shuffle=True if cfg.Train.sampler is None else False)
    return train_loader, val_loader


def make_nas_dataloader(cfg):
    if cfg.dataset not in datasets:
        raise NotImplementedError

    train_dataset = datasets[cfg.dataset](root_dir=cfg.data_root, transforms=cfg.Train.transforms, split='train')
    val_dataset = datasets[cfg.dataset](root_dir=cfg.data_root, transforms=cfg.Val.transforms, split='val')
    half_num = len(train_dataset)
    supernet_sampler = SubsetRandomSampler(range(half_num))
    controller_sampler = SubsetRandomSampler(range(half_num, 2 * half_num))

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
    return supernet_sampler, controller_sampler, val_loader
