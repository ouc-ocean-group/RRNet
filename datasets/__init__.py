from .drones_det import DronesDET
from torch.utils.data import DataLoader
from .dataloader import Dataloader as _Dataloader

datasets = {
    'drones_det': DronesDET
}


def make_dataloader(cfg, collate_fn=None):
    if cfg.dataset not in datasets:
        raise NotImplementedError

    train_dataset = datasets[cfg.dataset](root_dir=cfg.data_root, transforms=cfg.Train.transforms, split='train',
                                          with_road_map=cfg.Train.with_road)
    val_dataset = datasets[cfg.dataset](root_dir=cfg.data_root, transforms=cfg.Val.transforms, split='val')

    if collate_fn is 'ctnet':
        collate_fn = train_dataset.collate_fn_ctnet
    elif collate_fn is 'rrnet':
        collate_fn = train_dataset.collate_fn_ctnet
    else:
        collate_fn = train_dataset.collate_fn

    train_loader = _Dataloader(DataLoader(train_dataset,
                                          batch_size=cfg.Train.batch_size, num_workers=cfg.Train.num_workers,
                                          sampler=cfg.Train.sampler(train_dataset) if cfg.Train.sampler else None,
                                          pin_memory=True, collate_fn=collate_fn,
                                          shuffle=True if cfg.Train.sampler is None else False))
    val_loader = _Dataloader(DataLoader(val_dataset,
                                        batch_size=cfg.Val.batch_size, num_workers=cfg.Val.num_workers,
                                        sampler=cfg.Val.sampler(val_dataset) if cfg.Val.sampler else None,
                                        pin_memory=True, collate_fn=train_dataset.collate_fn,
                                        shuffle=True if cfg.Val.sampler is None else False))

    return train_loader, val_loader
