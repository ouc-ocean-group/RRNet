from .drones_det import DronesDET
from torch.utils.data import DataLoader

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
                              pin_memory=True, collate_fn=train_dataset.collate_fn)
    val_loader = DataLoader(val_dataset,
                            batch_size=cfg.Val.batch_size, num_workers=cfg.Val.num_workers,
                            sampler=cfg.Val.sampler(val_dataset) if cfg.Train.sampler else None,
                            pin_memory=True, collate_fn=val_dataset.collate_fn)
    return train_loader, val_loader
