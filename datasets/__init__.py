from .drones_det import DronesDET
from torch.utils.data import DataLoader

datasets = {
    'drones_det': DronesDET
}


def make_dataloader(cfg):
    if cfg.dataset not in datasets:
        raise NotImplementedError

    train_dataset = datasets[cfg.dataset](root_dir=cfg.root_path, transforms=cfg.Train.transforms, split='train')
    val_dataset = datasets[cfg.dataset](root_dir=cfg.root_path, transforms=cfg.Val.transforms, split='val')

    train_loader = DataLoader(train_dataset,
                              batch_size=cfg.Train.batch_size, num_workers=cfg.Train.num_workers,
                              sampler=cfg.Train.sampler, pin_memory=True)
    val_loader = DataLoader(val_dataset,
                            batch_size=cfg.Val.batch_size, num_workers=cfg.Val.num_workers,
                            sampler=cfg.Val.sampler, pin_memory=True)
    return train_loader, val_loader
