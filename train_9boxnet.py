from configs.box9_config import Config
from datasets import make_dataloader


if __name__ == '__main__':
    train_loader, _ = make_dataloader(Config, collate_fn='9boxnet')

    for i, data in enumerate(train_loader):
        imgs, annos, hms, whs, inds, offsets, reg_masks, names = data
