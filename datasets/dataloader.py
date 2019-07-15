import torch


class Dataloader(object):
    """
    New dataloader class, which fit the iter num instead of the epoch.
    """
    def __init__(self, loader, epoch=0):
        self._loader = loader
        self.epoch = epoch
        self._loader.sampler.set_epoch(self.epoch)

        self.loader = iter(self._loader)

    @staticmethod
    def to_device(data, device='cuda'):
        _data = []
        for item in data:
            if isinstance(item, torch.Tensor):
                _data.append(item.to(device))
            else:
                _data.append(item)
        return _data

    def get_batch(self, device='cuda'):
        try:
            data = next(self.loader)
        except StopIteration:
            self.epoch += 1
            self._loader.sampler.set_epoch(self.epoch)
            self.loader = iter(self._loader)
            data = next(self.loader)

        return self.to_device(data, device)
