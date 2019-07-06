from datasets import make_dataloader
from configs.centernet_config import Config
from tqdm import tqdm


loader, _ = make_dataloader(Config)


for i, batch in enumerate(tqdm(loader)):
    pass