from datasets import make_dataloader
from configs.kmeans_config import Config
from tqdm import tqdm
import torch
from ext.kmeans.kmeans import lloyd


train_loader, val_loader = make_dataloader(Config)

all_w = []
all_h = []
all_d = []

with torch.no_grad():
    for i, batch in enumerate(tqdm(train_loader)):
        annos = batch[1][0]
        # d = (annos[:, 2].pow(2) + annos[:, 3].pow(2)).sqrt()
        all_w.append(annos[:, 3].clone())
        all_h.append(annos[:, 2].clone())

all_w = torch.cat(all_w).unsqueeze(1)
all_h = torch.cat(all_h).unsqueeze(1)

h_results = lloyd(all_h, 3)
print(h_results)  # 20.3807, 73.2261, 182.68274

w_results = lloyd(all_w, 3)
print(w_results)  # 21.9839, 63.8345, 155.8799
