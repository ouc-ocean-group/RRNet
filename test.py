from configs.retinanet_config import Config
from models.retinanet import build_net

cfg=Config

net=build_net(cfg)
print(net)