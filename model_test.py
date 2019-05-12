import torch
from models.retinanet import build_net
from configs.demo_config import Config



net=build_net(Config.Model)
loc_pre,cls_pre=net(torch.zeros([1,3,224,224]))
print(loc_pre.shape)
print(cls_pre.shape)

