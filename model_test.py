from models.retinanet import build_net
import torch
from  configs.demo_config import Config
from torch.autograd import Variable
import numpy as np
from modules.loss.functional import focal_loss

y1=[[[0.1,0.2,0.3,1.2],[0.3,0.1,0.4,0.2]],[[1.2,0.2,0.3,0.1],[0.3,0.2,0.1,0.5]],[[1.2,0.2,0.3,0.1],[0.3,0.2,0.1,0.5]]]
y=[[1,3],[0,2],[0,2]]
cls=[1,1,1,0.5]
y1=torch.tensor(y1).cuda()

y=torch.LongTensor(y).cuda()

fl=focal_loss(y1,y,2.0,4,cls)
print(fl)

'''
print(y.shape)
pos=y>-1

mask=pos.unsqueeze(2).expand_as(y1)
y1=y1[mask].view(-1,4)
x=torch.eye(4)
ytemp=y[pos]
print(ytemp.shape)
ytemp=ytemp.data.cpu()
x=x[ytemp]
y=Variable(x).cuda()

print(y1.shape)
print(y.shape)
q2=y1.sigmoid()
print(1-q2)
q1=y.eq(1)
print(q1)
result=torch.where(q1,q2,(1-q2))
print(result)
'''
