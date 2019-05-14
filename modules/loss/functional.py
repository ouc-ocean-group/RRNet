import torch
import torch.nn as nn
from torch.autograd import Variable


def focal_loss(inputs, targets, alpha=0.25, gamma=2.0, num_classes=12):
    '''
    cal focal loss for retinanet
    :param inputs: [N,H*W*anchors,classes]
    :param targets: [N,H*W*anchors]
    :param alpha: float32
    :param gamma: float32
    :param num_classes: classes
    :return: tensor focal loss,not mean
    '''
    # make targets inputs,and alpha in same format [N*H*W*anchors,classes]
    pos=targets>=0
    #num_pos = pos.data.long().sum()
    tar = one_hot(targets.data.cpu(), num_classes)
    inp = inputs.view(-1, num_classes).data.cuda()
    tar = Variable(tar).view(-1,num_classes).data.cuda()
    tar2= targets.view(-1).data.cuda()
    anchors_num=len(tar)
    # calc Pt
    inp2 = inp.sigmoid()
    pt = torch.where(tar.eq(1), inp2, (1 - inp2))
    #  part2=(1-Pt)^gamma ,   part3=CE  , FL=alpha*part2*part3
    part2 = torch.pow((1-pt),gamma)
    part3 = nn.CrossEntropyLoss(reduction='mean')
    print(inp2)
    print(tar2)
    part3= part3(inp,tar2)
    print(part3)
    focal_loss=part2*part3
    focal_loss=torch.where(tar.eq(1),(alpha*focal_loss),((1-alpha)*focal_loss))
    loss=focal_loss.sum()
    return loss



def one_hot(inputs, num_classes=12):
    '''
    change [A] to [A,num_classes]
    :param inputs: [A]
    :param num_classes: int
    :return: tensor [A,num_classes]
    '''
    y = torch.eye(num_classes)
    return y[inputs]
