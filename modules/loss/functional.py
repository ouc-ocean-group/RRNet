import torch
import torch.nn as nn
from torch.autograd import Variable


def focal_loss(inputs, targets, alpha=0.25, gamma=2.0, num_classes=12, ignore=-1):
    '''
    cal focal loss for retinanet
    :param inputs: [N,H*W*anchors,classes]
    :param targets: [N,H*W*anchors]
    :param alpha: float32
    :param gamma: float32
    :param num_classes: classes
    :param ignore: ignore index
    :return: tensor focal loss,not mean
    '''
    # reomve target where label==-1
    pos = targets != ignore
    mask = pos.unsqueeze(2).expand_as(inputs)
    # make targets inputs,and alpha in same format [N*H*W*anchors,classes]
    tar = one_hot(targets[pos].data.cpu(), num_classes)
    inp = inputs[mask].view(-1, num_classes).data.cuda()
    tar = Variable(tar).data.cuda()
    tar2 = targets[pos]
    # calc Pt
    inp2 = inp.softmax(dim=1)
    pt = torch.where(tar.eq(1), inp2, (1 - inp2))
    #  part2=(1-Pt)^gamma ,   part3=CE  , FL=alpha*part2*part3
    part2 = torch.pow((1 - pt), gamma)
    part3 = nn.CrossEntropyLoss(reduction='mean')
    part3 = part3(inp, tar2)
    loss = part2 * part3
    loss = torch.where(tar.eq(1), (alpha * loss), ((1 - alpha) * loss))
    loss = loss.mean()
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
