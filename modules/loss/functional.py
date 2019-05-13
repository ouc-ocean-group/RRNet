import torch
import torch.nn as nn
from torch.autograd import Variable


def focal_loss(inputs, targets, gamma=2.0, num_classes=12, alpha=None):
    '''
    cal focal loss for retinanet
    :param inputs: [N,H*W*anchors,classes]
    :param targets: [N,H*W*anchors]
    :param alpha: [classes]
    :param gamma: float32
    :param num_classes: classes
    :return: tensor focal loss
    '''
    if alpha == None:
        alpha = torch.ones(num_classes).cuda()
    else:
        alpha = torch.tensor(alpha).cuda()
    # make targets inputs,and alpha in same format [N*H*W*anchors,classes]
    tar = one_hot(targets.data.cpu(), num_classes)
    inp = inputs.view(-1, num_classes).data.cuda()
    tar = Variable(tar).view(-1,num_classes).data.cuda()
    alpha = alpha.expand(len(tar), num_classes)
    anchors_num=len(tar)
    # calc Pt
    inp = inp.sigmoid()
    pt = torch.where(tar.eq(1), inp, (1 - inp))
    # part1=-alpha   ,  part2=(1-Pt)^gamma   part3=log(Pt)
    part1 = -alpha
    part2 = torch.pow((1-pt),gamma)
    part3 = pt.log()
    focal_loss=part1*part2*part3
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
