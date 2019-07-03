from backbones.resnet import resnet10, resnet50, resnet101
from backbones.hourglass import hourglass_net
from backbones.dense_hourglass import dense_hourglass_net
from backbones.hrnet import hrnetw48
from backbones.hrnetv2 import hrnetv2
# from backbones.trident import trident_res50v2, trident_res50v2_deform, trident_res101v2, trident_res101v2_deform


def get_backbone(backbone, pretrained=False, num_stacks=2):
    if backbone == 'resnet10':
        return resnet10(pretrained=pretrained)
    elif backbone == 'resnet50':
        return resnet50(pretrained=pretrained)
    elif backbone == 'resnet101':
        return resnet101(pretrained=pretrained)
    # elif backbone == 'trires50':
    #     return trident_res50v2()
    # elif backbone == 'trires50deform':
    #     return trident_res50v2_deform()
    # elif backbone == 'trires101':
    #     return trident_res101v2()
    # elif backbone == 'trires101deform':
    #     return trident_res101v2_deform()
    elif backbone == 'hourglass':
        return hourglass_net(num_stacks=num_stacks)
    elif backbone == 'dense_hourglass':
        return dense_hourglass_net(num_stacks=num_stacks)
    elif backbone == 'hrnet':
        return hrnetw48(pretrained=True)
    elif backbone == 'hrnetv2':
        return hrnetv2(pretrained=True)
    else:
        return resnet50(pretrained=pretrained)
