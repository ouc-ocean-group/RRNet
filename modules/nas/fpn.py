import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNRelu(nn.Module):
    def __init__(self, inplane, plane, kernel_size=3, padding=1):
        super(ConvBNRelu, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(inplane, plane, kernel_size, 1, padding),
            nn.BatchNorm2d(plane),
            nn.ReLU()
        )

    def forward(self, x):
        return self.layer(x)


class SwapCell(nn.Module):
    def __init__(self, layer_idx, path_idx, inplane=256, swap_flag=True):
        super(SwapCell, self).__init__()
        self.layer_idx = layer_idx
        self.path_idx = path_idx
        self.swap_layer = ConvBNRelu(inplane, inplane) if swap_flag else None

    def forward(self, x1, x2):
        x1_path_idx = x1['path_idx']
        x1_layer_idx = x1['layer_idx']
        x1_feat = x1['feat']

        x2_path_idx = x2['path_idx']
        x2_layer_idx = x2['layer_idx']
        x2_feat = x2['feat']

        if (x1_path_idx == x2_path_idx and x1_layer_idx == x2_layer_idx) or self.swap_layer is None:
            x = x1_feat
        else:
            x2_feat = F.interpolate(x2_feat, size=x1_feat.size()[-2:], mode='nearest')
            x = self.swap_layer(x1_feat + x2_feat)
        return {'feat': x, 'path_idx': self.path_idx, 'layer_idx': self.layer_idx}


class NASSuperFPN(nn.Module):
    def __init__(self, cfg):
        super(NASSuperFPN, self).__init__()
        self.cfg = cfg
        self.layers_num = self.cfg.NAS.ss_num  # 7 by default
        self.path_num = self.cfg.NAS.path_num  # 3 by default
        self.inplane = self.cfg.NAS.fpn_inplane
        self.plane = self.cfg.NAS.fpn_plane  # 256 by default
        self.super_fpn = self.make_super_fpn()
        self.input_layers = nn.ModuleList(
            [ConvBNRelu(self.inplane[i], self.plane, kernel_size=1, padding=0) for i in range(self.path_num)]
        )

    def make_super_fpn(self):
        paths = []
        for path_idx in range(1, self.path_num+1):
            layers = []
            for layer_idx in range(1, self.layers_num+1):
                layers.append(SwapCell(layer_idx, path_idx, inplane=self.plane))
            layers = nn.ModuleList(layers)
            paths.append(layers)
        super_fpn = nn.ModuleList(paths)
        return super_fpn

    def forward(self, xs, p_seq, l_seq):
        """
        :param xs:
        :param p_seq: tensor([1, 1, 1, 1, 2, 1, 0, 1, 0, 2, 1, 0, 1, 2, 1, 2, 0, 1, 1, 0, 0])
        :param l_seq: tensor([0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 2, 0, 4, 1, 4, 0, 1, 0, 4, 2, 3])
        :return:
        """
        all_xs = [[{'feat': self.input_layers[i](xs[i]), 'path_idx': -1, 'layer_idx': -1} for i in range(self.path_num)]]
        i = 0
        for cur_layer_idx in range(self.layers_num):
            cur_xs = []
            for cur_path_idx in range(self.path_num):
                other_layer_idx = l_seq[i]
                other_path_idx = p_seq[i]
                cur_xs.append(
                    self.super_fpn[cur_path_idx][cur_layer_idx](
                        all_xs[-1][cur_path_idx], all_xs[other_layer_idx][other_path_idx])
                )
            all_xs.append(cur_xs)
        outs = []
        for xs in all_xs[-1]:
            outs.append(xs['feat'])
        return outs


class NASSearchedFPN(nn.Module):
    def __init__(self, cfg):
        super(NASSearchedFPN, self).__init__()
        self.cfg = cfg
        self.layers_num = self.cfg.NAS.ss_num  # 7 by default
        self.path_num = self.cfg.NAS.path_num  # 3 by default
        self.inplane = self.cfg.NAS.fpn_inplane
        self.plane = self.cfg.NAS.fpn_plane  # 256 by default

        self.p_seq = cfg.NAS.p_seq
        self.l_seq = cfg.NAS.l_seq

        self.searched_fpn = self.make_searched_fpn()
        self.input_layers = nn.ModuleList(
            [ConvBNRelu(self.inplane[i], self.plane, kernel_size=1, padding=0) for i in range(self.path_num)]
        )

    def make_searched_fpn(self):
        paths = []
        idx = 0
        for path_idx in range(1, self.path_num+1):
            layers = []
            for layer_idx in range(1, self.layers_num+1):
                other_p = self.p_seq[idx]
                other_l = self.l_seq[idx]
                idx += 1
                swap_flag = path_idx - 1 == other_p and layer_idx - 1 == other_l
                layers.append(SwapCell(layer_idx, path_idx, inplane=self.plane, swap_flag=swap_flag))
            layers = nn.ModuleList(layers)
            paths.append(layers)
        searched_fpn = nn.ModuleList(paths)
        return searched_fpn

    def forward(self, xs):
        """
        :param xs:
        :return:
        """
        all_xs = [[{'feat': self.input_layers[i](xs[i]), 'path_idx': -1, 'layer_idx': -1} for i in range(self.path_num)]]
        i = 0
        for cur_layer_idx in range(self.layers_num):
            cur_xs = []
            for cur_path_idx in range(self.path_num):
                other_layer_idx = self.l_seq[i]
                other_path_idx = self.p_seq[i]
                cur_xs.append(
                    self.searched_fpn[cur_path_idx][cur_layer_idx](
                        all_xs[-1][cur_path_idx], all_xs[other_layer_idx][other_path_idx])
                )
            all_xs.append(cur_xs)
        outs = []
        for xs in all_xs[-1]:
            outs.append(xs['feat'])
        return outs


if __name__ == '__main__':
    # from configs.nas_retinanet_config import Config
    # nassuperfpn = NASSuperFPN(Config)
    # p_seq = torch.tensor([1, 1, 1, 1, 2, 1, 0, 1, 0, 2, 1, 0, 1, 2, 1, 2, 0, 1, 1, 0, 0])
    # l_seq = torch.tensor([0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 2, 0, 4, 1, 4, 0, 1, 0, 4, 2, 3])
    # feats = [torch.randn(1, 2**(i+5), 2**i, 2**i) for i in range(4, 7)]
    #
    # outs = nassuperfpn(feats, p_seq, l_seq)

    # from configs.retinanet_config import Config
    #
    # nassearchedfpn = NASSearchedFPN(Config)
    #
    # outs = nassearchedfpn(feats)
