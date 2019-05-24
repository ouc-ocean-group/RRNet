import torch
import torch.nn as nn
import torch.nn.functional as F


class Controller(torch.nn.Module):
    def __init__(self, cfg):
        torch.nn.Module.__init__(self)
        self.cfg = cfg

        # self.operations_num = cfg.ss_operations_num
        self.layers_num = cfg.NAS.ss_num  # 7 by default
        self.path_num = cfg.NAS.path_num  # 3 by default

        self.lstm_size = cfg.NAS.lstm_size
        self.temperature = cfg.NAS.temperature
        self.tanh_constant = cfg.NAS.controller_tanh_constant

        # Operation embedding and the init input embedding
        self.embedding = [nn.Embedding(self.layers_num+1, self.lstm_size) for _ in range(self.path_num)]

        self.path_lstm = []
        self.path_w_attn_1 = []
        self.path_w_attn_2 = []
        self.path_index_fc = []
        for _ in range(self.path_num):
            self.path_lstm.append(nn.LSTMCell(self.lstm_size, self.lstm_size))
            # Highway attention
            self.path_w_attn_1.append(nn.Linear(self.lstm_size, self.lstm_size, bias=False))
            self.path_w_attn_2.append(nn.Linear(self.lstm_size, self.lstm_size, bias=False))
            self.path_index_fc.append(nn.Linear(self.lstm_size, 1, bias=False))

        self.path_lstm = nn.ModuleList(self.path_lstm)
        self.path_w_attn_1 = nn.ModuleList(self.path_w_attn_1)
        self.path_w_attn_2 = nn.ModuleList(self.path_w_attn_2)
        self.path_index_fc = nn.ModuleList(self.path_index_fc)
        self.reset_param()

    def reset_param(self):
        for name, param in self.named_parameters():
            nn.init.uniform_(param, -0.1, 0.1)

    def forward(self):
        path_seq, layer_seq, entropy, log_prob = self.sample_arch(device=self.path_index_fc[0].weight.device)
        return path_seq, layer_seq, entropy, log_prob

    def sample_arch(self, device):
        anchors_w_1 = []
        arch_seq = []
        entropy = []
        log_prob = []

        next_c = [torch.zeros(1, self.lstm_size, device=device) for _ in range(self.path_num)]
        next_h = [torch.zeros(1, self.lstm_size, device=device) for _ in range(self.path_num)]

        # Make input node.
        for path_idx in range(self.path_num):
            inputs = self.embedding[path_idx](torch.tensor([self.layers_num], dtype=torch.long, device=device))
            next_c[path_idx], next_h[path_idx], _, _, _, anchor_w_1 = \
                self.construct_node(inputs, None, next_c[path_idx], next_h[path_idx], path_idx)
            anchors_w_1.append(anchor_w_1)

        # Make rest layer nodes.
        for l in range(self.layers_num):
            pre_query = torch.stack(anchors_w_1, dim=1).view(-1, self.lstm_size)
            cur_anchors_w_1 = []
            for path_idx in range(self.path_num):
                if l == 0:
                    inputs = self.embedding[path_idx](torch.tensor([self.layers_num], dtype=torch.long, device=device))
                else:
                    layer_index = node_index // self.path_num
                    path_index = node_index % self.path_num
                    inputs = self.embedding[path_index](torch.tensor([layer_index], dtype=torch.long, device=device))
                next_c[path_idx], next_h[path_idx], node_index, node_log_prob, node_entropy, anchor_w_1 = \
                    self.construct_node(inputs, pre_query, next_c[path_idx], next_h[path_idx], path_idx)
                arch_seq.append(node_index)
                entropy.append(node_entropy)
                log_prob.append(node_log_prob)
                cur_anchors_w_1.append(anchor_w_1)
            anchors_w_1 += cur_anchors_w_1

        arch_seq = torch.cat(arch_seq)
        entropy = sum(entropy)
        log_prob = sum(log_prob)

        path_seq = arch_seq % self.path_num
        layer_seq = arch_seq // self.path_num

        return path_seq, layer_seq, entropy, log_prob

    def construct_node(self, index_inputs, pre_query, prev_c, prev_h, path_idx):
        next_h, next_c = self.path_lstm[path_idx](index_inputs, (prev_h, prev_c))
        anchor = self.path_w_attn_2[path_idx](next_h)
        anchor_w_1 = self.path_w_attn_1[path_idx](next_h)
        if pre_query is None:
            return next_c, next_h, None, None, None, anchor_w_1

        query = torch.tanh(pre_query + anchor)
        index_logits = self.path_index_fc[path_idx](query).view(1, -1)

        if self.temperature is not None:
            index_logits /= self.temperature
        if self.tanh_constant is not None:
            index_logits = self.tanh_constant * torch.tanh(index_logits)

        index_prob = F.softmax(index_logits, dim=-1)
        node_index = torch.multinomial(index_prob, 1).long().view(1)

        node_log_prob = F.cross_entropy(index_logits, node_index)
        node_entropy = -torch.mean(
            torch.sum(torch.mul(F.log_softmax(index_logits, dim=-1), index_prob), dim=1)).detach()

        return next_c, next_h, node_index, node_log_prob, node_entropy, anchor_w_1


if __name__ == '__main__':
    from configs.nas_retinanet_config import Config

    ctl = Controller(Config)

    p_seq, l_seq, ent, prb = ctl.forward()


