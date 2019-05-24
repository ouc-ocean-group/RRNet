import torch
import torch.nn as nn
import torch.nn.functional as F


class Controller(torch.nn.Module):
    def __init__(self, cfg):
        torch.nn.Module.__init__(self)
        self.cfg = cfg

        # self.operations_num = cfg.ss_operations_num
        self.layers_num = cfg.ss_num  # 7 by default
        self.path_num = cfg.path_num  # 3 by default

        self.lstm_size = cfg.lstm_size
        self.temperature = cfg.temperature
        self.tanh_constant = cfg.controller_tanh_constant
        self.op_tanh_reduce = cfg.controller_op_tanh_reduce

        # Operation embedding and the init input embedding
        self.embedding = nn.Embedding(self.operations_num + 1, self.lstm_size)

        self.lstm = nn.LSTMCell(self.lstm_size, self.lstm_size)

        # Highway attention
        self.w_attn_1 = nn.Linear(self.lstm_size, self.lstm_size, bias=False)
        self.w_attn_2 = nn.Linear(self.lstm_size, self.lstm_size, bias=False)
        self.index_fc = nn.Linear(self.lstm_size, 1, bias=False)
        self.op_fc = nn.Linear(self.lstm_size, self.operations_num, bias=False)

        # attention
        self.reset_param()

    def reset_param(self):
        for name, param in self.named_parameters():
            nn.init.uniform_(param, -0.1, 0.1)

    def forward(self):
        arch_seq, entropy, log_prob = self.sample_arch(device=self.op_fc.weight.device)
        return arch_seq, entropy, log_prob

    def sample_arch(self, device):
        anchors = []
        anchors_w_1 = []
        arch_seq = []

        entropy = []
        log_prob = []

        # Get the last embedding as the init input embedding.
        inputs = self.embedding(torch.tensor([self.operations_num]).long().to(device))

        next_h, next_c, _, _, _, _, _, _ = self.construct_node(inputs, anchors, anchors_w_1, None, None, device)

        for layer in range(self.layers_num):
            next_h, next_c, node_index, node_log_prob, node_entropy, op_index, op_log_prob, op_entropy = \
                self.construct_node(inputs, anchors, anchors_w_1, next_c, next_h, device)
            arch_seq.append(node_index)
            arch_seq.append(op_index)
            inputs = self.embedding(op_index)

            entropy += [node_entropy, op_entropy]
            log_prob += [node_log_prob, op_log_prob]

        arch_seq = torch.cat(arch_seq)

        entropy = sum(entropy)
        log_prob = sum(log_prob)

        return arch_seq, entropy, log_prob

    def construct_node(self, index_inputs, anchors, anchors_w_1, prev_c, prev_h, device):
        if prev_c is None and prev_h is None:
            prev_c = torch.zeros(1, self.lstm_size).to(device)
            prev_h = torch.zeros(1, self.lstm_size).to(device)

            next_h, next_c = self.lstm(index_inputs, (prev_h, prev_c))

            # Return for the first input index.
            anchor = torch.zeros_like(next_h)
            anchors.append(anchor)
            anchors_w_1.append(self.w_attn_1(next_h))
            return next_h, next_c, None, None, None, None, None, None

        next_h, next_c = self.lstm(index_inputs, (prev_h, prev_c))
        # Sample index
        query = torch.stack(anchors_w_1, dim=1)
        query = query.view(-1, self.lstm_size)
        query = torch.tanh(query + self.w_attn_2(next_h))
        index_logits = self.index_fc(query).view(1, -1)

        if self.temperature is not None:
            index_logits /= self.temperature
        if self.tanh_constant is not None:
            index_logits = self.tanh_constant * torch.tanh(index_logits)

        index_prob = F.softmax(index_logits, dim=-1)

        node_index = torch.multinomial(index_prob, 1).long().view(1)

        node_log_prob = F.cross_entropy(index_logits, node_index)
        node_entropy = -torch.mean(
            torch.sum(torch.mul(F.log_softmax(index_logits, dim=-1), index_prob), dim=1)).detach()

        # Sample operation
        op_inputs = anchors[node_index].view(1, -1).requires_grad_()

        next_h, next_c = self.lstm(op_inputs, (next_h, next_c))

        op_logits = self.op_fc(next_h)
        if self.temperature is not None:
            op_logits /= self.temperature
        if self.tanh_constant is not None:
            op_tanh = self.tanh_constant / self.op_tanh_reduce
            op_logits = op_tanh * torch.tanh(op_logits)

        op_prob = F.softmax(op_logits, dim=-1)

        op_index = torch.multinomial(op_prob, 1).long().view(1)

        op_log_prob = F.cross_entropy(op_logits, op_index)
        op_entropy = -torch.mean(torch.sum(torch.mul(F.log_softmax(op_logits, dim=-1), op_prob), dim=1)).detach()

        # Insert current node embedding into the anchor list.
        anchors.append(next_h)
        anchors_w_1.append(self.w_attn_1(next_h))

        return next_h, next_c, node_index, node_log_prob, node_entropy, op_index, op_log_prob, op_entropy
