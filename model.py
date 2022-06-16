import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.nn.utils.rnn import pack_padded_sequence as pack
import torch.utils.model_zoo as model_zoo

from allennlp.modules.elmo import Elmo
from pytorch_pretrained_bert import BertModel

class Atten(nn.Module):
    def __init__(self, conf):
        super(Atten, self).__init__()
        self.conf = conf
        self.w = nn.Conv1d(self.conf.cnn_dim, self.conf.cnn_dim, 1)
        self.temper = np.power(self.conf.cnn_dim, 0.5)
        self.dropout = nn.Dropout(self.conf.attn_dropout)
        self.softmax = nn.Softmax(-1)

    def forward(self, q, v):
        q_ = self.w(q.transpose(1, 2)).transpose(1, 2)
        attn = torch.bmm(q_, v.transpose(1, 2)) / self.temper
        vr = torch.bmm(self.dropout(self.softmax(attn)), v)
        qr = torch.bmm(self.dropout(self.softmax(attn.transpose(1, 2))), q)
        vr = torch.topk(vr, k=self.conf.attn_topk, dim=1)[0]
        vr = vr.view(vr.size(0), -1)
        qr = torch.topk(qr, k=self.conf.attn_topk, dim=1)[0]
        qr = qr.view(qr.size(0), -1)
        return qr, vr, attn

class Highway(nn.Module):
    def __init__(self, size):
        super(Highway, self).__init__()
        self.highway_linear = nn.Linear(size, size)
        self.gate_linear = nn.Linear(size, size)
        self.nonlinear = nn.ReLU()

    def forward(self, input):
        gate = torch.sigmoid(self.gate_linear(input))
        m = self.nonlinear(self.highway_linear(input))
        return gate * m + (1 - gate) * input

class RNNLayer(nn.Module):
    def __init__(self, in_dim, conf):
        super(RNNLayer, self).__init__()
        self.conf = conf
        outdim = 500
        self.rnn = nn.GRU(in_dim, outdim, num_layers=1, bidirectional=True)
        self.conv = nn.Conv1d(outdim*2, in_dim, 1)
        nn.init.xavier_uniform_(self.conv.weight)
        self.conv.bias.data.fill_(0)
        self.dropout = nn.Dropout(self.conf.cnn_dropout)

    def forward(self, input, length, hidden=None):
        lens, indices = torch.sort(length, 0, True)
        maxlen = lens[0]
        outputs, hidden_t = self.rnn(pack(self.dropout(input)[indices], lens.tolist(), batch_first=True), hidden)
        outputs = unpack(outputs, batch_first=True)[0]
        _, _indices = torch.sort(indices, 0)
        outputs = outputs[_indices]
        size = outputs.size()
        outputs = F.pad(outputs.unsqueeze(0), (0,0,0,self.conf.max_sent_len-maxlen)).view(size[0],-1,size[2])
        outputs = self.conv(outputs.transpose(1, 2)).transpose(1, 2)
        return outputs + input

class CNNLayer(nn.Module):
    def __init__(self, conf, in_dim, k, res=True):
        super(CNNLayer, self).__init__()
        self.conf = conf
        self.res = res
        self.conv = nn.Conv1d(in_dim, in_dim*2, k, stride=1, padding=k//2)
        nn.init.xavier_uniform_(self.conv.weight)
        self.conv.bias.data.fill_(0)
        self.dropout = nn.Dropout(self.conf.cnn_dropout)

    def forward(self, input):
        output = self.dropout(input.transpose(1, 2))
        tmp = self.conv(output)
        if tmp.size(2) > output.size(2):
            output = tmp[:, :, 1:]
        else:
            output = tmp
        output = output.transpose(1, 2)
        a, b = torch.chunk(output, 2, dim=2)
        output = a * torch.sigmoid(b)
        if self.res:
            output = output + input
        return output

class CharLayer(nn.Module):
    def __init__(self, char_table, conf):
        super(CharLayer, self).__init__()
        self.conf = conf
        lookup, length = char_table
        self.char_embed = nn.Embedding(self.conf.char_num, self.conf.char_embed_dim, padding_idx=self.conf.char_padding_idx)
        self.lookup = nn.Embedding(lookup.size(0), lookup.size(1))
        self.lookup.weight.data.copy_(lookup)
        self.lookup.weight.requires_grad = False
        self.convs = nn.ModuleList()
        for i in range(self.conf.char_filter_num):
            self.convs.append(nn.Conv1d(
                self.conf.char_embed_dim, self.conf.char_enc_dim, self.conf.char_filter_dim[i],
                stride=1, padding=self.conf.char_filter_dim[i]//2
            ))
            nn.init.xavier_uniform_(self.convs[i].weight)
        self.nonlinear = nn.Tanh()
        self.mask = nn.Embedding(lookup.size(0), self.conf.char_hid_dim)
        self.mask.weight.data.fill_(1)
        self.mask.weight.data[0].fill_(0)
        self.mask.weight.data[1].fill_(0)
        self.mask.weight.requires_grad = False
        self.highway = Highway(self.conf.char_hid_dim)
        del lookup
        del length

    def forward(self, input):
        charseq = self.lookup(input).long().view(input.size(0)*input.size(1), -1)
        charseq = self.char_embed(charseq).transpose(1, 2)
        conv_out = []
        for i in range(self.conf.char_filter_num):
            tmp = self.nonlinear(self.convs[i](charseq))
            if tmp.size(2) > charseq.size(2):
                tmp = tmp[:, :, 1:]
            tmp = torch.topk(tmp, k=1)[0]
            conv_out.append(torch.squeeze(tmp, dim=2))
        hid = torch.cat(conv_out, dim=1)
        hid = self.highway(hid)
        hid = hid.view(input.size(0), input.size(1), -1)
        mask = self.mask(input)
        hid = hid * mask
        return hid

class ElmoLayer(nn.Module):
    def __init__(self, char_table, conf):
        super(ElmoLayer, self).__init__()
        self.conf = conf
        lookup, length = char_table
        self.lookup = nn.Embedding(lookup.size(0), lookup.size(1))
        self.lookup.weight.data.copy_(lookup)
        self.lookup.weight.requires_grad = False
        self.elmo = Elmo(
            os.path.expanduser(self.conf.elmo_options), os.path.expanduser(self.conf.elmo_weights),
            num_output_representations=2, do_layer_norm=False, dropout=self.conf.embed_dropout
        )
        for p in self.elmo.parameters():
            p.requires_grad = False
        self.w = nn.Parameter(torch.Tensor([0.5, 0.5]))
        self.gamma = nn.Parameter(torch.ones(1))
        self.conv = nn.Conv1d(1024, self.conf.elmo_dim, 1)
        nn.init.xavier_uniform_(self.conv.weight)
        self.conv.bias.data.fill_(0)

    def forward(self, input):
        charseq = self.lookup(input).long()
        with torch.no_grad():
            res = self.elmo(charseq)['elmo_representations']
        w = F.softmax(self.w, dim=0)
        res = self.gamma * (w[0] * res[0] + w[1] * res[1])
        res = self.conv(res.transpose(1, 2)).transpose(1, 2)
        return res


class BertLayer(nn.Module):
    def __init__(self, conf):
        super(BertLayer, self).__init__()
        self.conf = conf
        self.bert = BertModel.from_pretrained(os.path.expanduser(conf.bert_path))
        for p in self.bert.parameters():
            p.requires_grad = False
        self.conv = nn.Conv1d(1024, self.conf.bert_dim, 1)
        nn.init.xavier_uniform_(self.conv.weight)
        self.conv.bias.data.fill_(0)

    def forward(self, input):
        with torch.no_grad():
            res = self.bert(input, output_all_encoded_layers=False)[0][:, 1:, :]
        res = self.conv(res.transpose(1, 2)).transpose(1, 2)
        return res


class MemoryBank(nn.Module):
    def __init__(self, conf, mem_mat, use_cuda):
        super(MemoryBank, self).__init__()
        self.conf = conf
        self.use_cuda = use_cuda
        if self.conf.corpus_splitting == 3:
            self.nclass = 4
        else:
            self.nclass = 11
        sent_dim = self.conf.pair_rep_dim
        bankmem = torch.Tensor(len(mem_mat), sent_dim)
        nn.init.uniform_(bankmem, -1, 1)
        self.biaffine = self.conf.mem_biaffine
        if self.biaffine:
            ones = torch.ones((bankmem.size(0), 1))                 # for biaffine
            bankmem = torch.cat((bankmem, ones), dim=1)
            self.w = nn.Parameter(torch.Tensor(sent_dim+1, sent_dim+1))
            nn.init.uniform_(self.w, -0.1, 0.1)
        self.bankmem = nn.Parameter(bankmem)
        self.bankmem.requires_grad = False

        self.mem_mat = mem_mat
        self.sense_emb = nn.Parameter(torch.zeros((self.nclass, self.nclass)))
        for i in range(self.nclass):
                self.sense_emb.data[i, i] = 1
        self.sense_emb.requires_grad = False
        self.sensemem = nn.Parameter(self.sense_emb[self.mem_mat])
        self.sensemem.requires_grad = False

        self.softmax = nn.Softmax(-1)
        self.dropout = nn.Dropout(self.conf.bank_dropout)
        self.idx = None
        self.coeff_emb = nn.Parameter(torch.zeros((self.nclass, 1)))
        self.coeff_emb.requires_grad = False
        self.coeff = None
        # self.linear = nn.Linear(sent_dim, 300)

    def forward(self, idx, input):
        if self.idx is None:
            return input.new_full((input.size(0), self.conf.pair_rep_dim), fill_value=0), input.new_full((input.size(0), self.nclass), fill_value=0), None
        if self.biaffine:
            ones = torch.ones((input.size(0), 1))
            if self.use_cuda:
                ones = ones.cuda()
            input = torch.cat((input, ones), dim=1)
            e = torch.mm(torch.mm(input, self.w), self.bankmem[self.idx].transpose(0, 1))
        else:
            e = torch.mm(input, self.bankmem[self.idx].transpose(0,1))
            # e = torch.mm(self.linear(input), self.linear(self.bankmem[self.idx]).transpose(0,1))
        e = self.softmax(self.dropout(e))
        if self.biaffine:
            return torch.mm(e, self.bankmem[self.idx] * self.coeff)[:, :-1], torch.mm(e, self.sensemem[self.idx] * self.coeff), e
        else:
            return torch.mm(e, self.bankmem[self.idx] * self.coeff), torch.mm(e, self.sensemem[self.idx] * self.coeff), e

    def update(self, idx, repr):
        if self.biaffine:
            ones = torch.ones((repr.size(0), 1))
            if self.use_cuda:
                ones = ones.cuda()
            repr = torch.cat((repr, ones), dim=1)
        self.bankmem.data[idx] = repr.data

    def make_idx(self, idx_list):
        if len(idx_list) == 0:
            self.idx = None
        else:
            self.idx = idx_list
            self.coeff = torch.zeros_like(idx_list).float()

            self.coeff_emb.data.fill_(0)
            for idx in idx_list:
                s = self.mem_mat[idx].item()
                self.coeff_emb.data[s] += 1
            print(self.coeff_emb)
            self.coeff = 1 / self.coeff_emb[self.mem_mat[idx_list]]


class ArgEncoder(nn.Module):
    def __init__(self, conf, we_tensor, char_table=None, sub_table=None, mem_mat=None, use_cuda=False, attnvis=False):
        super(ArgEncoder, self).__init__()
        self.conf = conf
        self.attnvis = attnvis
        if self.conf.use_rnn:
            self.usecuda = use_cuda
        self.embed = nn.Embedding(we_tensor.size(0), we_tensor.size(1))
        self.embed.weight.data.copy_(we_tensor)
        # self.embed.weight.requires_grad = False
        if self.conf.need_char:
            self.charenc = CharLayer(char_table, self.conf)
        if self.conf.need_sub:
            self.charenc = CharLayer(sub_table, self.conf)
        if self.conf.need_elmo:
            self.elmo = ElmoLayer(char_table, self.conf)
        if self.conf.need_bert:
            self.bert = BertLayer(self.conf)
        self.dropout = nn.Dropout(self.conf.embed_dropout)

        self.block1 = nn.ModuleList()
        self.block2 = nn.ModuleList()
        self.attn = Atten(self.conf)
        for i in range(self.conf.cnn_layer_num):
            if self.conf.use_rnn:
                self.block1.append(RNNLayer(self.conf.cnn_dim, self.conf))
                self.block2.append(RNNLayer(self.conf.cnn_dim, self.conf))
            else:
                self.block1.append(CNNLayer(self.conf, self.conf.cnn_dim, self.conf.cnn_kernal_size[i]))
                self.block2.append(CNNLayer(self.conf, self.conf.cnn_dim, self.conf.cnn_kernal_size[i]))
        if self.conf.need_mem_bank:
            self.bank = MemoryBank(conf, mem_mat, use_cuda)
    
    def forward(self, idx, a1, a2, ba1, ba2):
        if self.conf.use_rnn:
            len1 = torch.LongTensor([torch.max(a1[i,:].nonzero())+1 for i in range(a1.size(0))])
            len2 = torch.LongTensor([torch.max(a2[i,:].nonzero())+1 for i in range(a2.size(0))])
            if self.usecuda:
                len1 = len1.cuda()
                len2 = len2.cuda()
        arg1repr = self.embed(a1)
        arg2repr = self.embed(a2)
        if self.conf.need_char or self.conf.need_sub:
            char1 = self.charenc(a1)
            char2 = self.charenc(a2)
            arg1repr = torch.cat((arg1repr, char1), dim=2)
            arg2repr = torch.cat((arg2repr, char2), dim=2)
        if self.conf.need_elmo:
            arg1repr = torch.cat((arg1repr, self.elmo(a1)), dim=2)
            arg2repr = torch.cat((arg2repr, self.elmo(a2)), dim=2)
        if self.conf.need_bert:
            arg1repr = torch.cat((arg1repr, self.bert(ba1)), dim=2)
            arg2repr = torch.cat((arg2repr, self.bert(ba2)), dim=2)
        arg1repr = self.dropout(arg1repr)
        arg2repr = self.dropout(arg2repr)
        outputs = []
        attns = []
        for i in range(self.conf.cnn_layer_num):
            if self.conf.use_rnn:
                arg1repr = self.block1[i](arg1repr, len1)
                arg2repr = self.block2[i](arg2repr, len2)
            else:
                arg1repr = self.block1[i](arg1repr)
                arg2repr = self.block2[i](arg2repr)
            outputc1, outputc2, attnw = self.attn(arg1repr, arg2repr)
            outputs.append(outputc1)
            outputs.append(outputc2)
            attns.append(attnw)
        output = torch.cat(outputs, 1)
        if self.conf.need_mem_bank:
            o2, o3, e = self.bank(idx, output)
            return output, o2, o3, e
        else:
            return output, None, None, None

class Classifier(nn.Module):
    def __init__(self, nclass, conf, is_conn=False):
        super(Classifier, self).__init__()
        self.conf = conf
        self.is_conn = is_conn          # connective classifier
        pair_rep_dim = self.conf.pair_rep_dim
        self.dropout = nn.Dropout(self.conf.clf_dropout)
        self.fc = nn.ModuleList()
        if self.conf.clf_fc_num > 0:
            self.fc.append(nn.Linear(pair_rep_dim, self.conf.clf_fc_dim))
            for i in range(self.conf.clf_fc_num - 1):
                self.fc.append(nn.Linear(self.conf.clf_fc_dim, self.conf.clf_fc_dim))
            self.nonlinear = nn.Tanh()
            lastfcdim = self.conf.clf_fc_dim
        else:
            lastfcdim = pair_rep_dim
        self.lastfc = nn.Linear(lastfcdim, nclass)
        if self.conf.need_mem_bank and (not is_conn):
            in_dim = 4 if self.conf.corpus_splitting == 3 else 11
            self.memfc = nn.Linear(in_dim, nclass)
        self._init_weight()
        self.only_mem = False

    def _init_weight(self):
        for i in range(self.conf.clf_fc_num):
            self.fc[i].bias.data.fill_(0)
            nn.init.uniform_(self.fc[i].weight, -0.01, 0.01)
        self.lastfc.bias.data.fill_(0)
        nn.init.uniform_(self.lastfc.weight, -0.01, 0.01)
        if self.conf.need_mem_bank and (not self.is_conn):
            self.memfc.bias.data.fill_(0)
            nn.init.uniform_(self.memfc.weight, -0.01, 0.01)

    def forward(self, input):
        output = input[0]
        for i in range(self.conf.clf_fc_num):
            output = self.nonlinear(self.dropout(self.fc[i](output)))
        output = self.lastfc(self.dropout(output))
        if self.conf.need_mem_bank and (not self.is_conn):
            if self.conf.use_mem_value:
                o2 = self.memfc(input[2])
                o4 = o2
            if self.conf.use_mem_key:
                o3 = input[1]
                for i in range(self.conf.clf_fc_num):
                    o3 = self.nonlinear(self.dropout(self.fc[i](o3)))
                o3 = self.lastfc(self.dropout(o3))
                o4 = o3
            if self.conf.use_mem_key and self.conf.use_mem_value:
                o4 = o2 * self.conf.mem_beta + o3 * (1 - self.conf.mem_beta)
            output = output * (1 - self.conf.mem_alpha) + o4 * self.conf.mem_alpha
        return output

class IDRCModel(nn.Module):
    def __init__(self, conf, we_tensor, char_table, sub_table, mem_mat, use_cuda):
        super(IDRCModel, self).__init__()
        self.enc = ArgEncoder(conf, we_tensor, char_table, sub_table, mem_mat, use_cuda)
        self.clf = Classifier(conf.clf_class_num, conf)

    def forward(self, idx, arg1, arg2):
        return self.clf(self.enc(idx, arg1, arg2))

def test():
    from data import Data
    from config import Config
    conf = Config()
    usecuda = True
    if usecuda:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    we = torch.load('./data/processed/ji/we.pkl')
    char_table = None
    sub_table = None
    mem_mat = None
    if conf.need_char or conf.need_elmo:
        char_table = torch.load('./data/processed/ji/char_table.pkl')
    if conf.need_sub:
        sub_table = torch.load('./data/processed/ji/sub_table.pkl')
    if conf.need_mem_bank:
        mem_mat = torch.load('./data/processed/ji/sent_bank.pkl')
    model = IDRCModel(conf, we, char_table, sub_table, mem_mat, usecuda)
    model.to(device)
    d = Data(usecuda, conf)
    for idx, a1, a2, sense, conn in d.train_loader:
        idx, a1, a2 = idx.to(device), a1.to(device), a2.to(device)
        break
    model.eval()
    out = model(idx, a1, a2)
    print(out)
    print(out.size())

if __name__ == '__main__':
    test()