import torch.nn as nn
import torch
import math
import numpy as np
import torch.nn.functional as F
from torch.nn.parameter import Parameter


# SGF-A in paper
class SGF(nn.Module): 
    def __init__(self, nfeat, nlayers, nhidden, nclass, dropout):
        super(SGF, self).__init__()
        self.filters = nn.ModuleList()
        for _ in range(nlayers):
            self.filters.append(GraphFilter(0.5, 0.5, "A"))
        self.fc_in = nn.Linear(nfeat, nhidden)
        self.fc_out = nn.Linear(nhidden, nclass)
        self.params1 = list(self.filters.parameters())
        self.params2 = list(self.fc_in.parameters())
        self.params2.extend(list(self.fc_out.parameters()))
        self.act_fn = nn.ReLU()
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        H_0 = self.act_fn(self.fc_in(x))
        H_l = H_0
        for i, filt in enumerate(self.filters):
            H_l = F.dropout(H_l, self.dropout, training=self.training)
            H_l = filt(H_l, adj, H_0)
        H_l = F.dropout(H_l, self.dropout, training=self.training)
        y_hat = self.fc_out(H_l)
        return F.log_softmax(y_hat, dim=1)


class SGFS(nn.Module):
    def __init__(self, nfeat, nlayers, nhidden, nclass, dropout):
        super(SGFS, self).__init__()
        self.filters = nn.ModuleList()
        for _ in range(nlayers):
            self.filters.append(GraphFilterS(0.))
        self.fc_in = nn.Linear(nfeat, nhidden)
        self.fc_out = nn.Linear(nhidden, nclass)
        self.params1 = list(self.filters.parameters())
        self.params2 = list(self.fc_in.parameters())
        self.params2.extend(list(self.fc_out.parameters()))
        self.act_fn = nn.ReLU()
        self.dropout = dropout
        self.nlayers = nlayers

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        H_0 = self.act_fn(self.fc_in(x))
        H_l = H_0
        skip_accum = H_0
        for i, filt in enumerate(self.filters):
            H_l = F.dropout(H_l, self.dropout, training=self.training)
            H_l, skip = filt(H_l, adj)
            skip_accum += skip
        H_l = F.dropout(skip_accum, self.dropout, training=self.training)
        y_hat = self.fc_out(H_l)
        return F.log_softmax(y_hat, dim=1)


class GraphFilter(nn.Module):
    def __init__(self, alpha1, alpha2, skip="A"):
        super(GraphFilter, self).__init__()
        assert skip in ["A", "B"]
        self.skip = skip
        self.alpha1 = Parameter(torch.FloatTensor([alpha1]))
        self.alpha2 = Parameter(torch.FloatTensor([alpha2]))

    def _skip_from_input(self, inp, M, x):
        H_l = torch.spmm(M, inp)
        H_l = self.alpha1 * H_l + self.alpha2 * x
        return H_l

    def _skip_to_output(self, inp, M, x):
        H_l = torch.spmm(M, inp)
        return self.alpha1 * H_l, self.alpha2 * H_l

    def forward(self, inp, adj, x):
        if self.skip == "A":
            return self._skip_from_input(inp, adj, x)
        else:
            return self._skip_to_output(inp, adj, x)


class GraphFilterS(nn.Module):
    def __init__(self, alpha, *args, **kwargs):
        super(GraphFilterS, self).__init__()
        self.alpha = Parameter(torch.FloatTensor([alpha]))

    def forward(self, inp, M):
        H_l = torch.spmm(M, inp)
        return H_l, self.alpha * H_l
