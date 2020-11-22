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


class ChebNet(nn.Module):
    def __init__(self, nfeat, nlayers, nhidden, nclass, dropout, init_t=0.5):
        super(ChebNet, self).__init__()
        assert nlayers >= 2, "Need at least order 2 Chebyshev."
        self.filters = nn.ModuleList()
        for _ in range(nlayers):
            self.filters.append(ChebLayer(init_t))
        self.fc1 = nn.Linear(nfeat, nhidden)
        self.fc2 = nn.Linear(nhidden, nclass)
        self.params1 = list(self.filters.parameters())
        self.params2 = list(self.fc1.parameters())
        self.params2.extend(list(self.fc2.parameters()))
        self.act_fn = nn.ReLU()
        self.dropout = dropout
        self.nlayers = nlayers
            
    def forward(self, x, L):
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.act_fn(self.fc1(x))
        T_0, poly = self.filters[0](x, None)
        T_1, term = self.filters[1](T_0, None, L)
        poly += term
        prevs = [T_0, T_1]
        for i, filt in enumerate(self.filters[2:]):
            T_i, term = filt(prevs[0], prevs[1], L)
            prevs[1] = prevs[0]
            prevs[0] = T_i
            poly += term
        poly = F.dropout(poly, self.dropout, training=self.training)
        y_hat = self.fc2(poly)
        return F.log_softmax(y_hat, dim=1)
        
class ChebLayer(nn.Module):
    def __init__(self, theta):
        super(ChebLayer, self).__init__()
        self.theta = Parameter(torch.FloatTensor([theta]))
    
    def forward(self, T_n_1, T_n_2, M=None):
        if M is not None and T_n_2 is not None:
            H_l = 2 * torch.spmm(M, T_n_1) 
            H_l = H_l - T_n_2
        elif M is not None and T_n_2 is None:
            H_l = torch.spmm(M, T_n_1) 
        else:
            H_l = T_n_1
        return H_l, self.theta * H_l
        

class GraphFilter(nn.Module):
    def __init__(self, alpha1, alpha2, skip="A"):
        super(GraphFilter, self).__init__()
        assert skip in ["A", "B"]
        self.skip = skip
        self.alpha1 = Parameter(torch.FloatTensor([alpha1]))
        self.alpha2 = Parameter(torch.FloatTensor([alpha2]))
        torch.nn.init.uniform_(self.alpha1, -1, 1)
        torch.nn.init.uniform_(self.alpha2, -1, 1)

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
