import random
import numpy as np
import scipy.sparse as sp
import torch
import sys
import os
import pickle as pkl
import networkx as nx
from normalization import fetch_normalization, row_normalize
from time import perf_counter
from torch.utils import data
from sklearn.model_selection import train_test_split


dataf = os.path.expanduser("{}/data/".format(os.path.dirname(__file__)))

def get_coeff(alphas, betas, lap=True):
    K = len(alphas)
    if not lap:
        return -1
    else:
        coeffs = []
        for i in range(K):
            c = np.prod([alphas[j] for j in range(i,K)])
            if i > 0:
                c *= betas[i-1]
            coeffs.append(c)
        coeffs.append(betas[-1])
    return coeffs

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def load_data(dataset_str="cora",
              normalization=[],
              feat_normalize=True,
              cuda=False,
              split="0.6",
              rs=0,
              **kwargs):
    """
    Load pickle packed datasets.
    """
    with open(dataf+dataset_str+".graph", "rb") as f:
        graph = pkl.load(f)
    with open(dataf+dataset_str+".X", "rb") as f:
        X = pkl.load(f)
    with open(dataf+dataset_str+".y", "rb") as f:
        y = pkl.load(f)
    if split == "0.6":
        with open(dataf+dataset_str+".split", "rb") as f:
            split_index = rs % 10
            split = pkl.load(f)
            idx_train = split['train'][split_index]
            idx_test = split['test'][split_index]
            idx_val = split['valid'][split_index]
    elif split == "original":
        with open(dataf+dataset_str+".split", "rb") as f:
            split = pkl.load(f)
            idx_train = split['train']
            idx_test = split['test']
            idx_val = split['valid']
    else:
        tr_size, va_size, te_size = [float(i) for i in split.split("_")]
        idx_train, idx_val, idx_test = \
            train_val_test_split(np.arange(len(y)), train_size=tr_size,
                                 val_size=va_size, test_size=te_size,
                                 stratify=y, random_state=rs) 

    normed_adj = []
    if len(normalization) > 0:
        adj = nx.adj_matrix(graph)
        for n in normalization:
            nf = fetch_normalization(n, **kwargs)
            normed_adj.append(nf(adj))

    if feat_normalize:
        X = row_normalize(X)

    X = torch.FloatTensor(X).float()
    y = torch.LongTensor(y)
    normed_adj = [sparse_mx_to_torch_sparse_tensor(adj).float() \
                  for adj in normed_adj]
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    if cuda:
        X = X.cuda()
        normed_adj = [adj.cuda() for adj in normed_adj]
        y = y.cuda()
        idx_train = idx_train.cuda()
        idx_val = idx_val.cuda()
        idx_test = idx_test.cuda()

    return graph, normed_adj, X, y, idx_train, idx_val, idx_test

def train_val_test_split(*arrays, 
                         train_size=0.5, 
                         val_size=0.3, 
                         test_size=0.2, 
                         stratify=None, 
                         random_state=None):

    if len(set(array.shape[0] for array in arrays)) != 1:
        raise ValueError("Arrays must have equal first dimension.")
    idx = np.arange(arrays[0].shape[0])
    idx_train_and_val, idx_test = train_test_split(idx,
                                                   random_state=random_state,
                                                   train_size=(train_size + val_size),
                                                   test_size=test_size,
                                                   stratify=stratify)
    if stratify is not None:
        stratify = stratify[idx_train_and_val]
    idx_train, idx_val = train_test_split(idx_train_and_val,
                                          random_state=random_state,
                                          train_size=(train_size / (train_size + val_size)),
                                          test_size=(val_size / (train_size + val_size)),
                                          stratify=stratify)
    result = []
    for X in arrays:
        result.append(X[idx_train])
        result.append(X[idx_val])
        result.append(X[idx_test])
    return result

def sgc_precompute(features, adj, degree):
    t = perf_counter()
    for i in range(degree):
        features = torch.spmm(adj, features)
    precompute_time = perf_counter()-t
    return features, precompute_time
