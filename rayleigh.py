import torch
import torch.nn.functional as F

def rayleigh_quotient_i(X, M, i=0):
    r = X[:, i].unsqueeze(0) @ torch.spmm(M, X[:, i].unsqueeze(1))
    d = torch.norm(X[:, index], p=2).item()**2
    if d == 0:
        return 0
    return r / d

def rayleigh_quotient(X, M):
    r = torch.sum(X * (torch.spmm(M, X)), dim=0)
    d = torch.norm(X, dim=0, p=2)**2
    r = r / d
    r[r == float("Inf")] = 0
    return r

def qform(X, M, i=0):
    r = X[:, i].unsqueeze(0) @ (M @ X[:, i].unsqueeze(1))
    return r

def rayleigh_sub(X, M, idx, one_hot=False):
    with torch.no_grad():
        subM = M.to_dense()[idx][:,idx]
        if one_hot:
            subX = F.one_hot(X[idx], num_classes=X.max()+1).float() - 0.5
        else:
            subX = X[idx]
        subM_diag = torch.diag(torch.diag(subM))
        q = len(idx) / M.size(0)
        est = []
        for i in range(subX.size(1)):
            r = qform(subX, subM, i=i) - (1-q) * qform(subX, subM_diag, i=i)
            r /= (0.25* M.size(0))
            est.append(r.item() / q**2)
        return torch.clamp(torch.FloatTensor(est), 0, 2)
