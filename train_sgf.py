import time
import random
import argparse
import numpy as np
import pickle as pkl
import torch
import torch.nn.functional as F
import torch.optim as optim
from model import SGF
import uuid
from collections import Counter
from sklearn.metrics import confusion_matrix
from rayleigh import rayleigh_quotient, rayleigh_sub
from utils import load_data, accuracy, get_coeff

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--data', default='cora', help='Dataset')
parser.add_argument('--seed', type=int, default=0, help='Random seed')
parser.add_argument('--epochs', type=int, default=1500, help='Max epoch.')
parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
parser.add_argument('--wd', type=float, default=0.0005, help='Weight decay')
parser.add_argument('--layer', type=int, default=16, help='Number of layers')
parser.add_argument('--hidden', type=int, default=64, help='Hidden dimension')
parser.add_argument('--dropout', type=float, default=0.7, help='Dropout rate')
parser.add_argument('--patience', type=int, default=100, help='Patience')
parser.add_argument('--cuda', action="store_true", default=False, help='Train on CPU or GPU')
parser.add_argument('--gpu_id', type=int, default=0, help='GPU device id')
parser.add_argument('--test', action='store_true', default=False, help='evaluation on test set.')
parser.add_argument('--test_study', action='store_true', default=False, help='print info on the test result.')
parser.add_argument("--log_period", type=int, default=50, help="Log every x epochs")
parser.add_argument("--split", type=str, default="0.6_0.2_0.2")
parser.add_argument("--use_laplacian", action="store_true", default=False)
args = parser.parse_args()
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

# Load data
data_package = load_data(args.data, ["AugNormAdj", "SymNormLap"], 
                         split=args.split, rs=args.seed)
_, normed_adjs, features, labels, idx_train, idx_val, idx_test = data_package
adj, L = normed_adjs
if args.use_laplacian:
    adj = L 

### This is only used to print rayleigh loss, not for training!
### The training process strictly use only idx_train
est_rayleigh = rayleigh_sub(labels, L, torch.cat((idx_train, idx_val)), 
                            one_hot=True) 
######

# Setup device
if args.cuda:
    cudaid = "cuda:"+str(args.gpu_id)
else:
    cudaid = "cpu"
device = torch.device(cudaid)
features = features.to(device)
adj = adj.to(device)
L = L.to(device)
checkpt_file = 'checkpoints/'+uuid.uuid4().hex[:4]+'-'+args.data+'.pt'
poly_file = 'checkpoints/'+uuid.uuid4().hex[:4]+'-'+args.data+'.poly'
print(cudaid, checkpt_file)
if args.test_study:
    print(poly_file)

model = SGF(nfeat=features.shape[1],
            nlayers=args.layer,
            nhidden=args.hidden,
            nclass=int(labels.max()) + 1,
            dropout=args.dropout).to(device)
opt_config = [
    {'params': model.params1, 'weight_decay': args.wd, 'lr': args.lr},
    {'params': model.params2, 'weight_decay': args.wd, 'lr': args.lr/4},
]
optimizer = optim.Adam(opt_config)

def train():
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    acc_train = accuracy(output[idx_train], labels[idx_train].to(device))
    loss_train = F.nll_loss(output[idx_train], labels[idx_train].to(device))
    loss_train.backward()
    optimizer.step()
    return loss_train.item(), acc_train.item()

def validate():
    model.eval()
    with torch.no_grad():
        output = model(features, adj)
        loss_val = F.nll_loss(output[idx_val], labels[idx_val].to(device))
        acc_val = accuracy(output[idx_val], labels[idx_val].to(device))
        return loss_val.item(), acc_val.item()

def test():
    model.load_state_dict(torch.load(checkpt_file))
    model.eval()
    with torch.no_grad():
        output = model(features, adj)
        loss_test = F.nll_loss(output[idx_test], labels[idx_test].to(device))
        acc_test = accuracy(output[idx_test], labels[idx_test].to(device))
        return loss_test.item(), acc_test.item()

def filt_eval():
    model.eval()
    with torch.no_grad():
        out = model(features, adj)
        out_r = rayleigh_quotient(out, L)
        loss_r = F.l1_loss(out_r, est_rayleigh.to(device))
    return loss_r.item()

def test_study():
    model.load_state_dict(torch.load(checkpt_file))
    model.eval()
    with torch.no_grad():
        output = model(features, adj)
        loss_test = F.nll_loss(output[idx_test], labels[idx_test].to(device))
        preds = output[idx_test].max(1)[1].type_as(labels[idx_test])
        correct = preds.eq(labels[idx_test]).double()
        correct_count = correct.sum()
        acc_test = correct_count / len(idx_test)

        wrong_labels = (1+labels[idx_test]) * (1-correct)
        wrong_labels = Counter(wrong_labels.cpu().numpy().astype(int))
        test_labels = Counter(labels[idx_test].cpu().numpy().astype(int))
       
        for k, v in test_labels.items():
            print(k, 1 - wrong_labels[k+1]/v)

        alphas = [f.alpha1.item() for f in model.filters]
        betas = [f.alpha2.item() for f in model.filters]
        poly_coeffs = get_coeff(alphas, betas)
        print(poly_coeffs)
        with open(poly_file, "wb") as f:
            pkl.dump(poly_coeffs, f)

        print(confusion_matrix(labels[idx_test], preds))

        return loss_test.item(), acc_test.item()
    
t_total = time.time()
c = 0
best = 7e7
best_epoch = 0
acc = 0

for epoch in range(args.epochs):
    # _ = filt_train()
    loss_tra, acc_tra = train()
    loss_val, acc_val = validate()
    loss_filt = filt_eval()
    if (epoch+1)%args.log_period == 0 or epoch == 0: 
        print('Epoch:{:04d}'.format(epoch+1),
            'loss:{:.3f}'.format(loss_tra),
            'acc:{:.2f}'.format(acc_tra*100),
            ' | ',
            'loss:{:.3f}'.format(loss_val),
            'acc:{:.2f}'.format(acc_val*100),
            ' | ', 
            'f_loss:{:.3f}'.format(loss_filt))
    if loss_val < best:
        best = loss_val
        best_epoch = epoch
        acc = acc_val
        torch.save(model.state_dict(), checkpt_file)
        c = 0
    else:
        c += 1

    if c == args.patience:
        break

if args.test:
    acc = test()[1]

if args.test_study:
    acc = test_study()[1]

print("Train cost: {:.4f}s".format(time.time() - t_total))
print('Load {}th epoch'.format(best_epoch))
print("Test" if args.test or args.test_study else "Val","acc.:{:.1f}".format(acc*100)) 
