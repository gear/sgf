Source code for SGF. Uses PyTorch.

![](./misc/sgfa.png)

Create environment:
```
conda install pytorch-gpu networkx scipy scikit-learn
```
or this if GPU is not available
```
conda install pytorch networkx scipy scikit-learn
```

We can use same model for all 3 different frequency data. Remove the `--cuda` flag to train on CPU, it will be about x10 slower. The default split is 0.6/0.2/0.2. Train with skip-from-input version (SGF-A):
```bash
python train_sgf.py --data cora --test_study --cuda
python train_sgf.py --data wisconsin --test_study --cuda
python train_sgf.py --data bipartite --test_study --cuda
```
or train with skip-to-output version (SGF-B):
```bash
python b_train_sgf.py --data cora --test_study --cuda
python b_train_sgf.py --data wisconsin --test_study --cuda
python b_train_sgf.py --data bipartite --test_study --cuda
```

Cora/Citeseer/Pubmed has original split (similar to what used in the GCN paper).
```bash
python train_sgf.py --data cora --original_split --test --cuda
python train_sgf.py --data citeseer --original_split --test --cuda
python train_sgf.py --data pubmed --original_split --test --cuda
```

Note: There are 3 hyperparameter: Dropout, weight decay, learning rate, number of filter layers.

We pack the data using `pickle` in the following format:
```
dname.graph: networkx graph
dname.X: numpy feature matrix shape=(num_nodes, feature_dim)
dname.y: numpy label matrix shape=(num_nodes,)
dname.split (optional): dictionary of splits
```
