Source code for SGF. Requires PyTorch and CUDA 10+.

Create environment:
```
conda install pytorch-gpu networkx scipy scikit-learn
```

We can use same model for all 3 different frequency data. Remove the `--cuda` flag to train on CPU, it will be about x10 slower.
```bash
python train_sgf.py --data cora --test_study --cuda
python train_sgf.py --data wisconsin --test_study --cuda
python train_sgf.py --data bipartite --test_study --cuda
```

Cora/Citeseer/Pubmed has original split.
```bash
python train_sgf.py --data cora --original_split --test --cuda
python train_sgf.py --data citeseer --original_split --test --cuda
python train_sgf.py --data pubmed --original_split --test --cuda
```

Note: There are 3 hyperparameter: Dropout, weight decay, learning rate, number of filter layers.
