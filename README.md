Source code for SGF. Requires PyTorch and CUDA 10+.

Create environment:
```
conda install pytorch-gpu networkx scipy scikit-learn
```

We can use same model for all 3 different frequency data.
```bash
python train_sgf.py --data cora --layer 32 --dropout 0.6 --log_period 20 --test_study
python train_sgf.py --data wisconsin --layer 32 --dropout 0.6 --log_period 20 --test_study
python train_sgf.py --data bipartite --layer 32 --dropout 0.6 --log_period 20 --test_study
```

Cora/Citeseer/Pubmed has original split.
```bash
python train_sgf.py --data cora --layer 32 --dropout 0.6 --log_period 20 --original_split --test
python train_sgf.py --data citeseer --layer 32 --dropout 0.6 --log_period 20 --original_split --test
python train_sgf.py --data pubmed --layer 32 --dropout 0.6 --log_period 20 --original_split --test
```

Note: There are 3 hyperparameter to tune in order of importance: Dropout, weight decay, and learning rate.
