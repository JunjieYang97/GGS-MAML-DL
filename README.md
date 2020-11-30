#  GGS_MAML_DL
PyTorch implementation of the regression and rank-one matrix factorization experiments for the paper:
[Multi-Step Model-Agnostic Meta-Learning: Convergence and Improved Algorithms](https://arxiv.org/pdf/2002.07836v1.pdf)

MAML and First-order MAML(FOMAML) code part is based on the [code](https://github.com/dragen1860/MAML-Pytorch)


# Platform
- python: 3.x
- Pytorch: 0.4+

# Rank-One Matrix Factorization
## File Structure
- matrix_rank_train.py: The training file of rank-one matrix factorization problem
- meta_matrix_rank.py: The meta configure file of rank-one matrix factorization problem
- linear_matrix_rank.py: linear network file


## How to train
The command example:
```python
python matrix_rank_train.py --approx_method=zero_order --approx_delta=1e-4
```

# Regression
## File Structure
- regression_train.py: The training file of regression problem
- meta_regression.py: The meta configure file of regression problem
- MLP.py: MLP network file


## How to train
The command example
e.g.
```python
python regression_train.py --approx_method=first_approx --approx_delta=1e-7
```
