# Test CIFAR-10 poisons
Code is adapted and modified from Liu Kuang's PyTorch CIFAR [repo](https://github.com/kuangliu/pytorch-cifar)

## Prerequisites
- Python 3.6+
- PyTorch 1.0+

## How to run the experiment
### Evaluate your poisons under Standard Training (ST) with the following command:

```
python main.py --load_path /path/to/your/saved/poisons --runs number_of_runs
```
### Evaluate your poisons under Adversarial Training (AT) with the following command:
```
python main_adv.py --step 7 --epsilon 2 --load_path /path/to/your/saved/poisons --runs number_of_runs
```
Other defense baselines are also included for thorough evaluation such as Gaussian Smoothing (_main_gaussian_smoothing.py_), [DPSGD](https://arxiv.org/abs/1607.00133) (_main_DPSGD.py_) and [mixup](https://arxiv.org/abs/1710.09412?context=cs) (_main_mixup.py_).