# DICE: Domain-attack Invariant Causal Learning for Improved Data Privacy Protection and Adversarial Robustness

Code for [KDD 2022](http://kdd.org/kdd2022/index.html/) Paper: "DICE: Domain-attack Invariant Causal Learning for Improved Data Privacy Protection and Adversarial Robustness" by Qibing Ren, Yiting Chen, Yichuan Mo, Qitian Wu and Junchi Yan.

## News

06/25/2022 - Our code is released.

## Requisite
* python = 3.9.5
* torch = 1.10.1

## What is in this repository
1. `main.py` is the main program includes loading dataset, training, and evaluation.
2. `train.py` specifies our three training modes: `causal`, `causal_poison`, and `causal_adv`, also implements the vanilla standard and adversarial training.
3. `attacks.py` is about our adversarial attack functions.
4. `model/` contains the necessary modules of our DICE model with the series of baseline backbones.
5. `poison/` is about adversarial poisoning generation and evaluation, modified based on [adversarial poisons](https://github.com/lhfowl/adversarial_poisons), which implements our poison attack with DICE.

<!-- ## What is new in the code
![#c5f015](https://via.placeholder.com/15/c5f015/000000?text=+) Instead of stopping the gradient flow of the confounding branch shown in the paper, a useful trick is to allow such flow through the backbone and update the backbone parameters together with the gradient flow of causal branch. We apply this trick to all the downstream tasks to further boost performance of DICE. -->

<!-- 6. We include the specific configurations of DICE for different downstream tasks in ``scripts/``, namely ``causal.yaml`` for attack transferability, ``causal_poison.yaml`` for data poisoning, and ``causal_adv.yaml`` for adversarial robustness. -->

## Run the Experiment
To reproduce DICE in our paper for attacks and defense, we present the script examples on CIFAR-10 below. 


### Data Privacy Protection
```bash
python main.py --cfg scripts/cifar10/causal_poison.yaml --prefix your/exp/name
```
For adversarial poisoning generation, please refer to the directory `poison/`.

### Attack Transferability
```bash
python main.py --cfg scripts/cifar10/causal_attack.yaml --prefix your/exp/name
```

### Adversarial Robustness
```bash
python main.py --cfg scripts/cifar10/causal_adv.yaml --prefix your/exp/name
```

For robustness evaluation, run the following script:

```bash
python main.py --cfg scripts/cifar10/eval.yaml --prefix your/exp/name
```

Note that in `eval.yaml`, you need to specify the model path to the variable ``PRETRAINED_PATH`` for loading model parameters. Your are welcome to try your own configurations. If you find a better yaml configuration, please let us know by raising an issue or a PR and we will update the benchmark!


## Pretrained Models

_DICE_ provides pretrained models of DICE for the three downstream tasks in paper. The model weights are available via [google drive]().

## Citing this work

```
@inproceedings{ren2022dice,
    title={DICE: Domain-attack Invariant Causal Learning for Improved Data Privacy Protection and Adversarial Robustness},
    author={Qibing Ren, Yiting Chen, Yichuan Mo, Qitian Wu and Junchi Yan},
    booktitle={KDD},
    year={2022}
}
```

## Reference code
[1]: adversarial poisons: https://github.com/lhfowl/adversarial_poisons

[2]: Bag-of-Tricks-for-AT: https://github.com/P2333/Bag-of-Tricks-for-AT
