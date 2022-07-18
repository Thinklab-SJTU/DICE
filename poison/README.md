# Adversarial poison generation and evaluation with our causal model.

We use and adapt the framework from the [adversarial_poisons](https://github.com/lhfowl/adversarial_poisons) and implement our poison attack with causal model.

The codes are mainly in ```village```, where ```clients``` contains the code for model, ```material``` contains the code for dataset and ```shop``` contains the code for poison attack. The arguments are contained in ```options.py```

## How to run the experiment
### Adversarial poison generation
Run the following command to generete the poisons under PGD-200 attack ensembled with our causal model. You need explicitly specify `causal_model_path`, `causal_config`, and `poison_path`.
```
python anneal.py --net ResNet18 --poisonkey 3 --modelkey 1 --recipe targeted --eps 8 --budget 1.0 \
                 --save poison_dataset --poison_path /path/to/save/your/poisons \
                 --attackoptim PGD --restarts 1 \
                 --attackiter 200 --ensemble 2 --causal_model_path /path/to/save/your/checkpoint \
                 --causal_config /path/to/save/your/config/file --causal_data_no_normalize \
                 --causal_loss_type  perturb_s_output \
```

### Poison evaluation
You need to go into the sub-directory `poison_evaluation/` for the evaluation of your poisons with multiple defense baselines.

## Our modification of code
### ```village/clients```
We add ```client_single_w_causal.py```, where we define our causal model.

We add ```causal_criterion.py```, where we calculate several types of loss using our causal model

### ```village/material```
A parameter ```causal_data_no_normalize``` in ```option.py``` to contral the **normalize** .

### ```village/shop```
We add ```forgemaster_targeted_w_causal.py```, where we employ different **causal_criterion** in ```_define_objective``` and combine the causal loss with the original CrossEntropy loss for the ST model to perform poison attack

We add ```forgemaster_targeted_mc.py```, where we change the way how the target label is generated comapring to ```forgemaster_targeted.py```

### ```village/options.py```

Since line **103**, we add arguments for causal model. When these arguments are not specified, the baseline poison attack will be conducted.

- ```causal_model_path```: path to checkpoint of the causal model
- ```causal_config```： the ```.yaml``` file used to train the causal model, used to determine the architecture of the causal model. 
- ```causal_data_no_normalize```： If called, then the data preprocess will not use normalize
- ```causal_beta```： the coefficient for the loss of causal criterion when combining the loss from causal criterion for the causal model and the loss from CrossEntropy for ST model
- ```causal_loss_type```： the type of the causal criterion used
- ```only_causal```： only use causal model for poison attack, otherwise the loss from causal model and ST model will be combined
- ```causal_reverse```： change from attacking **s** to attacking **v** or from attacking **v** to attacking **s**，only works for causal criterion “perturb_s_output" and ”perturb_v_output"
- ```st4causal```： replace ST model with the causal model to calculate the CE loss
