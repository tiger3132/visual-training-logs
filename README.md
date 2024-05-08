# Visualization of Training Logs and Performance Prediction

This repository contains the implementation of Visualization of Training Logs and Performance Prediction

## ResNet18 Training and Visualization of Training Logs

### Installation
```python
pip install -r requirements.txt
```
### Training random configurations
```python
python train/train_resnet_random.py
```
### Training Logs Analysis experiment

**Example A:**
```python
python train/train_resnet.py --batch_size 64 --optimizer adamcpr --lr 0.001 --kappa_init_param 1000 --kappa_init_method warm_start --wd_schedule_type cosine --lr_warmup_steps 200 --lr_decay_factor 0.1
```

**Example B:**
```python
python train/train_resnet.py --batch_size 64 --optimizer adamcpr --lr 0.1 --kappa_init_param 1000 --kappa_init_method warm_start --wd_schedule_type cosine --lr_warmup_steps 200 --lr_decay_factor 0.1 --data_transform 0
```

**Example C:**
```python
python train/train_resnet.py --batch_size 128 --optimizer sgd --lr 0.01 --momentum 0.5 --weight_decay 0.01 --wd_schedule_type cosine --lr_warmup_steps 200 --lr_decay_factor 0.1
```

**Example D:**
```python
python train/train_resnet.py --batch_size 256 --optimizer sgd --lr 0.0001 --momentum 0.25 --weight_decay 0.01 --wd_schedule_type cosine --lr_warmup_steps 400 --lr_decay_factor 0.1 --data_transform 0
```

**Example E:**
```python
python train/train_resnet.py --batch_size 128 --optimizer adamw --lr 0.001 --beta1 0.9 --beta2 0.98 --weight_decay 0.001 --wd_schedule_type cosine --lr_warmup_steps 200 --lr_decay_factor 0.1
```

**Example F:**
```python
python train/train_resnet.py --batch_size 128 --optimizer adamw --lr 0.001 --beta1 0.9 --beta2 0.98 --weight_decay 0.001 --wd_schedule_type cosine --lr_warmup_steps 200 --lr_decay_factor 0.1 --data_transform 0
```

**Plot parameter graphs for all models already trained:**
```python
python graph/all_graphs
```

## For Performance Prediction

### Installation
```python
pip install -r requirements_pred.txt
```

### Dataset and features from ResNet18 parameters

**For weights and biases:**
```python
python prediction/table_finale.py
```
**and for validation loss:**
```python
python prediction/table_creator_3.py
```

### HPO Experiment examples

Hyperparameter Optimization of XGBoost on dataset containing 0-5000 timesteps, batch normalization layer and mean of gradient:
```python
python prediction/xgb_prediction_finale.py --hp N --p0to5000 Y --p5000to10000 N --p10000to15000 N --p15000to20000 N --trainloss0to5000 N --trainloss5000to10000 N --trainloss10000to15000 N --trainloss15000to20000 N --validloss0to5000 N --validloss5000to10000 N --validloss10000to15000 N --validloss15000to20000 N --batch_norm Y --conv N --before_relu N --after_relu N --downsample N --gradnorm N --gradmean Y --gradpercent N --first_layer_1 N --first_layer_2 N --first_layer_3 N --first_layer_4 N --middle_layer_1 N --middle_layer_2 N --middle_layer_3 N --middle_layer_4 N --last_layer_1 N --last_layer_2 N --last_layer_3 N --last_layer_4 N
```
Evaluation of XGBoost with optimal hyperparameters:
```python
python prediction/xgb_prediction_finale.py --tree_no 94 --mode test --hp N --p0to5000 Y --p5000to10000 N --p10000to15000 N --p15000to20000 N --trainloss0to5000 N --trainloss5000to10000 N --trainloss10000to15000 N --trainloss15000to20000 N --validloss0to5000 N --validloss5000to10000 N --validloss10000to15000 N --validloss15000to20000 N --batch_norm Y --conv N --before_relu N --after_relu N --downsample N --gradnorm N --gradmean Y --gradpercent N --first_layer_1 N --first_layer_2 N --first_layer_3 N --first_layer_4 N --middle_layer_1 N --middle_layer_2 N --middle_layer_3 N --middle_layer_4 N --last_layer_1 N --last_layer_2 N --last_layer_3 N --last_layer_4 N
```


