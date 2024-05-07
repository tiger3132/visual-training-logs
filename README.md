# Visualization of Training Logs and Performance Prediction

This repository contains the implementation of Visualization of Training Logs and Performance Prediction

## ResNet18 Training and Visualization of Training Logs

### Installation
```python
pip install -r requirements.txt
```
**Training random configurations**
```python
python train/train_resnet_random.py
```
**Training Logs Analysis experiment**
Example A:
```python
python train/train_resnet.py --batch_size 64 --optimizer adamcpr --lr 0.001 --kappa_init_param 1000 --kappa_init_method warm_start --wd_schedule_type cosine --lr_warmup_steps 200 --lr_decay_factor 0.1
```

Example B:
```python
python train/train_resnet.py --batch_size 64 --optimizer adamcpr --lr 0.1 --kappa_init_param 1000 --kappa_init_method warm_start --wd_schedule_type cosine --lr_warmup_steps 200 --lr_decay_factor 0.1 --data_transform 0
```

Example C:
```python
python train/train_resnet.py --batch_size 128 --optimizer sgd --lr 0.01 --momentum 0.5 --weight_decay 0.01 --wd_schedule_type cosine --lr_warmup_steps 200 --lr_decay_factor 0.1
```

Example D:
```python
python train/train_resnet.py --batch_size 256 --optimizer sgd --lr 0.0001 --momentum 0.25 --weight_decay 0.01 --wd_schedule_type cosine --lr_warmup_steps 400 --lr_decay_factor 0.1 --data_transform 0
```

Example E:
```python
python train/train_resnet.py --batch_size 128 --optimizer adamw --lr 0.001 --beta1 0.9 --beta2 0.98 --weight_decay 0.001 --wd_schedule_type cosine --lr_warmup_steps 200 --lr_decay_factor 0.1
```

Example F:
```python
python train/train_resnet.py --batch_size 128 --optimizer adamw --lr 0.001 --beta1 0.9 --beta2 0.98 --weight_decay 0.001 --wd_schedule_type cosine --lr_warmup_steps 200 --lr_decay_factor 0.1 --data_transform 0
```

## For Performance Prediction

### Installation
```python
pip install -r requirements_pred.txt
```


