from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from argparse import ArgumentParser
import os
import numpy as np
import pandas as pd
import torch


parser = ArgumentParser()
parser.add_argument("--model_name", type=str, default="ResNet18")
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--max_train_steps", type=int, default=20000)
parser.add_argument("--output_dir", type=str, default="cifar100")
parser.add_argument("--session", type=str, default='test_resnet')
parser.add_argument("--remove", type=str, default='none')
args = parser.parse_args()


#task_name = f"{args.model_name}_seed{args.seed}_steps{args.max_train_steps}"

#directory_str = f"/work/dlclarge1/nawongsk-MySpace/{args.output_dir}/{args.session}/{task_name}"
directory_str = f"/work/dlclarge1/nawongsk-MySpace/{args.output_dir}/{args.session}"

directory = os.fsencode(directory_str)

y_arr = []
# Norm
valid_loss_norm_arr = [[], [], [], []]
train_loss_norm_arr = [[], [], [], []]
# Magnitude of grad
valid_loss_grad_norm_arr = [[], [], [], []]
train_loss_grad_norm_arr = [[], [], [], []]
# Mean
valid_loss_mean_arr = [[], [], [], []]
train_loss_mean_arr = [[], [], [], []]
# Mean of grad
valid_loss_grad_mean_arr = [[], [], [], []]
train_loss_grad_mean_arr = [[], [], [], []]

task_names = []
"""
hp_df = pd.DataFrame({
    "batch_size": [],
    "optimizer": [],
    "lr": [],
    "weight_decay": [],
    "wd_schedule_type": [],
    "momentum": [],
    "lr_warmup_steps": [],
    "lr_decay_factor": [],
    "data_transform": [],
    "batch_norm": [],
    "beta1": [],
    "beta2": [],
    "kappa_init_param": [],
    "kappa_init_method": [],
    "reg_function": [],
    "kappa_update": [],
    "kappa_adapt": [],
    "apply_lr": [],
    })
"""
hp_dict = {
    "batch_size": [],
    "optimizer": [],
    "lr": [],
    "weight_decay": [],
    "wd_schedule_type": [],
    "momentum": [],
    "lr_warmup_steps": [],
    "lr_decay_factor": [],
    "data_transform": [],
    "batch_norm": [],
    "beta1": [],
    "beta2": [],
    "kappa_init_param": [],
    "kappa_init_method": [],
    "reg_function": [],
    "kappa_update": [],
    "kappa_adapt": [],
    "apply_lr": [],
    }


for seed_file in os.listdir(directory):
    seedfilename = os.fsdecode(seed_file)
    print(seedfilename)
    task_name = seedfilename
    print(directory_str)
    task_folder = directory_str + "/" + seedfilename
#for file in os.listdir(directory):
    for file in os.listdir(task_folder):
        filename = os.fsdecode(file)
        print(f"file name: {filename}")

        if f"{args.remove}" == "sgd" or f"{args.remove}" == "adam":
            if f"{args.remove}" in filename:
                print("continue")
                continue

        hp_arr = filename.split("_")

        if "cosine" in filename or "linear" in filename:
            if "sgd" in filename:
                """
                df_new = {"batch_size": int(hp_arr[0][1:]), 
                        "optimizer": hp_arr[1], 
                        "lr": float(hp_arr[2][1:]), 
                        "weight_decay": float(hp_arr[3][1:]), 
                        "wd_schedule_type": hp_arr[4][1:], 
                        "momentum": float(hp_arr[5][6:]), 
                        "lr_warmup_steps": int(hp_arr[6][6:]), 
                        "lr_decay_factor": float(hp_arr[7][7:]), 
                        "data_transform": hp_arr[8][5:], 
                        "batch_norm": hp_arr[9][2:], 
                        "beta1": None, 
                        "beta2": None, 
                        "kappa_init_param": None, 
                        "kappa_init_method": None, 
                        "reg_function": None, 
                        "kappa_update": None, 
                        "kappa_adapt": None, 
                        "apply_lr": None}
                """
                hp_dict["batch_size"].append(int(hp_arr[0][1:]))
                hp_dict["optimizer"].append(hp_arr[1])
                hp_dict["lr"].append(float(hp_arr[2][1:]))
                hp_dict["weight_decay"].append(float(hp_arr[3][1:]))
                hp_dict["wd_schedule_type"].append(hp_arr[4][1:])
                hp_dict["momentum"].append(float(hp_arr[5][6:]))
                hp_dict["lr_warmup_steps"].append(int(hp_arr[6][6:]))
                hp_dict["lr_decay_factor"].append(float(hp_arr[7][7:]))
                hp_dict["data_transform"].append(hp_arr[8][5:])
                hp_dict["batch_norm"].append(hp_arr[9][2:])
                hp_dict["beta1"].append(None)
                hp_dict["beta2"].append(None)
                hp_dict["kappa_init_param"].append(None)
                hp_dict["kappa_init_method"].append(None)
                hp_dict["reg_function"].append(None)
                hp_dict["kappa_update"].append(None)
                hp_dict["kappa_adapt"].append(None)
                hp_dict["apply_lr"].append(None)
            elif "adamw" in filename:
                hp_dict["batch_size"].append(int(hp_arr[0][1:]))
                hp_dict["optimizer"].append(hp_arr[1])
                hp_dict["lr"].append(float(hp_arr[2][1:]))
                hp_dict["weight_decay"].append(float(hp_arr[3][1:]))
                hp_dict["wd_schedule_type"].append(hp_arr[4][1:])
                hp_dict["momentum"].append(None)
                hp_dict["lr_warmup_steps"].append(int(hp_arr[5][6:]))
                hp_dict["lr_decay_factor"].append(float(hp_arr[6][7:]))
                hp_dict["data_transform"].append(hp_arr[7][5:])
                hp_dict["batch_norm"].append(hp_arr[8][2:])
                hp_dict["beta1"].append(float(hp_arr[9][2:]))
                hp_dict["beta2"].append(float(hp_arr[10][2:]))
                hp_dict["kappa_init_param"].append(None)
                hp_dict["kappa_init_method"].append(None)
                hp_dict["reg_function"].append(None)
                hp_dict["kappa_update"].append(None)
                hp_dict["kappa_adapt"].append(None)
                hp_dict["apply_lr"].append(None)
            elif "adamcpr" in filename:
                if "warm_start" in filename:
                    hp_dict["batch_size"].append(int(hp_arr[0][1:]))
                    hp_dict["optimizer"].append(hp_arr[1])
                    hp_dict["lr"].append(float(hp_arr[7][1:]))
                    hp_dict["weight_decay"].append(None)
                    hp_dict["wd_schedule_type"].append(hp_arr[14][1:])
                    hp_dict["momentum"].append(None)
                    hp_dict["lr_warmup_steps"].append(int(hp_arr[10][6:]))
                    hp_dict["lr_decay_factor"].append(float(hp_arr[11][7:]))
                    hp_dict["data_transform"].append(hp_arr[12][5:])
                    hp_dict["batch_norm"].append(hp_arr[13][2:])
                    hp_dict["beta1"].append(None)
                    hp_dict["beta2"].append(None)
                    hp_dict["kappa_init_param"].append(int(hp_arr[2][1:]) if hp_arr[2][1:].isdecimal() else float(hp_arr[2][1:]))
                    hp_dict["kappa_init_method"].append(hp_arr[3][1:] + "_" + hp_arr[4])
                    hp_dict["reg_function"].append(hp_arr[5][2:])
                    hp_dict["kappa_update"].append(float(hp_arr[6][1:]))
                    hp_dict["kappa_adapt"].append(hp_arr[8][5:])
                    hp_dict["apply_lr"].append(hp_arr[9][1:])
                else:
                    hp_dict["batch_size"].append(int(hp_arr[0][1:]))
                    hp_dict["optimizer"].append(hp_arr[1])
                    hp_dict["lr"].append(float(hp_arr[6][1:]))
                    hp_dict["weight_decay"].append(None)
                    hp_dict["wd_schedule_type"].append(hp_arr[13][1:])
                    hp_dict["momentum"].append(None)
                    hp_dict["lr_warmup_steps"].append(int(hp_arr[9][6:]))
                    hp_dict["lr_decay_factor"].append(float(hp_arr[10][7:]))
                    hp_dict["data_transform"].append(hp_arr[11][5:])
                    hp_dict["batch_norm"].append(hp_arr[12][2:])
                    hp_dict["beta1"].append(None)
                    hp_dict["beta2"].append(None)
                    hp_dict["kappa_init_param"].append(int(hp_arr[2][1:]) if hp_arr[2][1:].isdecimal() else float(hp_arr[2][1:]))
                    hp_dict["kappa_init_method"].append(hp_arr[3][1:])
                    hp_dict["reg_function"].append(hp_arr[4][2:])
                    hp_dict["kappa_update"].append(float(hp_arr[5][1:]))
                    hp_dict["kappa_adapt"].append(hp_arr[7][5:])
                    hp_dict["apply_lr"].append(hp_arr[8][1:])
        else:
            if "sgd" in filename:
                hp_dict["batch_size"].append(int(hp_arr[0][1:]))
                hp_dict["optimizer"].append(hp_arr[1])
                hp_dict["lr"].append(float(hp_arr[2][1:]))
                hp_dict["weight_decay"].append(float(hp_arr[3][1:]))
                hp_dict["wd_schedule_type"].append(hp_arr[4][1:])
                hp_dict["momentum"].append(float(hp_arr[5][6:]))
                hp_dict["lr_warmup_steps"].append(None)
                hp_dict["lr_decay_factor"].append(None)
                hp_dict["data_transform"].append(hp_arr[6][5:])
                hp_dict["batch_norm"].append(hp_arr[7][2:])
                hp_dict["beta1"].append(None)
                hp_dict["beta2"].append(None)
                hp_dict["kappa_init_param"].append(None)
                hp_dict["kappa_init_method"].append(None)
                hp_dict["reg_function"].append(None)
                hp_dict["kappa_update"].append(None)
                hp_dict["kappa_adapt"].append(None)
                hp_dict["apply_lr"].append(None)
            elif "adamw" in filename:
                hp_dict["batch_size"].append(int(hp_arr[0][1:]))
                hp_dict["optimizer"].append(hp_arr[1])
                hp_dict["lr"].append(float(hp_arr[2][1:]))
                hp_dict["weight_decay"].append(float(hp_arr[3][1:]))
                hp_dict["wd_schedule_type"].append(hp_arr[4][1:])
                hp_dict["momentum"].append(None)
                hp_dict["lr_warmup_steps"].append(None)
                hp_dict["lr_decay_factor"].append(None)
                hp_dict["data_transform"].append(hp_arr[5][5:])
                hp_dict["batch_norm"].append(hp_arr[6][2:])
                hp_dict["beta1"].append(float(hp_arr[7][2:]))
                hp_dict["beta2"].append(float(hp_arr[8][2:]))
                hp_dict["kappa_init_param"].append(None)
                hp_dict["kappa_init_method"].append(None)
                hp_dict["reg_function"].append(None)
                hp_dict["kappa_update"].append(None)
                hp_dict["kappa_adapt"].append(None)
                hp_dict["apply_lr"].append(None)
            elif "adamcpr" in filename:
                if "warm_start" in filename:
                    hp_dict["batch_size"].append(int(hp_arr[0][1:]))
                    hp_dict["optimizer"].append(hp_arr[1])
                    hp_dict["lr"].append(float(hp_arr[7][1:]))
                    hp_dict["weight_decay"].append(None)
                    hp_dict["wd_schedule_type"].append(hp_arr[12][1:])
                    hp_dict["momentum"].append(None)
                    hp_dict["lr_warmup_steps"].append(None)
                    hp_dict["lr_decay_factor"].append(None)
                    hp_dict["data_transform"].append(hp_arr[10][5:])
                    hp_dict["batch_norm"].append(hp_arr[11][2:])
                    hp_dict["beta1"].append(None)
                    hp_dict["beta2"].append(None)
                    hp_dict["kappa_init_param"].append(int(hp_arr[2][1:]) if hp_arr[2][1:].isdecimal() else float(hp_arr[2][1:]))
                    hp_dict["kappa_init_method"].append(hp_arr[3][1:] + "_" + hp_arr[4])
                    hp_dict["reg_function"].append(hp_arr[5][2:])
                    hp_dict["kappa_update"].append(float(hp_arr[6][1:]))
                    hp_dict["kappa_adapt"].append(hp_arr[8][5:])
                    hp_dict["apply_lr"].append(hp_arr[9][1:])
                else:
                    hp_dict["batch_size"].append(int(hp_arr[0][1:]))
                    hp_dict["optimizer"].append(hp_arr[1])
                    hp_dict["lr"].append(float(hp_arr[6][1:]))
                    hp_dict["weight_decay"].append(None)
                    hp_dict["wd_schedule_type"].append(hp_arr[11][1:])
                    hp_dict["momentum"].append(None)
                    hp_dict["lr_warmup_steps"].append(None)
                    hp_dict["lr_decay_factor"].append(None)
                    hp_dict["data_transform"].append(hp_arr[9][5:])
                    hp_dict["batch_norm"].append(hp_arr[10][2:])
                    hp_dict["beta1"].append(None)
                    hp_dict["beta2"].append(None)
                    hp_dict["kappa_init_param"].append(int(hp_arr[2][1:]) if hp_arr[2][1:].isdecimal() else float(hp_arr[2][1:]))
                    hp_dict["kappa_init_method"].append(hp_arr[3][1:])
                    hp_dict["reg_function"].append(hp_arr[4][2:])
                    hp_dict["kappa_update"].append(float(hp_arr[5][1:]))
                    hp_dict["kappa_adapt"].append(hp_arr[7][5:])
                    hp_dict["apply_lr"].append(hp_arr[8][1:])


        if hp_dict["wd_schedule_type"][-1] == "cosine" or hp_dict["wd_schedule_type"][-1] == "linear":
            if hp_dict["optimizer"][-1] == "sgd":
                expt_name = f"b{hp_dict['batch_size'][-1]}_{hp_dict['optimizer'][-1]}_l{hp_dict['lr'][-1]}_w{hp_dict['weight_decay'][-1]}_t{hp_dict['wd_schedule_type'][-1]}_moment{hp_dict['momentum'][-1]}_lrwarm{hp_dict['lr_warmup_steps'][-1]}_lrdecay{hp_dict['lr_decay_factor'][-1]}_trans{hp_dict['data_transform'][-1]}_bn{hp_dict['batch_norm'][-1]}"
            elif hp_dict['optimizer'][-1] == "adamw":
                expt_name = f"b{hp_dict['batch_size'][-1]}_{hp_dict['optimizer'][-1]}_l{hp_dict['lr'][-1]}_w{hp_dict['weight_decay'][-1]}_t{hp_dict['wd_schedule_type'][-1]}_lrwarm{hp_dict['lr_warmup_steps'][-1]}_lrdecay{hp_dict['lr_decay_factor'][-1]}_trans{hp_dict['data_transform'][-1]}_bn{hp_dict['batch_norm'][-1]}_b1{hp_dict['beta1'][-1]}_b2{hp_dict['beta2'][-1]}"
            elif hp_dict['optimizer'][-1] == "adamcpr":
                expt_name = f"b{hp_dict['batch_size'][-1]}_{hp_dict['optimizer'][-1]}_p{hp_dict['kappa_init_param'][-1]}_m{hp_dict['kappa_init_method'][-1]}_kf{hp_dict['reg_function'][-1]}_r{hp_dict['kappa_update'][-1]}_l{hp_dict['lr'][-1]}_adapt{hp_dict['kappa_adapt'][-1]}_g{hp_dict['apply_lr'][-1]}_lrwarm{hp_dict['lr_warmup_steps'][-1]}_lrdecay{hp_dict['lr_decay_factor'][-1]}_trans{hp_dict['data_transform'][-1]}_bn{hp_dict['batch_norm'][-1]}_t{hp_dict['wd_schedule_type'][-1]}"
        else:
            if hp_dict['optimizer'][-1] == "sgd":
                expt_name = f"b{hp_dict['batch_size'][-1]}_{hp_dict['optimizer'][-1]}_l{hp_dict['lr'][-1]}_w{hp_dict['weight_decay'][-1]}_t{hp_dict['wd_schedule_type'][-1]}_moment{hp_dict['momentum'][-1]}_trans{hp_dict['data_transform'][-1]}_bn{hp_dict['batch_norm'][-1]}"
            elif hp_dict['optimizer'][-1] == "adamw":
                expt_name = f"b{hp_dict['batch_size'][-1]}_{hp_dict['optimizer'][-1]}_l{hp_dict['lr'][-1]}_w{hp_dict['weight_decay'][-1]}_t{hp_dict['wd_schedule_type'][-1]}_trans{hp_dict['data_transform'][-1]}_bn{hp_dict['batch_norm'][-1]}_b1{hp_dict['beta1'][-1]}_b2{hp_dict['beta2'][-1]}"
            elif hp_dict['optimizer'][-1] == "adamcpr":
                expt_name = f"b{hp_dict['batch_size'][-1]}_{hp_dict['optimizer'][-1]}_p{hp_dict['kappa_init_param'][-1]}_m{hp_dict['kappa_init_method'][-1]}_kf{hp_dict['reg_function'][-1]}_r{hp_dict['kappa_update'][-1]}_l{hp_dict['lr'][-1]}_adapt{hp_dict['kappa_adapt'][-1]}_g{hp_dict['apply_lr'][-1]}_trans{hp_dict['data_transform'][-1]}_bn{hp_dict['batch_norm'][-1]}_t{hp_dict['wd_schedule_type'][-1]}"
        print(f"test name {expt_name}")
        assert expt_name == filename 
        #hp_df = hp_df._append(df_new, ignore_index=True)
        #print(f"df hp shape: {hp_df.shape}")

        expt_dir = f"/work/dlclarge1/nawongsk-MySpace/{args.output_dir}/{args.session}/{task_name}/{filename}/version_0"
        event = EventAccumulator(expt_dir)
        event.Reload()
        y = event.Scalars('test_accuracy')
        y_arr.append(y[0].value)
        valid_loss = np.array([event_scalar.value for event_scalar in event.Scalars(f'validation_loss')])
        train_loss = np.array([event_scalar.value for event_scalar in event.Scalars(f'train_loss')])
        #print(f"valid loss: {np.shape(valid_loss)}")
        #print(f"train loss: {np.shape(train_loss)}")
        if np.shape(valid_loss)[0] == 113: 
            valid_loss = valid_loss[:-1]

        # Split into 4 sections
        valid_loss_split = np.array_split(valid_loss, 4)
        train_loss_split = np.array_split(train_loss, 4)
        #print(f"valid loss split: {np.shape(valid_loss_split)}")
        #print(f"train loss split: {np.shape(train_loss_split)}")

        # Gradient
        valid_loss_grad = np.gradient(valid_loss_split, axis=1) 
        train_loss_grad = np.gradient(train_loss_split, axis=1) 

        # Norm
        valid_loss_norm = np.linalg.norm(valid_loss_split, axis=1)
        train_loss_norm = np.linalg.norm(train_loss_split, axis=1)

        # MEAN
        valid_loss_mean = np.mean(valid_loss_split, axis=1)
        train_loss_mean = np.mean(train_loss_split, axis=1)

        # Magnitude of grad
        valid_loss_grad_norm = np.linalg.norm(valid_loss_grad, axis=1)
        train_loss_grad_norm = np.linalg.norm(train_loss_grad, axis=1)

        # Mean of grad
        valid_loss_grad_mean = np.mean(valid_loss_grad, axis=1)
        train_loss_grad_mean = np.mean(train_loss_grad, axis=1)

        for i in range(0, 4):
            valid_loss_norm_arr[i].append(valid_loss_norm[i])
            train_loss_norm_arr[i].append(train_loss_norm[i])

            valid_loss_grad_norm_arr[i].append(valid_loss_grad_norm[i])
            train_loss_grad_norm_arr[i].append(train_loss_grad_norm[i])

            valid_loss_mean_arr[i].append(valid_loss_mean[i])
            train_loss_mean_arr[i].append(train_loss_mean[i])

            valid_loss_grad_mean_arr[i].append(valid_loss_grad_mean[i])
            train_loss_grad_mean_arr[i].append(train_loss_grad_mean[i])

        task_names.append(filename)

hp_df = pd.DataFrame.from_dict(hp_dict)

print(f"valid_loss_norm_arr length: {len(valid_loss_norm_arr[0])}")
print(f"df hp shape: {hp_df.shape}")
# Dataframes for norm

df_train_val_loss = pd.DataFrame({
    'valLoss_norm0-5000': valid_loss_norm_arr[0],
    'valLoss_norm5000-10000': valid_loss_norm_arr[1],
    'valLoss_norm10000-15000': valid_loss_norm_arr[2],
    'valLoss_norm15000-20000': valid_loss_norm_arr[3],
    'trainLoss_norm0-5000': train_loss_norm_arr[0],
    'trainLoss_norm5000-10000': train_loss_norm_arr[1],
    'trainLoss_norm10000-15000': train_loss_norm_arr[2],
    'trainLoss_norm15000-20000': train_loss_norm_arr[3],
    'valLoss_gradnorm0-5000': valid_loss_grad_norm_arr[0],
    'valLoss_gradnorm5000-10000': valid_loss_grad_norm_arr[1],
    'valLoss_gradnorm10000-15000': valid_loss_grad_norm_arr[2],
    'valLoss_gradnorm15000-20000': valid_loss_grad_norm_arr[3],
    'trainLoss_gradnorm0-5000': train_loss_grad_norm_arr[0],
    'trainLoss_gradnorm5000-10000': train_loss_grad_norm_arr[1],
    'trainLoss_gradnorm10000-15000': train_loss_grad_norm_arr[2],
    'trainLoss_gradnorm15000-20000': train_loss_grad_norm_arr[3],
    'valLoss_mean0-5000': valid_loss_mean_arr[0],
    'valLoss_mean5000-10000': valid_loss_mean_arr[1],
    'valLoss_mean10000-15000': valid_loss_mean_arr[2],
    'valLoss_mean15000-20000': valid_loss_mean_arr[3],
    'trainLoss_mean0-5000': train_loss_mean_arr[0],
    'trainLoss_mean5000-10000': train_loss_mean_arr[1],
    'trainLoss_mean10000-15000': train_loss_mean_arr[2],
    'trainLoss_mean15000-20000': train_loss_mean_arr[3],
    'valLoss_gradmean0-5000': valid_loss_grad_mean_arr[0],
    'valLoss_gradmean5000-10000': valid_loss_grad_mean_arr[1],
    'valLoss_gradmean10000-15000': valid_loss_grad_mean_arr[2],
    'valLoss_gradmean15000-20000': valid_loss_grad_mean_arr[3],
    'trainLoss_gradmean0-5000': train_loss_grad_mean_arr[0],
    'trainLoss_gradmean5000-10000': train_loss_grad_mean_arr[1],
    'trainLoss_gradmean10000-15000': train_loss_grad_mean_arr[2],
    'trainLoss_gradmean15000-20000': train_loss_grad_mean_arr[3],
    'test_acc': y_arr
    })
df_train_val_loss_no_outlier = df_train_val_loss.drop(df_train_val_loss[df_train_val_loss.test_acc < 0.05].index)
if args.remove == "sgd" or args.remove == "adam":
    df_train_val_loss_no_outlier.to_csv(f"df_train_val_loss_no_outlier_&_{args.remove}.csv")
else:
    df_train_val_loss_no_outlier.to_csv("df_train_val_loss_no_outlier.csv")

hp_df["test_acc"] = y_arr
hp_df_no_outlier = hp_df.drop(hp_df[hp_df.test_acc < 0.05].index)
if args.remove == "sgd" or args.remove == "adam":
    hp_df_no_outlier.to_csv(f"hp_df_no_outlier_&_{args.remove}.csv")
else:
    hp_df_no_outlier.to_csv("hp_df_no_outlier.csv")
