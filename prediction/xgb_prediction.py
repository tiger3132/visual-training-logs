import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, cross_val_score, RepeatedStratifiedKFold, RandomizedSearchCV
import pandas as pd
import matplotlib.pyplot as plt
import graphviz
from argparse import ArgumentParser
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, roc_auc_score, f1_score
import numpy as np
import shap

import os
from columns_select import remove_table_col 


parser = ArgumentParser()
parser.add_argument("--tree_no", type=int, default=0)
parser.add_argument("--mode", type=str, default='cv', choices=["cv", "test"])

parser.add_argument("--hp", type=str, default='Y')
parser.add_argument("--p0to5000", type=str, default='Y')
parser.add_argument("--p5000to10000", type=str, default='Y')
parser.add_argument("--p10000to15000", type=str, default='Y')
parser.add_argument("--p15000to20000", type=str, default='Y')
parser.add_argument("--trainloss0to5000", type=str, default='Y')
parser.add_argument("--trainloss5000to10000", type=str, default='Y')
parser.add_argument("--trainloss10000to15000", type=str, default='Y')
parser.add_argument("--trainloss15000to20000", type=str, default='Y')
parser.add_argument("--validloss0to5000", type=str, default='Y')
parser.add_argument("--validloss5000to10000", type=str, default='Y')
parser.add_argument("--validloss10000to15000", type=str, default='Y')
parser.add_argument("--validloss15000to20000", type=str, default='Y')

parser.add_argument("--batch_norm", type=str, default='Y')
parser.add_argument("--conv", type=str, default='Y')
parser.add_argument("--before_relu", type=str, default='Y')
parser.add_argument("--after_relu", type=str, default='Y')
parser.add_argument("--downsample", type=str, default='Y')
parser.add_argument("--gradnorm", type=str, default='Y')
parser.add_argument("--gradmean", type=str, default='Y')
parser.add_argument("--gradpercent", type=str, default='Y')
parser.add_argument("--first_layer_1", type=str, default='Y')
parser.add_argument("--first_layer_2", type=str, default='Y')
parser.add_argument("--first_layer_3", type=str, default='Y')
parser.add_argument("--first_layer_4", type=str, default='Y')
parser.add_argument("--middle_layer_1", type=str, default='Y')
parser.add_argument("--middle_layer_2", type=str, default='Y')
parser.add_argument("--middle_layer_3", type=str, default='Y')
parser.add_argument("--middle_layer_4", type=str, default='Y')
parser.add_argument("--last_layer_1", type=str, default='Y')
parser.add_argument("--last_layer_2", type=str, default='Y')
parser.add_argument("--last_layer_3", type=str, default='Y')
parser.add_argument("--last_layer_4", type=str, default='Y')
parser.add_argument("--norm", type=str, default='Y')
parser.add_argument("--mean", type=str, default='Y')

args = parser.parse_args()
"""
directory_str = f"/work/dlclarge1/nawongsk-MySpace/prediction"
all_df = pd.DataFrame()
all_df_sgd = pd.DataFrame()
all_df_adam = pd.DataFrame()
directory = os.fsencode(directory_str)
# rename column
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if filename.endswith(".csv") and not "hp_df_no_outlier" in filename and not "df_train_val_loss_no_outlier" in filename:
        new_col_arr = []
        #print(filename)
        df = pd.read_csv(f"/work/dlclarge1/nawongsk-MySpace/prediction/{filename}")
        trans = ""
        if "grad_mean" in filename:
            trans = "gradmean"
        elif "grad_norm" in filename:
            trans = "gradnorm"
        elif "mean" in filename and not "grad_mean" in filename:
            trans = "mean"
        elif "norm" in filename and not "grad_norm" in filename:
            trans = "norm"
        for col_name in df.columns:
            #print(f"old: {col_name}")
            t_range = ""
            stat = ""
            wb = ""
            layer = ""
            if col_name.startswith("P0") or col_name.startswith("0"):
                t_range = "0-5000"
            elif col_name.startswith("P1"):
                t_range = "5000-10000"
            elif col_name.startswith("P2"):
                t_range = "10000-15000"
            elif col_name.startswith("P3"):
                t_range = "15000-20000"
            else:
                new_col_arr.append(col_name)
                #print("skip")
                continue

            if "df_mean_grad" in filename or "df_mean_mean" in filename or "df_mean_norm" in filename or col_name.endswith("_m"):
                stat = "Mean"
            elif "df_std" in filename or col_name.endswith("_st"):
                stat = "Std"
            elif col_name.endswith("_mx"):
                stat = "Max"
            elif col_name.endswith("_mn"):
                stat = "Min"

            if "_w_" in col_name or col_name.endswith("_w"):
                wb = "Wt"
            elif "_b_" in col_name or col_name.endswith("_b"):
                wb = "Bs"
            else:
                wb = "Wt"

            if col_name.startswith("P0_l") or col_name.startswith("P1_l") or col_name.startswith("P2_l") or col_name.startswith("P3_l"):
                layer += "L" + col_name[4:6]
            elif "_fc_" in col_name:
                layer += "fc"

            if "_bn1" in col_name:
                layer += "bn1"
            elif "_cn1" in col_name:
                layer += "cn1"
            elif "_bn2" in col_name:
                layer += "bn2"
            elif "_cn2" in col_name:
                layer += "cn2"
            assert trans != ""
            assert t_range != ""
            assert stat != ""
            assert layer != ""
            assert wb != ""
            new_col = layer + wb + stat + "_" + trans + t_range
            new_col_arr.append(new_col)
            #print(f"new: {new_col}")
            files = ["df_std_grad_norm_no_outlier.csv", "df_grad_norm_2_no_outlier.csv", "df_grad_norm_3_no_outlier.csv", "df_grad_norm_1_no_outlier.csv", "df_mean_mean_no_outlier.csv", "df_grad_mean_no_outlier.csv", "df_std_grad_mean_no_outlier.csv", "df_norm_2_no_outlier.csv", "df_norm_2_no_outlier.csv",  "df_std_mean_no_outlier.csv", "df_norm_3_no_outlier.csv"]
            old_tests = ["P0_l41_bn2_w", "P0_l41_bn2_w_mn", "P0_l41_bn2_b_mn", "P0_l41_bn1_b_st", "P0_l10_cn1", "P1_fc_w_mx", "P2_l41_cn2_w", "P3_l41_bn2_w_mn", "P2_l31_bn2_b_m", "P3_l41_cn2_w", "P3_bn1_w_mx"]
            new_tests = ["L41bn2WtStd_gradnorm0-5000", "L41bn2WtMin_gradnorm0-5000", "L41bn2BsMin_gradnorm0-5000", "L41bn1BsStd_gradnorm0-5000", "L10cn1WtMean_mean0-5000", "fcWtMax_gradmean5000-10000", "L41cn2WtStd_gradmean10000-15000", "L41bn2WtMin_norm15000-20000", "L31bn2BsMean_norm10000-15000", "L41cn2WtStd_mean15000-20000", "bn1WtMax_norm15000-20000"]
            for i in range(len(files)):
                if filename == files[i] and col_name == old_tests[i]:
                    assert new_col == new_tests[i]
        df.columns = new_col_arr
    elif "hp_df_no_outlier" in filename or "df_train_val_loss_no_outlier" in filename:
        df = pd.read_csv(f"/work/dlclarge1/nawongsk-MySpace/prediction/{filename}")

    if filename.endswith("_no_outlier.csv"):
        all_df = pd.concat([all_df, df], axis=1)
    if filename.endswith("_no_outlier_&_sgd.csv"):
        all_df_adam = pd.concat([all_df_adam, df], axis=1)
    if filename.endswith("_no_outlier_&_adam.csv"):
        all_df_sgd = pd.concat([all_df_sgd, df], axis=1)
"""
param_df = pd.read_csv("params_0to20000.csv")
hp_df = pd.read_csv("hp_df_no_outlier.csv")
train_val_loss_df = pd.read_csv("df_train_val_loss_no_outlier.csv")

all_df = pd.concat([param_df, hp_df, train_val_loss_df], axis=1)

all_df = all_df.loc[:,~all_df.columns.duplicated()].copy()
all_df = all_df.drop(columns=["Unnamed: 0"]) # Remove unnamed column
#all_df_sgd = all_df_sgd.T.drop_duplicates().T # Remove duplicate test accuracy columns
#all_df_sgd = all_df_sgd.loc[:,~all_df_sgd.columns.duplicated()].copy()
#all_df_sgd = all_df_sgd.drop(columns=["Unnamed: 0"]) # Remove unnamed column
#all_df_adam = all_df_adam.T.drop_duplicates().T # Remove duplicate test accuracy columns
#all_df_adam = all_df_adam.loc[:,~all_df_adam.columns.duplicated()].copy()
#all_df_adam = all_df_adam.drop(columns=["Unnamed: 0"]) # Remove unnamed column

"""
columns_to_remove = list(all_df.columns)
columns_to_remove.remove("test_acc")

if args.hp == "Y": 
    for col in ["batch_size", "optimizer", "apply_lr", "lr", "weight_decay", "wd_schedule_type", "momentum", "lr_warmup_steps", "lr_decay_factor", "data_transform", "batch_norm", "beta1", "beta2", "kappa_init_param", "kappa_init_method", "reg_function", "kappa_update", "kappa_adapt"]:
        if col in all_df and col in columns_to_remove:
            columns_to_remove.remove(col)

ts_range = set()
for col in all_df.columns:
    if args.p0to5000 == "Y" and col in columns_to_remove:
        if col.endswith("0-5000") and not "valLoss" in col and not "trainLoss" in col:
            ts_range.add("0-5000")         
    if args.p5000to10000 == "Y" and col in columns_to_remove:
        if col.endswith("5000-10000") and not "valLoss" in col and not "trainLoss" in col:
            ts_range.add("5000-10000")         
    if args.p10000to15000 == "Y" and col in columns_to_remove:
        if col.endswith("10000-15000") and not "valLoss" in col and not "trainLoss" in col:
            ts_range.add("10000-15000")         
    if args.p15000to20000 == "Y" and col in columns_to_remove:
        if col.endswith("15000-20000") and not "valLoss" in col and not "trainLoss" in col: 
            ts_range.add("15000-20000")         
    if args.trainloss0to5000 == "Y" and col in columns_to_remove:
        if col.endswith("0-5000") and "trainLoss" in col:
            columns_to_remove.remove(col)
    if args.trainloss5000to10000 == "Y" and col in columns_to_remove:
        if col.endswith("5000-10000") and "trainLoss" in col:
            columns_to_remove.remove(col)
    if args.trainloss10000to15000 == "Y" and col in columns_to_remove:
        if col.endswith("10000-15000") and "trainLoss" in col:
            columns_to_remove.remove(col)
    if args.trainloss15000to20000 == "Y" and col in columns_to_remove:
        if col.endswith("10000-15000") and "trainLoss" in col:
            columns_to_remove.remove(col)
    if args.validloss0to5000 == "Y" and col in columns_to_remove:
        if col.endswith("0-5000") and "valLoss" in col:
            columns_to_remove.remove(col)
    if args.validloss5000to10000 == "Y" and col in columns_to_remove: 
        if col.endswith("5000-10000") and "valLoss" in col:
            columns_to_remove.remove(col)
    if args.validloss10000to15000 == "Y" and col in columns_to_remove:
        if col.endswith("10000-15000") and "valLoss" in col:
            columns_to_remove.remove(col)
    if args.validloss15000to20000 == "Y" and col in columns_to_remove:
        if col.endswith("15000-20000") and "valLoss" in col:
            columns_to_remove.remove(col)


    if args.batch_norm == "Y" and col in columns_to_remove:
        if ("bn1" in col or "bn2" in col) and any(ts in col for ts in ts_range):
            columns_to_remove.remove(col)
    if args.conv == "Y" and col in columns_to_remove:
        if ("cn1" in col or "cn2" in col) and any(ts in col for ts in ts_range):
            columns_to_remove.remove(col)
    if args.before_relu == "Y" and col in columns_to_remove:
        if ("bn1" in col or "ds1" in col or col.startswith("L10bn2") or col.startswith("L11bn2") or col.startswith("L21bn2") or col.startswith("L31bn2") or col.startswith("L41bn2")) and any(ts in col for ts in ts_range):
            columns_to_remove.remove(col)
    if args.after_relu == "Y" and col in columns_to_remove:
        if ("cn2" in col or ("cn1" in col and not col.startswith("cn1")) or col.startswith("fc")) and any(ts in col for ts in ts_range):
            columns_to_remove.remove(col)
    if args.downsample == "Y" and col in columns_to_remove:
        if ("ds0" in col or "ds1" in col) and any(ts in col for ts in ts_range):
            columns_to_remove.remove(col)

    if args.first_layer_1 == "Y" and col in columns_to_remove:
        if (col.startswith("cn1") or col.startswith("bn1")) and any(ts in col for ts in ts_range):
            columns_to_remove.remove(col)
    if args.first_layer_2 == "Y" and col in columns_to_remove:
        if (col.startswith("L10cn1") or col.startswith("L10bn1")) and any(ts in col for ts in ts_range):
            columns_to_remove.remove(col)
    if args.first_layer_3 == "Y" and col in columns_to_remove:
        if (col.startswith("L10cn2") or col.startswith("L10bn2")) and any(ts in col for ts in ts_range):
            columns_to_remove.remove(col)
    if args.first_layer_4 == "Y" and col in columns_to_remove:
        if (col.startswith("L11cn1") or col.startswith("L11bn1")) and any(ts in col for ts in ts_range):
            columns_to_remove.remove(col)
    if args.last_layer_1 == "Y" and col in columns_to_remove:
        if (col.startswith("fc")) and any(ts in col for ts in ts_range):
            columns_to_remove.remove(col)
    if args.last_layer_2 == "Y" and col in columns_to_remove:
        if (col.startswith("L41bn2") or col.startswith("L41cn2")) and any(ts in col for ts in ts_range):
            columns_to_remove.remove(col)
    if args.last_layer_3 == "Y" and col in columns_to_remove:
        if (col.startswith("L41bn1") or col.startswith("L41cn1")) and any(ts in col for ts in ts_range):
            columns_to_remove.remove(col)
    if args.last_layer_4 == "Y" and col in columns_to_remove:
        if (col.startswith("L40ds0") or col.startswith("L40ds1") or col.startswith("L40bn2") or col.startswith("L40cn2")) and any(ts in col for ts in ts_range):
            columns_to_remove.remove(col)
    if args.middle_layer_1 == "Y" and col in columns_to_remove:
        if (col.startswith("L21cn1") or col.startswith("L21bn1")) and any(ts in col for ts in ts_range):
            columns_to_remove.remove(col)
    if args.middle_layer_2 == "Y" and col in columns_to_remove:
        if (col.startswith("L21cn2") or col.startswith("L21bn2")) and any(ts in col for ts in ts_range):
            columns_to_remove.remove(col)
    if args.middle_layer_3 == "Y" and col in columns_to_remove:
        if (col.startswith("L30cn1") or col.startswith("L30bn1")) and any(ts in col for ts in ts_range):
            columns_to_remove.remove(col)
    if args.middle_layer_4 == "Y" and col in columns_to_remove:
        if (col.startswith("L30cn2") or col.startswith("L30bn2")) and any(ts in col for ts in ts_range):
            columns_to_remove.remove(col)

    # Add columns to remove. Has to be last!

    if args.gradnorm == "N" and not col in columns_to_remove:
        if ("gradnorm" in col) and any(ts in col for ts in ts_range):
            columns_to_remove.append(col)

    if args.gradmean == "N" and not col in columns_to_remove:
        if ("gradmean" in col) and any(ts in col for ts in ts_range):
            columns_to_remove.append(col)

    if args.gradpercent == "N" and not col in columns_to_remove:
        if ("gradpercent" in col) and any(ts in col for ts in ts_range):
            columns_to_remove.append(col)

if args.validloss0to5000 == "Y":
    if args.gradnorm == "N":
        columns_to_remove.append("valLoss_gradnorm0-5000")
    if args.gradmean == "N":
        columns_to_remove.append("valLoss_gradmean0-5000")
    if args.mean == "N":
        columns_to_remove.append("valLoss_mean0-5000")
    if args.norm == "N":
        columns_to_remove.append("valLoss_norm0-5000")

"""
columns_to_remove = remove_table_col(all_df, args.hp, args.p0to5000, args.p5000to10000, args.p10000to15000, args.p15000to20000, args.trainloss0to5000, args.trainloss5000to10000, args.trainloss10000to15000, args.trainloss15000to20000, args.validloss0to5000, args.validloss5000to10000, args.validloss10000to15000, args.validloss15000to20000, args.batch_norm, args.conv, args.before_relu, args.after_relu, args.downsample, args.first_layer_1, args.first_layer_2, args.first_layer_3, args.first_layer_4, args.last_layer_1, args.last_layer_2, args.last_layer_3, args.last_layer_4, args.middle_layer_1, args.middle_layer_2, args.middle_layer_3, args.middle_layer_4, args.gradnorm, args.gradmean, args.gradpercent, args.mean, args.norm) 

for col in columns_to_remove:
    all_df = all_df.drop(columns=col)

le = LabelEncoder()
for col in all_df.columns:
    if all_df[col].dtype == "object":
        all_df[col] = le.fit_transform(all_df[col])

X, y = all_df.loc[:,all_df.columns != 'test_acc'], all_df["test_acc"] > 0.7
print(X)
#X_adam, y_adam = all_df_adam.loc[:,all_df_adam.columns != 'test_acc'], all_df_adam["test_acc"] > 0.7
#X_sgd, y_sgd = all_df_sgd.loc[:,all_df_sgd.columns != 'test_acc'], all_df_sgd["test_acc"] > 0.7
feature_names = X.columns
target_names = ["< 0.7", "> 0.7"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1) # 1
#X_train_adam, X_test_adam, y_train_adam, y_test_adam = train_test_split(X_adam, y_adam, test_size=0.2, random_state=1)
#X_train_sgd, X_test_sgd, y_train_sgd, y_test_sgd = train_test_split(X_sgd, y_sgd, test_size=0.2, random_state=1)
print("-----------------------------------------------------")
print("features")
for f in X_train.columns:
    print(f)
"""
predictions = []
for index in range(X_train.shape[0]):
    X_i = X_train.iloc[index]
    score = 0
    #tree 0
    if X_i["L31bn1BsMax_gradmean0-5000"] < -3.81550744e-05:
        if X_i["L21bn2WtMax_gradmean0-5000"] < 0.000604193832:
            if X_i["L21bn2BsMax_gradmean0-5000"] < 8.28270713e-05:
                score += -0.432319999
            else:
                score += 0.402353406
        else:
            if X_i["bn1WtMax_gradmean0-5000"] < 0.004088399:
                score += 1.34640658
            else:
                score += 0.387424231
    else:
        if X_i["L40bn2BsMax_gradmean0-5000"] < -0.000384268584:
            if X_i["L40bn1WtMin_gradmean0-5000"] < -0.0102354549:
                score += -0.390667409
            else:
                score += 0.223428547
        else:
            if X_i["L41bn2WtMax_gradmean0-5000"] < -0.00521303341:
                score += -0.167507485
            else:
                score += -0.600126207
    #tree 1
    if X_i["L31bn1BsMax_gradmean0-5000"] < -8.25269672e-06:
        if X_i["bn1WtMean_gradmean0-5000"] < -1.43805146e-05:
            if X_i["L21bn2WtMax_gradmean0-5000"] < 0.000341480976:
                score += 0.0688590109
            else:
                score += 0.614435911
        else:
            score += -0.305337489
    else:
        if X_i["L40bn2BsMax_gradmean0-5000"] < -1.83213378e-05:
            if X_i["L11bn2WtMean_gradmean0-5000"] < -0.00552446954:
                score += -0.378948092
            else:
                score += 0.13864328
        else:
            if X_i["L10bn1BsMean_gradmean0-5000"] < 8.51871846e-06:
                score += -0.518966615
            else:
                score += -0.182436585

    # tree 2
    if X_i["L21bn1BsMax_gradmean0-5000"] < -3.53985743e-05:
        if X_i["L21bn2BsMax_gradmean0-5000"] < -1.76809153e-05:
            score += -0.217274621
        else:
            if X_i["L10bn2WtStd_gradmean0-5000"] < 0.00139823137:
                score += 0.684398949
            else:
                score += 0.0442623533
    else:
        if X_i["bn1BsMean_gradmean0-5000"] < 0.000107207896:
            score += -0.475088507
        else:
            if X_i["L31bn1WtMin_gradmean0-5000"] < -0.0102707483:
                score += -0.451304078
            else:
                score += 0.0800472349

    #tree 3
    if X_i["L20bn1BsMean_gradmean0-5000"] < -4.88685982e-05:
        if X_i["L30bn2BsMax_gradmean0-5000"] < 0.0116301896:
            if X_i["L20bn1BsMax_gradmean0-5000"] < 2.78752723e-05:
                score += -0.234789193
            else:
                score += 0.279988885
        else:
            score += -0.43941623
    else:
        score += -0.420974076


    #tree 4
    if X_i["L11bn1BsMin_gradmean0-5000"] < -0.000309818919:
        if X_i["L41bn2WtStd_gradmean0-5000"] < 0.000196408233:
            score += 0.43681702
        else:
            if X_i["bn1WtMean_gradmean0-5000"] < -0.000118187068:
                score += 0.0685745403
            else:
                score += -0.299337268
    else:
        score += -0.421968281

    #tree 5
    if X_i["L40bn2BsMax_gradmean0-5000"] < -0.00150590693:
        if X_i["bn1WtMean_gradmean0-5000"] < -0.000318497419:
            if X_i["L30bn2WtMean_gradmean0-5000"] < 0.000685735664:
                score += 0.13666968
            else:
                score += 0.603169382
        else:
            score += -0.159780756
    else:
        if X_i["L20bn1BsStd_gradmean0-5000"] < 0.00048163053:
            if X_i["bn1BsMean_gradmean0-5000"] < 0.000316695689:
                score += -0.224637941
            else:
                score += 0.332147926
        else:
            if X_i["L41bn2WtMin_gradmean0-5000"] < -0.010594504:
                score += -0.20242393
            else:
                score += -0.537496865

    if score <= 0:
        predictions.append(0)
    else:
        predictions.append(1)

print(classification_report(y_train, predictions))
print(f"roc_auc: {roc_auc_score(y_train, predictions)}")
"""
if args.mode == "cv":

    print("-----------------------------------------------------")
    print("cv random search")
    
    param_grid = {
        "learning_rate": [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.8, 0.9, 1],
        "max_depth": [2,3,4,5,6,7,8,10],
        "n_estimators": [1,2,3,4,5,6,7,8,9,10,11,12,14,16,18,20,40,80,100],
        "min_child_weight": [1,3,5,7,9,11],
        "gamma": [0.01,0.1,0.2,0.3,0.4,0.5],
        "subsample": [0.5,0.6,0.7,0.8,0.9,1],
        "colsample_bytree": [0.5,0.6,0.7,0.8,0.9,1],
        "lambda": [0.001, 0.01, 0.1, 1, 10, 100],
        "alpha": [0.001, 0.01, 0.1, 1, 10, 100]

    }

    """
    param_grid = {
        "learning_rate": [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.8, 0.9, 1],
        "max_depth": [2,3,4,5,6,7,8,9,10],
        "n_estimators": [1,2,3,4,5,6,7,8,9,10],
        "min_child_weight": [1,3,5,7,9,11],
        "gamma": [0.01,0.1,0.2,0.3,0.4,0.5],
        "subsample": [0.5,0.6,0.7,0.8,0.9,1],
        "colsample_bytree": [0.5,0.6,0.7,0.8,0.9,1],
        "lambda": [0.001, 0.01, 0.1, 1, 10, 100],
        "alpha": [0.001, 0.01, 0.1, 1, 10, 100]

    }
    """


    # For balancing the imbalanced dataset
    dtrain = xgb.DMatrix(X_train, y_train)
    label = dtrain.get_label()
    ratio = float(np.sum(label == 0)) / np.sum(label == 1)

    bst = XGBClassifier(scale_pos_weight = ratio, objective='binary:logistic', tree_method = "hist", device = "cuda")
    cv = RepeatedStratifiedKFold(n_splits = 5, n_repeats=25, random_state=2) # 35
    search = RandomizedSearchCV(estimator=bst, param_distributions=param_grid, cv=cv, n_iter=1800, n_jobs=-1, scoring='roc_auc') # 700 works # 1800 doesnt work # 2h20 for 90 4h for 94 
    search.fit(X_train,y_train)
    results_df = pd.DataFrame(search.cv_results_)
    results_df = results_df.sort_values(by=["rank_test_score"])
    print(results_df[["params", "mean_test_score", "std_test_score"]].head(25))
    #results_df.to_csv(f"xgb_hp{args.hp}_p0to5000{args.p0to5000}_p5000to10000{args.p5000to10000}_p10000to15000{args.p10000to15000}_p15000to20000{args.p15000to20000}_t0to5000{args.trainloss0to5000}_t5000to10000{args.trainloss5000to10000}_t10000to15000{args.trainloss10000to15000}_t15000to20000{args.trainloss15000to20000}_v0to5000{args.validloss0to5000}_v5000to10000{args.validloss5000to10000}_v10000to15000{args.validloss10000to15000}_v15000to20000{args.validloss15000to20000}_gdnrm{args.gradnorm}_gdmn{args.gradmean}_nrm{args.norm}_mn{args.mean}.csv")
    results_df.to_csv(f"xgb_fl_1{args.first_layer_1}_2{args.first_layer_2}_3{args.first_layer_3}_4{args.first_layer_4}_ml_1{args.middle_layer_1}_2{args.middle_layer_2}_3{args.middle_layer_3}_4{args.middle_layer_4}_ll_1{args.last_layer_1}_2{args.last_layer_2}_3{args.last_layer_3}_4{args.last_layer_4}_batchnorm{args.batch_norm}_conv{args.conv}_beforerelu{args.before_relu}_afterrelu{args.after_relu}_downsample{args.downsample}_gradnorm{args.gradnorm}_gradmean{args.gradmean}_gradpercent{args.gradpercent}_large.csv")
elif args.mode == "test":
    print("-----------------------------------------------------")
    print("test")
    dtrain = xgb.DMatrix(X_train, y_train)
    dtest = xgb.DMatrix(X_test, y_test)
    """
    params = {"objective": "binary:logistic", "eval_metric":["error", "auc", "logloss"], "device": "cuda", 'subsample': 0.9, 'n_estimators': 1, 'min_child_weight': 9, 'max_depth': 5, 'learning_rate': 0.35, 'lambda': 0.1, 'gamma': 0.1, 'colsample_bytree': 1, 'alpha': 1}
    n = 60
    model = xgb.cv(
       params=params,
       dtrain=dtrain,
       num_boost_round=n,
       nfold=5,
       verbose_eval=5,
       early_stopping_rounds=30,
    )
    #print(model.keys())
    """
    label = dtrain.get_label()
    ratio = float(np.sum(label == 0)) / np.sum(label == 1)
    if args.tree_no == 21:
        # Acc
        #bst = XGBClassifier(scale_pos_weight = ratio, n_estimators=14, max_depth=2, learning_rate=0.3, objective='binary:logistic', subsample=1, min_child_weight=9, reg_lambda=0.01, gamma=0.4, colsample_bytree=0.8, alpha=0.1)
        # AUC
        #bst = XGBClassifier(scale_pos_weight = ratio, n_estimators=40, max_depth=3, learning_rate=0.05, objective='binary:logistic', subsample=1, min_child_weight=1, reg_lambda=0.001, gamma=0.01, colsample_bytree=0.5, alpha=0.01)
        bst = XGBClassifier(scale_pos_weight = ratio, n_estimators=18, max_depth=10, learning_rate=0.6, objective='binary:logistic', subsample=0.8, min_child_weight=7, reg_lambda=100, gamma=0.4, colsample_bytree=0.9, alpha=1)
    elif args.tree_no == 89:
        #bst = XGBClassifier(scale_pos_weight = ratio, n_estimators=120, max_depth=6, learning_rate=0.8, objective='binary:logistic', subsample=0.7, min_child_weight=1, reg_lambda=10, gamma=0.01, colsample_bytree=0.9, alpha=0.01)
        # AUC
        bst = XGBClassifier(scale_pos_weight = ratio, n_estimators=120, max_depth=6, learning_rate=0.1, objective='binary:logistic', subsample=0.6, min_child_weight=1, reg_lambda=0.01, gamma=0.2, colsample_bytree=1, alpha=0.01)
    elif args.tree_no == 90:
        bst = XGBClassifier(scale_pos_weight=ratio, n_estimators=80, max_depth=4, learning_rate=0.1, objective='binary:logistic', subsample=0.6, min_child_weight=1, reg_lambda=0.1, gamma=0.01, colsample_bytree=0.5, alpha=0.1)
    elif args.tree_no == 91:
        # AUC
        bst = XGBClassifier(scale_pos_weight=ratio, n_estimators=80, max_depth=4, learning_rate=0.05, objective='binary:logistic', subsample=0.7, min_child_weight=1, reg_lambda=0.001, gamma=0.2, colsample_bytree=0.5, alpha=1)
    elif args.tree_no == 92:
        # AUC
        #bst = XGBClassifier(scale_pos_weight=ratio, n_estimators=80, max_depth=10, learning_rate=0.2, objective='binary:logistic', subsample=0.8, min_child_weight=1, reg_lambda=1, gamma=0.01, colsample_bytree=0.6, alpha=0.1)
        # simpler tree
        bst = XGBClassifier(scale_pos_weight=ratio, n_estimators=5, max_depth=2, learning_rate=0.55, objective='binary:logistic', subsample=1, min_child_weight=7, reg_lambda=0.1, gamma=0.3, colsample_bytree=0.9, alpha=0.001)
    elif args.tree_no == 93:
        bst = XGBClassifier(scale_pos_weight = ratio, n_estimators=6, max_depth=4, learning_rate=0.3, objective='binary:logistic', subsample=1, min_child_weight=9, reg_lambda=0.1, gamma=0.2, colsample_bytree=0.8, alpha=0.001)
    elif args.tree_no == 94:
        #bst = XGBClassifier(scale_pos_weight = ratio, n_estimators=120, max_depth=3, learning_rate=0.25, objective='binary:logistic', subsample=0.9, min_child_weight=5, reg_lambda=1, gamma=0.5, colsample_bytree=0.6, alpha=0.01)
        # simpler tree
        #bst = XGBClassifier(scale_pos_weight = ratio, n_estimators=10, max_depth=3, learning_rate=0.3, objective='binary:logistic', subsample=0.9, min_child_weight=5, reg_lambda=1, gamma=0.01, colsample_bytree=0.9, alpha=0.1)
        #bst = XGBClassifier(scale_pos_weight = ratio, n_estimators=5, max_depth=3, learning_rate=0.4, objective='binary:logistic', subsample=0.9, min_child_weight=5, reg_lambda=1, gamma=0.01, colsample_bytree=0.9, alpha=0.1)
        # old
        bst = XGBClassifier(scale_pos_weight = ratio, n_estimators=6, max_depth=3, learning_rate=0.45, objective='binary:logistic', subsample=0.9, min_child_weight=5, reg_lambda=1, gamma=0.01, colsample_bytree=0.9, alpha=0.1, random_state=0)
        # new
        #bst = XGBClassifier(scale_pos_weight = ratio, n_estimators=6, max_depth=3, learning_rate=0.45, objective='binary:logistic', subsample=0.8, min_child_weight=7, reg_lambda=0.001, gamma=0.4, colsample_bytree=1, alpha=0.001)
        # conv
        #bst = XGBClassifier(scale_pos_weight = ratio, n_estimators=6, max_depth=3, learning_rate=0.6, objective='binary:logistic', subsample=1, min_child_weight=5, reg_lambda=0.001, gamma=0.1, colsample_bytree=0.8, alpha=1)
    elif args.tree_no == 153:
        bst = XGBClassifier(scale_pos_weight = ratio, n_estimators=6, max_depth=3, learning_rate=0.7, objective='binary:logistic', subsample=1, min_child_weight=5, reg_lambda=0.001, gamma=0.5, colsample_bytree=0.5, alpha=1, random_state=0)
    elif args.tree_no == 171:
        bst = XGBClassifier(scale_pos_weight = ratio, n_estimators=100, max_depth=3, learning_rate=0.1, objective='binary:logistic', subsample=0.6, min_child_weight=1, reg_lambda=1, gamma=0.2, colsample_bytree=0.5, alpha=0.01, random_state=0)
    elif args.tree_no == 172:
        bst = XGBClassifier(scale_pos_weight = ratio, n_estimators=100, max_depth=3, learning_rate=0.1, objective='binary:logistic', subsample=0.5, min_child_weight=1, reg_lambda=1, gamma=0.2, colsample_bytree=0.5, alpha=0.01, random_state=0)
    elif args.tree_no == 173:
        bst = XGBClassifier(scale_pos_weight = ratio, n_estimators=100, max_depth=5, learning_rate=0.1, objective='binary:logistic', subsample=0.8, min_child_weight=1, reg_lambda=0.01, gamma=0.1, colsample_bytree=0.6, alpha=0.01, random_state=0)

    # fit model
    bst.fit(X_train, y_train)
    cv_score = cross_val_score(bst, X_train,y_train, cv = RepeatedStratifiedKFold(n_splits = 5, n_repeats=40, random_state=0), scoring='roc_auc')
    print(f"cv score roc_auc mean: {cv_score.mean()}")
    print(f"cv score roc_auc std: {cv_score.std()}")
    cv_score = cross_val_score(bst, X_train,y_train, cv = RepeatedStratifiedKFold(n_splits = 5, n_repeats=40, random_state=0))
    print(f"cv score acc mean: {cv_score.mean()}")
    print(f"cv score acc std: {cv_score.std()}")
    cv_score = cross_val_score(bst, X_train,y_train, cv = RepeatedStratifiedKFold(n_splits = 5, n_repeats=40, random_state=0), scoring='f1')
    print(f"cv score f1 mean: {cv_score.mean()}")
    print(f"cv score f1 std: {cv_score.std()}")
    features = {"features": feature_names, "importance": bst.feature_importances_}
    df_features = pd.DataFrame(features)
    df_features = df_features[df_features["importance"] > 0].sort_values(by=["importance"], ascending=False)
    print(df_features.head(30))
    bst.get_booster().dump_model("xgb_decisions.txt", with_stats=True)
    with open("xgb_decisions.txt", "r") as f:
        txt_model = f.read()
    print(txt_model)
    # make predictions
    y_pred = bst.predict(X_test)
    print(classification_report(y_test, y_pred))
    print(f"roc_auc: {roc_auc_score(y_test, y_pred)}")
    print(f"f1: {f1_score(y_test, y_pred)}")

    shap_explainer = shap.Explainer(bst)
    shap_values = shap_explainer(X_train)
    shap.plots.beeswarm(shap_values)
    plt.tight_layout()
    plt.savefig("xgb_shap.png")
    plt.close()
    """
    # Tree 0
    shap.plots.scatter(shap_values[:, "L31bn1BsMax_gradmean0-5000"])
    plt.tight_layout()
    plt.savefig("L31bn1BsMax.png")
    plt.close()
    shap.plots.scatter(shap_values[:, "L31bn1BsMax_gradmean0-5000"], color=shap_values[:, "L21bn2WtMax_gradmean0-5000"])
    plt.tight_layout()
    plt.savefig("L31bn1BsMax_L21bn2WtMax.png")
    plt.close()
    shap.plots.scatter(shap_values[:, "L31bn1BsMax_gradmean0-5000"], color=shap_values[:, "L40bn2BsMax_gradmean0-5000"])
    plt.tight_layout()
    plt.savefig("L31bn1BsMax_L40bn2BsMax.png")
    plt.close()
    # Tree 1
    shap.plots.scatter(shap_values[:, "L31bn1BsMax_gradmean0-5000"], color=shap_values[:, "L11bn2BsMax_gradmean0-5000"])
    plt.tight_layout()
    plt.savefig("L31bn1BsMax_L11bn2BsMax.png")
    plt.close()
    shap.plots.scatter(shap_values[:, "L31bn1BsMax_gradmean0-5000"], color=shap_values[:, "L11bn1BsMin_gradmean0-5000"])
    plt.tight_layout()
    plt.savefig("L31bn1BsMax_L11bn1BsMin.png")
    plt.close()
    # Tree 2
    shap.plots.scatter(shap_values[:, "L21bn1BsMax_gradmean0-5000"], color=shap_values[:, "L21bn2BsMax_gradmean0-5000"])
    plt.tight_layout()
    plt.savefig("L21bn1BsMax_L21bn2BsMax.png")
    plt.close()
    shap.plots.scatter(shap_values[:, "L21bn1BsMax_gradmean0-5000"], color=shap_values[:, "bn1BsMean_gradmean0-5000"])
    plt.tight_layout()
    plt.savefig("L21bn1BsMax_bn1BsMean.png")
    plt.close()
    # Tree 3
    shap.plots.scatter(shap_values[:, "L20bn1BsMean_gradmean0-5000"], color=shap_values[:, "L30bn2BsMax_gradmean0-5000"])
    plt.tight_layout()
    plt.savefig("L20bn1BsMean_L30bn2BsMax.png")
    plt.close()
    # Tree 4
    shap.plots.scatter(shap_values[:, "bn1WtMean_gradmean0-5000"], color=shap_values[:, "L11bn1BsMin_gradmean0-5000"])
    plt.tight_layout()
    plt.savefig("bn1WtMean_L11bn1BsMin.png")
    plt.close()
    # Tree 5
    shap.plots.scatter(shap_values[:, "L10bn2WtMean_gradmean0-5000"], color=shap_values[:, "L21bn2BsMean_gradmean0-5000"])
    plt.tight_layout()
    plt.savefig("L10bn2WtMean_L21bn2BsMean.png")
    plt.close()
    shap.plots.scatter(shap_values[:, "L10bn2WtMean_gradmean0-5000"], color=shap_values[:, "L21bn2WtMax_gradmean0-5000"])
    plt.tight_layout()
    plt.savefig("L10bn2WtMean_L21bn2WtMax.png")
    plt.close()
    """

    """
    # Feature importance graph
    bst.plot_importance(model)
    plt.tight_layout()
    plt.savefig("xgb_importance.png")
    plt.close()
    """
    """
    # Nice looking tree graph
    for i in range(6):
        graph = xgb.to_graphviz(bst, num_trees=i)
        graph.render(f"xgb_tree_{i}")
    """


