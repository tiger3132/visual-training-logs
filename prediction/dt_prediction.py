from sklearn import tree
import os
from sklearn.tree import plot_tree, export_text 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RepeatedStratifiedKFold, RandomizedSearchCV
from sklearn import metrics
from sklearn.metrics import classification_report, roc_auc_score, f1_score, accuracy_score
import graphviz
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.feature_selection import SelectKBest, f_classif
from argparse import ArgumentParser
from sklearn.preprocessing import LabelEncoder
from columns_select import remove_table_col 

parser = ArgumentParser()
parser.add_argument("--mode", type=str, default='cv', choices=["cv", "test"])
parser.add_argument("--tree_no", type=int, default=0)

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
            elif col_name.startswith("0_l") or col_name.startswith("P1_l") or col_name.startswith("2_l") or col_name.startswith("3_l"):
                layer += "L" + col_name[3:5] 
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
"""
all_df_sgd = all_df_sgd.loc[:,~all_df_sgd.columns.duplicated()].copy()
all_df_sgd = all_df_sgd.drop(columns=["Unnamed: 0"]) # Remove unnamed column
all_df_adam = all_df_adam.loc[:,~all_df_adam.columns.duplicated()].copy()
all_df_adam = all_df_adam.drop(columns=["Unnamed: 0"]) # Remove unnamed column
"""
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

"""
fig, ax = plt.subplots()
ax.scatter(df_dt_pred["P0_l41_bn2_w"], df_dt_pred["test_acc"], c=df_dt_pred["test_acc"]>0.7)
plt.xlabel("Normalized gradient of STD")
plt.ylabel("test_acc")
plt.savefig("P0_l41_bn2_w.png")

fig, ax = plt.subplots()
ax.scatter(df_dt_pred["P0_l10_cn1"], df_dt_pred["test_acc"])
plt.xlabel("Mean of mean")
plt.ylabel("test_acc")
plt.savefig("P0_l10_cn1.png")

fig, ax = plt.subplots()
ax.scatter(df_dt_pred["P3_bn1_w_mx"], df_dt_pred["test_acc"])
plt.xlabel("Normalized Maximum")
plt.ylabel("test_acc")
plt.savefig("P3_bn1_w_mx.png")

fig, ax = plt.subplots()
ax.scatter(df_dt_pred["P3_l41_cn2_w"], df_dt_pred["test_acc"])
plt.xlabel("Mean of std")
plt.ylabel("test_acc")
"""
print("columns:")
X, y = all_df.loc[:,all_df.columns != 'test_acc'], all_df["test_acc"] > 0.7
#X_adam, y_adam = all_df_adam.loc[:,all_df_adam.columns != 'test_acc'], all_df_adam["test_acc"] > 0.7
#X_sgd, y_sgd = all_df_sgd.loc[:,all_df_sgd.columns != 'test_acc'], all_df_sgd["test_acc"] > 0.7
target_names = ["< 0.7", "> 0.7"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0) #1
feature_names = X_train.columns
#X_train_adam, X_test_adam, y_train_adam, y_test_adam = train_test_split(X_adam, y_adam, test_size=0.2, random_state=1)
#X_train_sgd, X_test_sgd, y_train_sgd, y_test_sgd = train_test_split(X_sgd, y_sgd, test_size=0.2, random_state=1)


for col in X_train.columns:
    print(col)

print(f"Total number of y is {len(y)}")
unique, counts = np.unique(y, return_counts=True)
print(dict(zip(unique, counts)))

print(f"Total number of training instance is {len(X_train)}")
print(f"Total number of testing instance is {len(X_test)}")

print(X)


if args.mode == "cv":

    print("-----------------------------------------------------")
    print("random search cv")
    dt = tree.DecisionTreeClassifier(class_weight="balanced", random_state=0)

    #param_grid = [
    #        {"criterion": ["gini", "entropy", "log_loss"], "splitter": ["best", "random"], "max_depth": [None, 1, 2, 3, 4], "min_samples_split": [2, 5, 10, 15], "min_samples_leaf": [1, 5, 10, 15], "max_features": [None, "sqrt", "log2"], "min_impurity_decrease": [0,  0.04, 0.08, 0.1]}
    #]

    param_grid = [
            {"criterion": ["gini", "entropy", "log_loss"], "splitter": ["best"], "max_depth": [None, 1, 2, 3, 4, 8, 16, 32], "min_samples_split": [2,4,5,8,10,12], "min_samples_leaf": [1,3,5,8,10,12], "max_features": [None, "sqrt", "log2"], "min_impurity_decrease": [0,0.02,0.04,0.08,0.1,0.12,0.2]}
    ]
    cv = RepeatedStratifiedKFold(n_splits = 5, n_repeats=30, random_state=0) # 40 for validloss
    search = RandomizedSearchCV(estimator=dt, param_distributions=param_grid, cv=cv, n_iter=2100, n_jobs=-1, random_state=0, scoring='roc_auc') #5500 for validloss, 2100 for others
    #search = GridSearchCV(estimator=dt, param_grid=param_grid, cv=cv, n_jobs=-1, random_state=0)
    search.fit(X_train,y_train)
    results_df = pd.DataFrame(search.cv_results_)
    results_df = results_df.sort_values(by=["rank_test_score"])
    print(results_df[["params", "mean_test_score", "std_test_score"]].head(25))
    #results_df.to_csv(f"dt_hp{args.hp}_p0to5000{args.p0to5000}_p5000to10000{args.p5000to10000}_p10000to15000{args.p10000to15000}_p15000to20000{args.p15000to20000}_t0to5000{args.trainloss0to5000}_t5000to10000{args.trainloss5000to10000}_t10000to15000{args.trainloss10000to15000}_t15000to20000{args.trainloss15000to20000}_v0to5000{args.validloss0to5000}_v5000to10000{args.validloss5000to10000}_v10000to15000{args.validloss10000to15000}_v15000to20000{args.validloss15000to20000}_gdnrm{args.gradnorm}_gdmn{args.gradmean}_nrm{args.norm}_mn{args.mean}.csv")
    results_df.to_csv(f"dt_fl_1{args.first_layer_1}_2{args.first_layer_2}_3{args.first_layer_3}_4{args.first_layer_4}_ml_1{args.middle_layer_1}_2{args.middle_layer_2}_3{args.middle_layer_3}_4{args.middle_layer_4}_ll_1{args.last_layer_1}_2{args.last_layer_2}_3{args.last_layer_3}_4{args.last_layer_4}_batchnorm{args.batch_norm}_conv{args.conv}_beforerelu{args.before_relu}_afterrelu{args.after_relu}_downsample{args.downsample}_gradnorm{args.gradnorm}_gradmean{args.gradmean}_gradpercent{args.gradpercent}.csv")
elif args.mode == "test":
    print("-----------------------------------------------------")
    # 0
    if args.tree_no == 0:
        clf = tree.DecisionTreeClassifier(class_weight="balanced", criterion="log_loss", max_depth=2, max_features=None, splitter="best", min_impurity_decrease=0, min_samples_leaf=8, min_samples_split=12, random_state=0)
    elif args.tree_no == 1:
        #clf = tree.DecisionTreeClassifier(class_weight="balanced", criterion="entropy", max_depth=1, max_features=None, splitter="best", min_impurity_decrease=0, min_samples_leaf=5, min_samples_split=2, random_state=0)
        clf = tree.DecisionTreeClassifier(class_weight="balanced", criterion="gini", max_depth=8, max_features=None, splitter="best", min_impurity_decrease=0, min_samples_leaf=1, min_samples_split=2, random_state=0)
    elif args.tree_no == 2:
        clf = tree.DecisionTreeClassifier(class_weight="balanced", criterion="entropy", max_depth=1, max_features=None, splitter="best", min_impurity_decrease=0, min_samples_leaf=10, min_samples_split=8, random_state=0)
    elif args.tree_no == 3:
        clf = tree.DecisionTreeClassifier(class_weight="balanced", criterion="log_loss", max_depth=8, max_features=None, splitter="best", min_impurity_decrease=0, min_samples_leaf=3, min_samples_split=5, random_state=0)
    elif args.tree_no == 4:
        # Acc
        #clf = tree.DecisionTreeClassifier(class_weight="balanced", criterion="log_loss", max_depth=16, max_features=None, splitter="best", min_impurity_decrease=0, min_samples_leaf=1, min_samples_split=2, random_state=0)
        # AUC
        #clf = tree.DecisionTreeClassifier(class_weight="balanced", criterion="entropy", max_depth=4, max_features=None, splitter="best", min_impurity_decrease=0.02, min_samples_leaf=12, min_samples_split=8, random_state=0)
        clf = tree.DecisionTreeClassifier(class_weight="balanced", criterion="gini", max_depth=4, max_features=None, splitter="best", min_impurity_decrease=0, min_samples_leaf=12, min_samples_split=8, random_state=0)
    elif args.tree_no == 5:
        clf = tree.DecisionTreeClassifier(class_weight="balanced", criterion="log_loss", max_depth=8, max_features=None, splitter="best", min_impurity_decrease=0, min_samples_leaf=10, min_samples_split=5, random_state=0)
    elif args.tree_no == 6:
        clf = tree.DecisionTreeClassifier(class_weight="balanced", criterion="entropy", max_depth=4, max_features=None, splitter="best", min_impurity_decrease=0.02, min_samples_leaf=10, min_samples_split=8, random_state=0)
    elif args.tree_no == 7:
        clf = tree.DecisionTreeClassifier(class_weight="balanced", criterion="gini", max_depth=3, max_features=None, splitter="best", min_impurity_decrease=0.02, min_samples_leaf=5, min_samples_split=8, random_state=0)
    elif args.tree_no == 8:
        clf = tree.DecisionTreeClassifier(class_weight="balanced", criterion="entropy", max_depth=2, max_features=None, splitter="best", min_impurity_decrease=0, min_samples_leaf=10, min_samples_split=10, random_state=0)
    elif args.tree_no == 9:
        clf = tree.DecisionTreeClassifier(class_weight="balanced", criterion="entropy", max_depth=8, max_features=None, splitter="best", min_impurity_decrease=0.02, min_samples_leaf=1, min_samples_split=12, random_state=0)
    elif args.tree_no == 10:
        clf = tree.DecisionTreeClassifier(class_weight="balanced", criterion="gini", max_depth=3, max_features=None, splitter="best", min_impurity_decrease=0, min_samples_leaf=3, min_samples_split=2, random_state=0)
    elif args.tree_no == 11:
        clf = tree.DecisionTreeClassifier(class_weight="balanced", criterion="gini", max_depth=3, max_features=None, splitter="best", min_impurity_decrease=0.02, min_samples_leaf=5, min_samples_split=4, random_state=0)
    elif args.tree_no == 12:
        clf = tree.DecisionTreeClassifier(class_weight="balanced", criterion="log_loss", max_depth=4, max_features=None, splitter="best", min_impurity_decrease=0.04, min_samples_leaf=3, min_samples_split=2, random_state=0)
    elif args.tree_no == 13:
        clf = tree.DecisionTreeClassifier(class_weight="balanced", criterion="gini", max_depth=None, max_features=None, splitter="best", min_impurity_decrease=0, min_samples_leaf=1, min_samples_split=5, random_state=0)
    elif args.tree_no == 14:
        clf = tree.DecisionTreeClassifier(class_weight="balanced", criterion="entropy", max_depth=4, max_features=None, splitter="best", min_impurity_decrease=0.04, min_samples_leaf=3, min_samples_split=2, random_state=0)
    elif args.tree_no == 15:
        clf = tree.DecisionTreeClassifier(class_weight="balanced", criterion="log_loss", max_depth=1, max_features=None, splitter="best", min_impurity_decrease=0.1, min_samples_leaf=1, min_samples_split=5, random_state=0)
    elif args.tree_no == 16:
        clf = tree.DecisionTreeClassifier(class_weight="balanced", criterion="entropy", max_depth=1, max_features=None, splitter="best", min_impurity_decrease=0, min_samples_leaf=1, min_samples_split=5, random_state=0)
    elif args.tree_no == 51:
        clf = tree.DecisionTreeClassifier(class_weight="balanced", criterion="log_loss", max_depth=4, max_features="sqrt", splitter="best", min_impurity_decrease=0.04, min_samples_leaf=1, min_samples_split=12, random_state=0)
    elif args.tree_no == 52:
        #clf = tree.DecisionTreeClassifier(class_weight="balanced", criterion="entropy", max_depth=3, max_features="sqrt", splitter="best", min_impurity_decrease=0, min_samples_leaf=5, min_samples_split=2, random_state=0)
        # AUC
        clf = tree.DecisionTreeClassifier(class_weight="balanced", criterion="gini", max_depth=4, max_features=None, splitter="best", min_impurity_decrease=0, min_samples_leaf=12, min_samples_split=8, random_state=0)
    elif args.tree_no == 53:
        #clf = tree.DecisionTreeClassifier(class_weight="balanced", criterion="gini", max_depth=2, max_features=None, splitter="best", min_impurity_decrease=0, min_samples_leaf=12, min_samples_split=4, random_state=0)
        # AUC
        clf = tree.DecisionTreeClassifier(class_weight="balanced", criterion="gini", max_depth=3, max_features=None, splitter="best", min_impurity_decrease=0, min_samples_leaf=8, min_samples_split=12, random_state=0)

    elif args.tree_no == 54:
        #clf = tree.DecisionTreeClassifier(class_weight="balanced", criterion="gini", max_depth=1, max_features=None, splitter="best", min_impurity_decrease=0.04, min_samples_leaf=1, min_samples_split=5, random_state=0)
        # AUC
        clf = tree.DecisionTreeClassifier(class_weight="balanced", criterion="gini", max_depth=4, max_features=None, splitter="best", min_impurity_decrease=0, min_samples_leaf=12, min_samples_split=8, random_state=0)
    elif args.tree_no == 55:
        #clf = tree.DecisionTreeClassifier(class_weight="balanced", criterion="gini", max_depth=2, max_features=None, splitter="best", min_impurity_decrease=0.04, min_samples_leaf=5, min_samples_split=8, random_state=0)
        # AUC
        clf = tree.DecisionTreeClassifier(class_weight="balanced", criterion="gini", max_depth=16, max_features="log2", splitter="best", min_impurity_decrease=0, min_samples_leaf=12, min_samples_split=2, random_state=0)
    elif args.tree_no == 56:
        #clf = tree.DecisionTreeClassifier(class_weight="balanced", criterion="gini", max_depth=1, max_features=None, splitter="best", min_impurity_decrease=0.04, min_samples_leaf=1, min_samples_split=5, random_state=0)
        # AUC
        clf = tree.DecisionTreeClassifier(class_weight="balanced", criterion="gini", max_depth=4, max_features=None, splitter="best", min_impurity_decrease=0, min_samples_leaf=12, min_samples_split=8, random_state=0)
    elif args.tree_no == 57:
        clf = tree.DecisionTreeClassifier(class_weight="balanced", criterion="log_loss", max_depth=4, max_features="sqrt", splitter="best", min_impurity_decrease=0, min_samples_leaf=8, min_samples_split=12, random_state=0)
    elif args.tree_no == 58:
        clf = tree.DecisionTreeClassifier(class_weight="balanced", criterion="entropy", max_depth=8, max_features=None, splitter="best", min_impurity_decrease=0.04, min_samples_leaf=3, min_samples_split=10, random_state=0)
    elif args.tree_no == 59:
        clf = tree.DecisionTreeClassifier(class_weight="balanced", criterion="gini", max_depth=4, max_features=None, splitter="best", min_impurity_decrease=0.02, min_samples_leaf=8, min_samples_split=10, random_state=0)
    elif args.tree_no == 60:
        clf = tree.DecisionTreeClassifier(class_weight="balanced", criterion="gini", max_depth=3, max_features="log2", splitter="best", min_impurity_decrease=0, min_samples_leaf=1, min_samples_split=10, random_state=0)
    elif args.tree_no == 61:
        clf = tree.DecisionTreeClassifier(class_weight="balanced", criterion="entropy", max_depth=8, max_features=None, splitter="best", min_impurity_decrease=0.02, min_samples_leaf=3, min_samples_split=5, random_state=0)
    elif args.tree_no == 62:
        clf = tree.DecisionTreeClassifier(class_weight="balanced", criterion="log_loss", max_depth=8, max_features=None, splitter="best", min_impurity_decrease=0.04, min_samples_leaf=3, min_samples_split=2, random_state=0)
    elif args.tree_no == 63:
        clf = tree.DecisionTreeClassifier(class_weight="balanced", criterion="log_loss", max_depth=4, max_features="sqrt", splitter="best", min_impurity_decrease=0.02, min_samples_leaf=1, min_samples_split=10, random_state=0)
    elif args.tree_no == 64:
        clf = tree.DecisionTreeClassifier(class_weight="balanced", criterion="gini", max_depth=4, max_features=None, splitter="best", min_impurity_decrease=0.02, min_samples_leaf=12, min_samples_split=8, random_state=0)
    elif args.tree_no == 65:
        clf = tree.DecisionTreeClassifier(class_weight="balanced", criterion="entropy", max_depth=8, max_features=None, splitter="best", min_impurity_decrease=0, min_samples_leaf=12, min_samples_split=10, random_state=0)
    elif args.tree_no == 66:
        clf = tree.DecisionTreeClassifier(class_weight="balanced", criterion="gini", max_depth=2, max_features=None, splitter="best", min_impurity_decrease=0.1, min_samples_leaf=8, min_samples_split=2, random_state=0)
    elif args.tree_no == 67:
        clf = tree.DecisionTreeClassifier(class_weight="balanced", criterion="gini", max_depth=4, max_features="log2", splitter="best", min_impurity_decrease=0.02, min_samples_leaf=1, min_samples_split=8, random_state=0)
    elif args.tree_no == 68:
        clf = tree.DecisionTreeClassifier(class_weight="balanced", criterion="gini", max_depth=1, max_features=None, splitter="best", min_impurity_decrease=0.2, min_samples_leaf=3, min_samples_split=5, random_state=0)
    elif args.tree_no == 69:
        clf = tree.DecisionTreeClassifier(class_weight="balanced", criterion="entropy", max_depth=4, max_features=None, splitter="best", min_impurity_decrease=0.02, min_samples_leaf=1, min_samples_split=5, random_state=0)
    elif args.tree_no == 70:
        clf = tree.DecisionTreeClassifier(class_weight="balanced", criterion="log_loss", max_depth=8, max_features=None, splitter="best", min_impurity_decrease=0, min_samples_leaf=1, min_samples_split=10, random_state=0)
    elif args.tree_no == 71:
        clf = tree.DecisionTreeClassifier(class_weight="balanced", criterion="gini", max_depth=32, max_features="log2", splitter="best", min_impurity_decrease=0, min_samples_leaf=3, min_samples_split=12, random_state=0)
    elif args.tree_no == 72:
        clf = tree.DecisionTreeClassifier(class_weight="balanced", criterion="gini", max_depth=16, max_features="log2", splitter="best", min_impurity_decrease=0, min_samples_leaf=5, min_samples_split=4, random_state=0)
    elif args.tree_no == 73:
        clf = tree.DecisionTreeClassifier(class_weight="balanced", criterion="log_loss", max_depth=8, max_features="sqrt", splitter="best", min_impurity_decrease=0.02, min_samples_leaf=3, min_samples_split=12, random_state=0)
    elif args.tree_no == 74:
        clf = tree.DecisionTreeClassifier(class_weight="balanced", criterion="entropy", max_depth=4, max_features=None, splitter="best", min_impurity_decrease=0.04, min_samples_leaf=5, min_samples_split=4, random_state=0)
    elif args.tree_no == 75:
        clf = tree.DecisionTreeClassifier(class_weight="balanced", criterion="gini", max_depth=4, max_features="sqrt", splitter="best", min_impurity_decrease=0, min_samples_leaf=3, min_samples_split=4, random_state=0)
    elif args.tree_no == 76:
        clf = tree.DecisionTreeClassifier(class_weight="balanced", criterion="gini", max_depth=4, max_features="log2", splitter="best", min_impurity_decrease=0.02, min_samples_leaf=1, min_samples_split=4, random_state=0)
    elif args.tree_no == 77:
        clf = tree.DecisionTreeClassifier(class_weight="balanced", criterion="gini", max_depth=2, max_features=None, splitter="best", min_impurity_decrease=0.04, min_samples_leaf=10, min_samples_split=4, random_state=0)
    elif args.tree_no == 78:
        clf = tree.DecisionTreeClassifier(class_weight="balanced", criterion="gini", max_depth=3, max_features=None, splitter="best", min_impurity_decrease=0.04, min_samples_leaf=5, min_samples_split=8, random_state=0)
    elif args.tree_no == 79:
        clf = tree.DecisionTreeClassifier(class_weight="balanced", criterion="gini", max_depth=8, max_features="log2", splitter="best", min_impurity_decrease=0, min_samples_leaf=3, min_samples_split=10, random_state=0)
    elif args.tree_no == 80:
        clf = tree.DecisionTreeClassifier(class_weight="balanced", criterion="log_loss", max_depth=8, max_features=None, splitter="best", min_impurity_decrease=0, min_samples_leaf=1, min_samples_split=10, random_state=0)
        #clf = tree.DecisionTreeClassifier(class_weight="balanced", criterion="log_loss", max_depth=8, max_features=None, splitter="best", min_impurity_decrease=0, min_samples_leaf=1, min_samples_split=12, random_state=0)
    elif args.tree_no == 81:
        clf = tree.DecisionTreeClassifier(class_weight="balanced", criterion="gini", max_depth=32, max_features="log2", splitter="best", min_impurity_decrease=0.02, min_samples_leaf=1, min_samples_split=5, random_state=0)
    elif args.tree_no == 82:
        clf = tree.DecisionTreeClassifier(class_weight="balanced", criterion="log_loss", max_depth=16, max_features="log2", splitter="best", min_impurity_decrease=0, min_samples_leaf=3, min_samples_split=2, random_state=0)
    elif args.tree_no == 84:
        clf = tree.DecisionTreeClassifier(class_weight="balanced", criterion="entropy", max_depth=8, max_features=None, splitter="best", min_impurity_decrease=0.04, min_samples_leaf=5, min_samples_split=4, random_state=0)
    elif args.tree_no == 85:
        clf = tree.DecisionTreeClassifier(class_weight="balanced", criterion="entropy", max_depth=4, max_features=None, splitter="best", min_impurity_decrease=0, min_samples_leaf=5, min_samples_split=12, random_state=0)
    elif args.tree_no == 86:
        clf = tree.DecisionTreeClassifier(class_weight="balanced", criterion="gini", max_depth=2, max_features=None, splitter="best", min_impurity_decrease=0.02, min_samples_leaf=10, min_samples_split=12, random_state=0)
    elif args.tree_no == 87:
        clf = tree.DecisionTreeClassifier(class_weight="balanced", criterion="gini", max_depth=8, max_features=None, splitter="best", min_impurity_decrease=0, min_samples_leaf=12, min_samples_split=5, random_state=0)
    elif args.tree_no == 88:
        clf = tree.DecisionTreeClassifier(class_weight="balanced", criterion="entropy", max_depth=8, max_features=None, splitter="best", min_impurity_decrease=0, min_samples_leaf=3, min_samples_split=5, random_state=0)
    elif args.tree_no == 88:
        clf = tree.DecisionTreeClassifier(class_weight="balanced", criterion="entropy", max_depth=8, max_features=None, splitter="best", min_impurity_decrease=0, min_samples_leaf=3, min_samples_split=5, random_state=0)
    elif args.tree_no == 110:
        #clf = tree.DecisionTreeClassifier(class_weight="balanced", criterion="entropy", max_depth=4, max_features=None, splitter="best", min_impurity_decrease=0.04, min_samples_leaf=8, min_samples_split=8, random_state=0)
        clf = tree.DecisionTreeClassifier(class_weight="balanced", criterion="entropy", max_depth=8, max_features=None, splitter="best", min_impurity_decrease=0.04, min_samples_leaf=8, min_samples_split=10, random_state=0)
    elif args.tree_no == 111:
        clf = tree.DecisionTreeClassifier(class_weight="balanced", criterion="log_loss", max_depth=32, max_features=None, splitter="best", min_impurity_decrease=0, min_samples_leaf=12, min_samples_split=2, random_state=0)
    elif args.tree_no == 113:
        clf = tree.DecisionTreeClassifier(class_weight="balanced", criterion="gini", max_depth=4, max_features="log2", splitter="best", min_impurity_decrease=0, min_samples_leaf=8, min_samples_split=8, random_state=0)
    elif args.tree_no == 115:
        clf = tree.DecisionTreeClassifier(class_weight="balanced", criterion="entropy", max_depth=8, max_features="log2", splitter="best", min_impurity_decrease=0, min_samples_leaf=12, min_samples_split=5, random_state=0)
    elif args.tree_no == 160:
        clf = tree.DecisionTreeClassifier(class_weight="balanced", criterion="gini", max_depth=8, max_features=None, splitter="best", min_impurity_decrease=0, min_samples_leaf=8, min_samples_split=8, random_state=0)
    cv_score = cross_val_score(clf, X_train,y_train, cv = RepeatedStratifiedKFold(n_splits = 5, n_repeats=40, random_state=0), scoring='roc_auc')
    print(f"cv score roc mean: {cv_score.mean()}")
    print(f"cv score roc std: {cv_score.std()}")
    cv_score = cross_val_score(clf, X_train,y_train, cv = RepeatedStratifiedKFold(n_splits = 5, n_repeats=40, random_state=0))
    print(f"cv score mean: {cv_score.mean()}")
    print(f"cv score std: {cv_score.std()}")
    cv_score = cross_val_score(clf, X_train,y_train, cv = RepeatedStratifiedKFold(n_splits = 5, n_repeats=40, random_state=0), scoring='f1')
    print(f"cv score f1 mean: {cv_score.mean()}")
    print(f"cv score f1 std: {cv_score.std()}")
    clf = clf.fit(X_train, y_train)
    features = {"features": feature_names, "importance": clf.feature_importances_}
    df_features = pd.DataFrame(features)
    df_features = df_features[df_features["importance"] > 0].sort_values(by=["importance"], ascending=False)
    print(df_features)
    r = export_text(clf, feature_names=feature_names, class_names=target_names)
    #print(r)
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))
    print(f"accuracy: {accuracy_score(y_test, y_pred)}")
    print(f"f1: {f1_score(y_test, y_pred)}")
    print(f"roc_auc: {roc_auc_score(y_test, y_pred)}")

    # Simple tree graph
    #plot_tree(clf)
    #plt.savefig("dt_prediction_entropy_d1.png")
    #plt.close()

    # Nicer looking tree graph
    #dot_data = tree.export_graphviz(clf, out_file=None, feature_names=feature_names, class_names=target_names) 
    #graph = graphviz.Source(dot_data)
    #graph.render(f"dt_prediction_number_{args.tree_no}")

"""
print(all_df["bn1WtMean_mean5000-10000"])
print(all_df["test_acc"].shape)
fig, ax = plt.subplots()
ax.scatter(all_df.iloc[:, 485], all_df["test_acc"], c=all_df.iloc[:, 485] <= 0.962)
plt.xlabel("bn1WtMean_mean5000-10000")
plt.ylabel("test_acc")
plt.savefig("bn1WtMean_mean5000-10000.png")
"""
"""
X_sub1 = X_train[["P0_l10_cn1","P0_l41_bn2_w"]]
clf_sub1 = tree.DecisionTreeClassifier(max_depth=2).fit(X_sub1, y_train)
r = export_text(clf_sub1)
print(r)
_, ax = plt.subplots()
x_min, x_max, y_min, y_max = -0.25, 0.25, -0.25, 0.25
ax.set(xlim=(x_min, x_max), ylim=(y_min, y_max))
DecisionBoundaryDisplay.from_estimator(
        clf_sub1,
        X_sub1,
        ax=ax,
        response_method="predict",
        xlabel="Mean of 0-5000 timestep means of layer1.0.conv1 weights",
        ylabel="L2 Norm of grad. of 0-5000 timestep std of layer 4.1 bn2 weights",
        alpha=0.5,
)
scatter = ax.scatter(X_sub1["P0_l10_cn1"], X_sub1["P0_l41_bn2_w"], c=y_train, edgecolor="black", s=12)
ax.legend(scatter.legend_elements()[0],target_names, loc="lower right", borderpad=0, handletextpad=0)
_ = plt.savefig("decision_bound1_new.png")
_ = plt.close()
X_sub2 = X_train[["P0_l10_cn1","P3_l41_cn2_w"]]
clf_sub2 = tree.DecisionTreeClassifier(max_depth=2).fit(X_sub2, y_train)
r = export_text(clf_sub2)
print(r)
plot_tree(clf_sub2)
plt.savefig("test.png")
plt.close()
_, ax = plt.subplots()
x_min, x_max, y_min, y_max = -0.1, 0.2, -0.5, 1.5
ax.set(xlim=(x_min, x_max), ylim=(y_min, y_max))
DecisionBoundaryDisplay.from_estimator(
        clf_sub2,
        X_sub2,
        ax=ax,
        grid_resolution=200,
        response_method="predict",
        xlabel="Mean of 0-5000 timestep means of layer1.0.conv1 weights",
        ylabel="Mean of 10000-15000 timestep std of layer4.1.conv2 weights",
        alpha=0.5,
)
scatter = ax.scatter(X_sub2["P0_l10_cn1"], X_sub2["P3_l41_cn2_w"], c=y_train, edgecolor="black", s=12)
ax.legend(scatter.legend_elements()[0], target_names, loc="upper right", borderpad=0, handletextpad=0)
_ = plt.savefig("decision_bound2_new.png")
_ = plt.close()
"""
