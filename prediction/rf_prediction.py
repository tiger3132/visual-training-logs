import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split, RepeatedStratifiedKFold, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, f1_score
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from sklearn.preprocessing import LabelEncoder

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
param_df = pd.read_csv("params_0to20000.csv")
print(len(param_df))
raise NotImplementedError
hp_df = pd.read_csv("hp_df_no_outlier.csv")
train_val_loss_df = pd.read_csv("df_train_val_loss_no_outlier.csv")

all_df = pd.concat([param_df, hp_df, train_val_loss_df], axis=1)
#all_df = all_df.T.drop_duplicates().T # Remove duplicate test accuracy columns
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
"""

columns_to_remove = remove_table_col(all_df, args.hp, args.p0to5000, args.p5000to10000, args.p10000to15000, args.p15000to20000, args.trainloss0to5000, args.trainloss5000to10000, args.trainloss10000to15000, args.trainloss15000to20000, args.validloss0to5000, args.validloss5000to10000, args.validloss10000to15000, args.validloss15000to20000, args.batch_norm, args.conv, args.before_relu, args.after_relu, args.downsample, args.first_layer_1, args.first_layer_2, args.first_layer_3, args.first_layer_4, args.last_layer_1, args.last_layer_2, args.last_layer_3, args.last_layer_4, args.middle_layer_1, args.middle_layer_2, args.middle_layer_3, args.middle_layer_4, args.gradnorm, args.gradmean, args.gradpercent, args.mean, args.norm) 

for col in columns_to_remove:
    all_df = all_df.drop(columns=col)

le = LabelEncoder()
for col in all_df.columns:
    if all_df[col].dtype == "object":
        all_df[col] = le.fit_transform(all_df[col])

X, y = all_df.loc[:,all_df.columns != 'test_acc'], all_df["test_acc"] > 0.7
#X_adam, y_adam = all_df_adam.loc[:,all_df_adam.columns != 'test_acc'], all_df_adam["test_acc"] > 0.7
#X_sgd, y_sgd = all_df_sgd.loc[:,all_df_sgd.columns != 'test_acc'], all_df_sgd["test_acc"] > 0.7
feature_names = X.columns
target_names = ["< 0.7", "> 0.7"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
#X_train_adam, X_test_adam, y_train_adam, y_test_adam = train_test_split(X_adam, y_adam, test_size=0.2, random_state=1)
#X_train_sgd, X_test_sgd, y_train_sgd, y_test_sgd = train_test_split(X_sgd, y_sgd, test_size=0.2, random_state=1)
print("-----------------------------------------------------")
print("features")
for f in X_train.columns:
    print(f)

if args.mode == "cv":
    print("-----------------------------------------------------")
    print("random search cv")

    rf = RandomForestClassifier(random_state=0, class_weight="balanced", n_jobs=-1)

    param_grid = [
            {'bootstrap': [True, False], 'n_estimators': [2, 3, 4, 5, 6, 7, 8, 10, 25, 50, 100, 125], "criterion": ["gini", "entropy", "log_loss"], "max_depth": [None,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,40], "min_samples_split": [2, 4, 5, 8, 10, 12, 16, 20], "min_samples_leaf": [1, 3, 5, 6, 9, 10, 12, 15, 20], "max_features": [None, "sqrt", "log2"], "min_impurity_decrease": [0,0.02,0.04,0.08,0.1,0.12,0.2]}
    ]
    cv = RepeatedStratifiedKFold(n_splits = 5, n_repeats=25, random_state=0)
    search = RandomizedSearchCV(estimator=rf, param_distributions=param_grid, cv=cv, n_iter=1700, n_jobs=-1, scoring='roc_auc')
    search.fit(X_train,y_train)
    results_df = pd.DataFrame(search.cv_results_)
    results_df = results_df.sort_values(by=["rank_test_score"])
    print(results_df[["params", "rank_test_score", "mean_test_score", "std_test_score"]].head(25))
    #results_df.to_csv(f"rf_hp{args.hp}_p0to5000{args.p0to5000}_p5000to10000{args.p5000to10000}_p10000to15000{args.p10000to15000}_p15000to20000{args.p15000to20000}_t0to5000{args.trainloss0to5000}_t5000to10000{args.trainloss5000to10000}_t10000to15000{args.trainloss10000to15000}_t15000to20000{args.trainloss15000to20000}_v0to5000{args.validloss0to5000}_v5000to10000{args.validloss5000to10000}_v10000to15000{args.validloss10000to15000}_v15000to20000{args.validloss15000to20000}.csv")
    results_df.to_csv(f"rf_fl_1{args.first_layer_1}_2{args.first_layer_2}_3{args.first_layer_3}_4{args.first_layer_4}_ml_1{args.middle_layer_1}_2{args.middle_layer_2}_3{args.middle_layer_3}_4{args.middle_layer_4}_ll_1{args.last_layer_1}_2{args.last_layer_2}_3{args.last_layer_3}_4{args.last_layer_4}_batchnorm{args.batch_norm}_conv{args.conv}_beforerelu{args.before_relu}_afterrelu{args.after_relu}_downsample{args.downsample}_gradnorm{args.gradnorm}_gradmean{args.gradmean}_gradpercent{args.gradpercent}.csv")
elif args.mode == "test":
    print("-----------------------------------------------------")
    print("test")
    if args.tree_no == 38:
        clf = RandomForestClassifier(class_weight="balanced", n_estimators=50, min_samples_split=20, min_impurity_decrease=0.02, min_samples_leaf=6, max_features= 'sqrt', max_depth=14, criterion= 'log_loss', bootstrap= True, random_state=0)
    elif args.tree_no == 119:
        clf = RandomForestClassifier(class_weight="balanced", n_estimators=50, min_samples_split=5, min_impurity_decrease=0, min_samples_leaf=5, max_features= 'sqrt', max_depth=14, criterion= 'entropy', bootstrap= True, random_state=0)
    elif args.tree_no == 123:
        clf = RandomForestClassifier(class_weight="balanced", n_estimators=50, min_samples_split=12, min_impurity_decrease=0, min_samples_leaf=9, max_features= 'sqrt', max_depth=14, criterion= 'entropy', bootstrap= False, random_state=0)
    elif args.tree_no == 167:
        clf = RandomForestClassifier(class_weight="balanced", n_estimators=50, min_samples_split=5, min_impurity_decrease=0.02, min_samples_leaf=6, max_features= 'sqrt', max_depth=14, criterion= 'log_loss', bootstrap= False, random_state=0)
    cv_score = cross_val_score(clf, X_train,y_train, cv = RepeatedStratifiedKFold(n_splits = 5, n_repeats=40, random_state=0), scoring='roc_auc')
    print(f"cv score auc mean: {cv_score.mean()}")
    print(f"cv score auc std: {cv_score.std()}")
    cv_score = cross_val_score(clf, X_train,y_train, cv = RepeatedStratifiedKFold(n_splits = 5, n_repeats=40, random_state=0))
    print(f"cv score acc mean: {cv_score.mean()}")
    print(f"cv score acc std: {cv_score.std()}")
    cv_score = cross_val_score(clf, X_train,y_train, cv = RepeatedStratifiedKFold(n_splits = 5, n_repeats=40, random_state=0), scoring='f1')
    print(f"cv score f1 mean: {cv_score.mean()}")
    print(f"cv score f1 std: {cv_score.std()}")
    clf = clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    #features = {"features": feature_names, "importance": clf.feature_importances_}
    #df_features = pd.DataFrame(features)
    #df_features = df_features[df_features["importance"] > 0].sort_values(by=["importance"], ascending=False)
    #print(df_features.head(30))
    #print(r)
    # make predictions
    print(classification_report(y_test, y_pred))
    print(f"roc_auc: {roc_auc_score(y_test, y_pred)}")
    print(f"f1: {f1_score(y_test, y_pred)}")

    feature_importances = pd.Series(clf.feature_importances_, index=X_train.columns).sort_values(ascending=False)
    print(feature_importances)
    feature_importances = feature_importances[feature_importances != 0.000000]
    print(feature_importances)
    """
    feature_importances.plot.bar();
    plt.tight_layout()
    plt.savefig("rf_importance.png")
    plt.close()
    """
