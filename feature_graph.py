from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from argparse import ArgumentParser
import os
import numpy as np
import pandas as pd
import math
import re
import matplotlib.pyplot as plt



param_df = pd.read_csv("params_0to20000.csv")
param_df = param_df.loc[param_df['test_acc'] <= 0.7]
#param_df = param_df.loc[param_df['test_acc'] > 0.7]
print(param_df)

#param_name = "L31bn1BsMax_gradmean"
#param_name = "L21bn2WtMax_gradmean"
#param_name = "L11bn1BsMin_gradmean"
#param_name = "L21bn2BsMax_gradmean"
param_name = "L21bn1BsMax_gradmean"
#x = [] 
#print(param_df)
#print(param_df["L31bn1BsMax_grad0-5000"])
#x = np.array([list(instance) for instance in param_df["L31bn1BsMax_grad0-5000"]])

#for i in range(0, len(param_df["L31bn1BsMax_grad0-5000"])):
    #print(param_df["L31bn1BsMax_grad0-5000"][i])
    #numbers = param_df["L31bn1BsMax_grad0-5000"][i].replace('[', '').replace(']', '').split()
    #np_array = np.array(numbers, dtype=np.float64)

    #print(np_array)
    #instance = re.sub('\s+', ',', param_df["L31bn1BsMax_grad0-5000"][i])
    #y = np.fromstring(instance, dtype=np.float64, sep=',')
    #x.append(instance)

#print(x)

#print(np.std(param_df["L31bn1BsMax_gradmean0-5000"]))

y = np.repeat([np.mean(param_df[param_name + "0-5000"]), np.mean(param_df[param_name + "5000-10000"]), np.mean(param_df[param_name + "10000-15000"]), np.mean(param_df[param_name + "15000-20000"])], 5000)
std = np.repeat([np.std(param_df[param_name + "0-5000"]), np.std(param_df[param_name + "5000-10000"]), np.std(param_df[param_name + "10000-15000"]), np.std(param_df[param_name + "15000-20000"])], 5000)
#print(y)
#print(std)

x = np.arange(20000)
fig, ax = plt.subplots()
ax.fill_between(x, y-std, y+std, alpha=0.2)
ax.set_xlabel("Time Step")
ax.set_ylabel("Mean gradient")
ax.set_ylim([-0.1, 0.1]) # All except L11bn1BsMin_gradmean
#ax.set_ylim([-0.6, 0.6]) # For L11bn1BsMin_gradmean
ax.plot(x, y)
plt.savefig(param_name + "_low.png", bbox_inches="tight")
plt.close()

"""
param_name = "valid_loss"
#param_name = "train_loss"
loss_df = pd.read_csv("loss.csv")
x_unique = np.unique(loss_df["x_" + param_name])
x_0 = np.array(x_unique[0].replace('[', '').replace(']', '').split()).astype(np.float64)
x_1 = np.array(x_unique[1].replace('[', '').replace(']', '').split()).astype(np.float64)
x_2 = np.array(x_unique[2].replace('[', '').replace(']', '').split()).astype(np.float64)
y_0 = []
y_1 = []
y_2 = []
#for i in range(len(loss_df["x_" + param_name])):
#    print(loss_df["x_" + param_name][i])
loss_df = loss_df.loc[loss_df['test_acc'] <= 0.7]
loss_df = loss_df.drop(columns=["Unnamed: 0"]) # Remove unnamed column
# Convert string to array
#train_loss_arr = np.array([np.array(i.replace('[', '').replace(']', '').split()).astype(np.float64) for i in loss_df["y_"+param_name]])
for instance in loss_df["y_" + param_name]:
    y = np.array(instance.replace('[', '').replace(']', '').split()).astype(np.float64)
    if len(y) == len(x_0):
        y_0.append(y)
    elif len(y) == len(x_1):
        y_1.append(y)
    elif len(y) == len(x_2):
        y_2.append(y)
df0 = pd.DataFrame(y_0, columns=x_0)
df1 = pd.DataFrame(y_1, columns=x_1)
df2 = pd.DataFrame(y_2, columns=x_2)
#print(df0)
#print(df1)
#print(df2)
df_merge = pd.concat([df0, df1, df2])
#for test in df_merge[175.0]:
#    print(test)
y = np.mean(df_merge, axis=0)
std = np.std(df_merge, axis=0)
#[y for i in range(len(loss_df["train_loss"]))]
#numbers = loss_df["train_loss"][0].replace('[', '').replace(']', '').split()
#np_array = np.array(numbers, dtype=np.float64)
#print(train_loss_arr[0][0])
x = df_merge.columns

fig, ax = plt.subplots()
ax.fill_between(x, y-std, y+std, alpha=0.2)
ax.set_xlabel("Time Step")
ax.set_ylabel("Loss")
ax.set_ylim([0, 4.5]) # train and valid loss
ax.plot(x, y)
plt.savefig(param_name + "_low.png", bbox_inches="tight")
plt.close()
"""
