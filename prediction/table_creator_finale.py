from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from argparse import ArgumentParser
import os
import numpy as np
import pandas as pd
import math
from pandas.testing import assert_frame_equal
import pathlib


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
directory_str = pathlib.Path(args.output_dir) / args.session
#directory_str = f"/work/dlclarge1/nawongsk-MySpace/{args.output_dir}/{args.session}"

directory = os.fsencode(directory_str)

stats = []
y_arr = []
params = []
create_column = True
for seed_file in os.listdir(directory):
    seedfilename = os.fsdecode(seed_file)
    print(seedfilename)
    task_name = seedfilename
    print(directory_str)
    task_folder = directory_str / seedfilename
    #task_folder = directory_str + "/" + seedfilename
#for file in os.listdir(directory):
    for file in os.listdir(task_folder):
        filename = os.fsdecode(file)
        print(f"file name: {filename}")

        #expt_dir = f"/work/dlclarge1/nawongsk-MySpace/{args.output_dir}/{args.session}/{task_name}/{filename}/version_0"
        expt_dir = directory_str / task_name / filename / "version_0"
        event = EventAccumulator(str(expt_dir))
        event.Reload()
        y = event.Scalars('test_accuracy')
        y_arr.append(y[0].value)
        stat_instance = []
        for param in event.Tags()["scalars"]:
            if param.startswith("param/"):

                param_stats = np.array([event_scalar.value for event_scalar in event.Scalars(f'{param}')])

                split = np.array(np.array_split(param_stats, 4))

                grad = np.gradient(split, axis=1)

                #if not (math.isnan(grad[0][0]) and math.isnan(np.gradient(split[0])[0])):
                #    assert grad[0][0] == np.gradient(split[0])[0]

                grad_norm = np.linalg.norm(grad, axis=1)
                grad_mean = np.mean(grad, axis=1)
                grad_percent = ((split[:,-1] - split[:,0])/(split[:,0] + 1e-10)) * 100
                if not (math.isnan(abs(split[:,0] - split[:,-1])[0]) and math.isnan(abs(split[0][0] - split[0][-1]))):
                    assert abs(split[:,0] - split[:,-1])[0] == abs(split[0][0] - split[0][-1])

                for i in range(4):
                    stat_instance.append(grad_norm[i])
                    stat_instance.append(grad_mean[i])
                    stat_instance.append(grad_percent[i])
                block = ""
                layer = ""
                wb = ""
                stat = ""
                if create_column:
                    param_split = param.split("/") # param/layer2.0.bn2.weight/std
                    if "conv1" in param:
                        block = "cn1"
                    elif "conv2" in param:
                        block = "cn2"
                    elif "downsample.0" in param:
                        block = "ds0"
                    elif "downsample.1" in param:
                        block = "ds1"
                    elif "fc" in param:
                        block = "fc"
                    elif "bn1" in param:
                        block = "bn1"
                    elif "bn2" in param:
                        block = "bn2"

                    if "layer" in param_split[1]:
                        layer = "L" + param_split[1][5] + param_split[1][7] 

                    stat = param_split[2].capitalize() 
                    wb = "Bs" if "bias" in param else "Wt"

                    param_name = layer + block + wb + stat

                    params.append(param_name + "_gradnorm0-5000")
                    params.append(param_name + "_gradmean0-5000")
                    params.append(param_name + "_gradpercent0-5000")
                    params.append(param_name + "_gradnorm5000-10000")
                    params.append(param_name + "_gradmean5000-10000")
                    params.append(param_name + "_gradpercent5000-10000")
                    params.append(param_name + "_gradnorm10000-15000")
                    params.append(param_name + "_gradmean10000-15000")
                    params.append(param_name + "_gradpercent10000-15000")
                    params.append(param_name + "_gradnorm15000-20000")
                    params.append(param_name + "_gradmean15000-20000")
                    params.append(param_name + "_gradpercent15000-20000")
        """
        df_test = pd.read_csv("/work/dlclarge1/nawongsk-MySpace/test.csv")
        # testing
        for col in df_test.columns:
            if "grad" in col and "0-5000" in col and not "Loss" in col:
                #print(col)
                #print(params)
                assert col in params
        """
        create_column = False
        stats.append(stat_instance)

#assert len(params) == 744
df = pd.DataFrame(stats, columns=params)
df.insert(len(df.columns), "test_acc", y_arr)
df_no_outlier = df.drop(df[df.test_acc < 0.05].index)
assert len(df_no_outlier) == 617
#df_no_outlier.to_csv("params_0to20000_2.csv")
#assert_frame_equal(pd.read_csv("params_0to20000_2.csv"), pd.read_csv("params_0to20000.csv"))
df_no_outlier.to_csv("params_0to20000.csv")
