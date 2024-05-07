from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import numpy as np
import matplotlib.pyplot as plt
import argparse
from argparse import ArgumentParser
import pathlib
import os

parser = ArgumentParser()
parser.add_argument("--session", type=str, default='test_resnet')
parser.add_argument("--max_train_steps", type=int, default=20000)
parser.add_argument("--model_name", type=str, default="ResNet18")
parser.add_argument("--output_dir", type=str, default="cifar100")

args = parser.parse_args()


#task_name = f"{args.model_name}_seed{args.seed}_steps{args.max_train_steps}"

#directory_str = f"/work/dlclarge1/nawongsk-MySpace/{args.output_dir}/{args.session}/{task_name}"
directory_str = f"/work/dlclarge1/nawongsk-MySpace/{args.output_dir}/{args.session}"

directory = os.fsencode(directory_str)

for seed_file in os.listdir(directory): # Look through each file
    seedfilename = os.fsdecode(seed_file)
    print(seedfilename)
    task_name = seedfilename
    print(directory_str)
    task_folder = directory_str + "/" + seedfilename
    for file in os.listdir(task_folder):
        filename = os.fsdecode(file)
        print(filename)
        expt_dir = f"/work/dlclarge1/nawongsk-MySpace/{args.output_dir}/{args.session}/{task_name}/{filename}/version_0"
        event = EventAccumulator(expt_dir)
        event.Reload()

        param_name = ["conv1", "bn1", "layer1.0.conv1", "layer1.0.bn1", "layer1.0.conv2", "layer1.0.bn2", "layer1.1.conv1", "layer1.1.bn1", "layer1.1.conv2", "layer1.1.bn2",
                        "layer2.0.conv1", "layer2.0.bn1", "layer2.0.conv2", "layer2.0.bn2", "layer2.0.downsample.0", "layer2.0.downsample.1", "layer2.1.conv1", "layer2.1.bn1",
                        "layer2.1.conv2", "layer2.1.bn2", "layer3.0.conv1", "layer3.0.bn1", "layer3.0.conv2", "layer3.0.bn2", "layer3.0.downsample.0", "layer3.0.downsample.1",
                        "layer3.1.conv1", "layer3.1.bn1", "layer3.1.conv2", "layer3.1.bn2", "layer4.0.conv1", "layer4.0.bn1", "layer4.0.conv2", "layer4.0.bn2", "layer4.0.downsample.0",
                        "layer4.0.downsample.1", "layer4.1.conv1", "layer4.1.bn1", "layer4.1.conv2", "layer4.1.bn2", "fc"]


        fig = plt.figure(figsize=(15, 75), layout="constrained")
        gs = fig.add_gridspec(len(param_name), 4)

        for i, param in enumerate(param_name):
            diff = (((len(param_name) - i)/len(param_name)) - ((len(param_name) - i + 1)/len(param_name))) / 2 # y-coordinate where to position text is current position + difference between current and next position

            text = fig.text(0, ((len(param_name) - i)/len(param_name)) + diff, param, horizontalalignment="right")

            if param.rfind("conv") != -1 or param.rfind("downsample.0") != -1: # plot for convolutional and downsample.0 layers


                weight = fig.add_subplot(gs[i, 1:3], ylim=(-1, 2))
                x = np.array([event_scalar.step for event_scalar in event.Scalars(f'param/{param}.weight/mean')])
                y = np.array([event_scalar.value for event_scalar in event.Scalars(f'param/{param}.weight/mean')])
                min = np.array([event_scalar.value for event_scalar in event.Scalars(f'param/{param}.weight/min')])
                std = np.array([event_scalar.value for event_scalar in event.Scalars(f'param/{param}.weight/std')])
                max = np.array([event_scalar.value for event_scalar in event.Scalars(f'param/{param}.weight/max')])
                weight.axes.set_title("Weight")
                weight.fill_between(x, y-std, y+std, alpha=0.2)
                weight.plot(x, min, '--')
                weight.plot(x, max, '--')
                weight.plot(x, y)



            elif param.rfind("bn") != -1 or param.rfind("downsample.1") != -1 or param.rfind("fc") != -1: # plot for batchnorm, downsample.1 and FC layers

                weight1 = fig.add_subplot(gs[i, 0:2], ylim=(-1, 2))
                x = np.array([event_scalar.step for event_scalar in event.Scalars(f'param/{param}.weight/mean')])
                y = np.array([event_scalar.value for event_scalar in event.Scalars(f'param/{param}.weight/mean')])
                min = np.array([event_scalar.value for event_scalar in event.Scalars(f'param/{param}.weight/min')])
                std = np.array([event_scalar.value for event_scalar in event.Scalars(f'param/{param}.weight/std')])
                max = np.array([event_scalar.value for event_scalar in event.Scalars(f'param/{param}.weight/max')])
                weight1.axes.set_title("Weight")
                weight1.fill_between(x, y-std, y+std, alpha=0.2)
                weight1.plot(x, min, '--')
                weight1.plot(x, max, '--')
                weight1.plot(x, y)

                bias1 = fig.add_subplot(gs[i, 2:], ylim=(-1, 2))
                x = np.array([event_scalar.step for event_scalar in event.Scalars(f'param/{param}.bias/mean')])
                y = np.array([event_scalar.value for event_scalar in event.Scalars(f'param/{param}.bias/mean')])
                min = np.array([event_scalar.value for event_scalar in event.Scalars(f'param/{param}.bias/min')])
                std = np.array([event_scalar.value for event_scalar in event.Scalars(f'param/{param}.bias/std')])
                max = np.array([event_scalar.value for event_scalar in event.Scalars(f'param/{param}.bias/max')])
                bias1.axes.set_title("Bias")
                bias1.fill_between(x, y-std, y+std, alpha=0.2)
                bias1.plot(x, min, '--')
                bias1.plot(x, max, '--')
                bias1.plot(x, y)


        task_name = f"{args.model_name}_seed{args.seed}_steps{args.max_train_steps}"
        expt_dir = pathlib.Path("hparam_graphs_p2") / args.output_dir / args.session / task_name
        expt_dir.mkdir(parents=True, exist_ok=True)
        base_dir = f"hparam_graphs_p2/{args.output_dir}/{args.session}"
        test_acc = event.Scalars('test_accuracy')[0].value
        expt_name = base_dir + "/" + task_name + "/" + filename + "_acc" + str(round(test_acc, 3)) + ".png"
        plt.savefig(expt_name, bbox_inches="tight")
        plt.close()
