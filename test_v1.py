# read the config file as json
import os
import json

config_path = "config_0205_test.json"
with open(config_path, "r") as f:
    config = json.load(f)
print(config)

# set the root folder
root_folder = config["save_folder"]
if not os.path.exists(root_folder):
    os.makedirs(root_folder)
result_folder = config["result_folder"]
if not os.path.exists(result_folder):
    os.makedirs(result_folder)

# set gpu list
gpu_list = config["gpu_list"]
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list
print('export CUDA_VISIBLE_DEVICES=' + gpu_list)
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# set the model
from model import Unet

model = Unet(
    dim = 64,
    dim_mults = (1, 2, 4, 8),
    channels=3,
    with_time_emb=True,
    residual=True,
)

# load the target model
model.load_state_dict(torch.load(root_folder+config["load_path"]))
model.to(device)

# load the data path
import glob
data_path_x = config["data_path_x"]
data_path_y = config["data_path_y"]
data_folder_x_test = sorted(glob.glob(data_path_x+"/test/*.npy"))
n_test = len(data_folder_x_test)

# train the model

import torch
import random
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

max_time = config["time_steps"]
img_channels = config["img_channels"]
img_size_x = config["img_size_x"]
img_size_y = config["img_size_y"]
eval_metrics = config["eval_metrics"]

for idx_case in range(n_test):
    
    # load the case
    data_x = np.load(data_folder_x_test[idx_case])
    filename = os.path.basename(data_folder_x_test[idx_case])
    data_y = np.load(data_path_y+"/test/"+filename)
    data_t = np.zeros((1, img_channels, img_size_x, img_size_y))
    data_t[0, :, :, :] = data_x
    data_t0 = torch.tensor(data_t).float().to(device)

    # iterate through the time
    for idx_t in range(max_time):
        time = torch.tensor([idx_t]).float().to(device)
        with torch.no_grad():
            pred_t1 = model(data_t0, time)
        data_t0 = pred_t1

    # save the metrics
    data_t1 = np.squeeze(pred_t1.cpu().numpy())
    if eval_metrics == "mae":
        metric = np.mean(np.abs(data_t1 - data_y)) * 4000
    else:
        raise ValueError("Unknown eval metrics")
    
    with open(result_folder + "test_metrics.txt", "a") as f:
        f.write("%s, %.6f\n" % (filename, metric))
    print("Case: %s, Metric: %.6f" % (filename, metric))

    # save for preview
    plt.figure(figsize=(12, 3))
    plt.subplot(1, 4, 1)
    plt.imshow(data_x[1, :, :], cmap="gray")
    plt.title("MR")
    plt.axis("off")

    plt.subplot(1, 4, 2)
    plt.imshow(data_y[1, :, :], cmap="gray")
    plt.title("CT")
    plt.axis("off")

    plt.subplot(1, 4, 3)
    plt.imshow(data_t1[1, :, :], cmap="gray")
    plt.title("pred")
    plt.axis("off")

    # show the diff using red-blue color map and make the zero as white
    plt.subplot(1, 4, 4)
    diff_img = data_y[1:, :, :]-data_t1[1, :, :]
    zero_whihte_norm = TwoSlopeNorm(vmin=np.amin(diff_img), vcenter=0, vmax=np.amax(diff_img))
    plt.imshow(data_y[1:, :, :]-data_t1[1, :, :], norm=zero_whihte_norm, cmap="seismic")
    plt.colorbar()
    plt.title("diff (y-pred)")
    plt.axis("off")

    # set the total title
    plt.suptitle("Case: %s, Metric: %.6f" % (filename, metric))

    plt.savefig(result_folder + "preview_%s.png" % filename)
    plt.close()

    # save the pred
    np.save(result_folder + filename, data_t1)

# finish the training
print("Finish the testing")
