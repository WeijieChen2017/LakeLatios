# read the config file as json
import os
import json

config_path = "config_0205.json"
with open(config_path, "r") as f:
    config = json.load(f)
print(config)

# set the root folder
root_folder = config["save_folder"]
if not os.path.exists(root_folder):
    os.makedirs(root_folder)

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
).to(device)

# set the adam optimizer
from torch.optim import Adam

train_lr = config["train_lr"]
optimizer = Adam(model.parameters(), lr=train_lr)

# load the data path
import glob
data_path_x = config["data_path_x"]
data_path_y = config["data_path_y"]
data_folder_x_train = sorted(glob.glob(data_path_x+"/train/*.npy"))
data_folder_x_val = sorted(glob.glob(data_path_x+"/val/*.npy"))
n_train = len(data_folder_x_train)
print(data_folder_x_train)

# train the model

import torch
import random
import numpy as np

import matplotlib.pyplot as plt

train_epochs = config["train_epochs"]
max_time = config["time_steps"]
batch_size = config["batch_size"]
img_channels = config["img_channels"]
img_size_x = config["img_size_x"]
img_size_y = config["img_size_y"]
loss_type = config["loss_type"]
if loss_type == "l1":
    loss_fn = torch.nn.L1Loss()
elif loss_type == "l2":
    loss_fn = torch.nn.MSELoss()
else:
    raise ValueError("Unknown loss type")

print_loss_step = config["print_loss_step"]
eval_step = config["eval_step"]
save_step = config["save_step"]
eval_batch_num = config["eval_batch_num"]

for idx_epoch in range(train_epochs):

    data_t0 = np.zeros((batch_size, img_channels, img_size_x, img_size_y))
    data_t1 = np.zeros((batch_size, img_channels, img_size_x, img_size_y))
    time = np.zeros((batch_size, 1))
    best_eval_loss = 1e10

    for idx_batch in range(batch_size):

        # randomly select the data
        # the data is normalized to [0, 1]
        idx_data = random.choice(range(n_train))
        data_x = np.load(data_folder_x_train[idx_data])
        filename = os.path.basename(data_folder_x_train[idx_data])
        data_y = np.load(data_path_y+"/train/"+filename)

        # generate the fusion data
        tb = np.random.randint(max_time)
        data_t0b = data_x * (1 - time / max_time) + data_y * (time / max_time)
        data_t1b = data_x * (1 - (time+1) / max_time) + data_y * ((time+1) / max_time)

        data_t0[idx_batch, :, :, :] = data_t0b
        data_t1[idx_batch, :, :, :] = data_t1b
        time[idx_batch] = tb

    # convert to tensor
    data_t0 = np.expand_dims(data_t0, axis=(0))
    data_t1 = np.expand_dims(data_t1, axis=(0))
    data_t0 = torch.tensor(data_t0).float().to(device)
    data_t1 = torch.tensor(data_t1).float().to(device)
    time = torch.tensor(time).float().to(device)

    # forward
    optimizer.zero_grad()
    pred_t1 = model(data_t0, time)
    loss = loss_fn(pred_t1, data_t1)
    loss.backward()
    optimizer.step()

    # save loss
    if idx_epoch % print_loss_step == 0:
        print("Epoch: %d, Loss: %.6f" % (idx_epoch, loss.item()))
        curr_loss = loss.item()
        with open(root_folder + "loss.txt", "a") as f:
            f.write("%d, %.6f\n" % (idx_epoch, curr_loss))
    
    # save model and optimizer
    if idx_epoch % save_step == 0:
        torch.save(model.state_dict(), root_folder + "model_%d.pth" % (idx_epoch))
        torch.save(optimizer.state_dict(), root_folder + "optimizer_%d.pth" % (idx_epoch))
    
    # save the preview
    if idx_epoch % eval_step == 0:
        plt.figure(figsize=(12, 3))
        plt.subplot(1, 4, 1)
        plt.imshow(data_x[0, :, :], cmap="gray")
        plt.title("MR")
        plt.axis("off")

        plt.subplot(1, 4, 2)
        plt.imshow(data_y[0, :, :], cmap="gray")
        plt.title("CT")
        plt.axis("off")

        plt.subplot(1, 4, 3)
        plt.imshow(data_t0[0, 0, :, :], cmap="gray")
        plt.title("fusion t0")
        plt.axis("off")

        plt.subplot(1, 4, 4)
        plt.imshow(data_t1[0, 0, :, :], cmap="gray")
        plt.title("fusion t1")
        plt.axis("off")

        plt.savefig(root_folder + "preview_%d_at_T_%d.png" % (idx_epoch, time[0].cpu().numpy()))
        plt.close()
    
    # evaluation and save the best model and optimizer
    
    if idx_epoch % eval_step == 0:
        eval_loss = 0
        for idx_eval in range(eval_batch_num):

            data_t0 = np.zeros((batch_size, img_channels, img_size_x, img_size_y))
            data_t1 = np.zeros((batch_size, img_channels, img_size_x, img_size_y))
            time = np.zeros((batch_size, 1))

            for idx_batch in range(batch_size):

                # randomly select the data
                # the data is normalized to [0, 1]
                idx_data = random.choice(range(n_train))
                data_x = np.load(data_folder_x_val[idx_data])
                filename = os.path.basename(data_folder_x_val[idx_data])
                data_y = np.load(data_path_y+"/val/"+filename)

                # generate the fusion data
                tb = np.random.randint(max_time)
                data_t0b = data_x * (1 - time / max_time) + data_y * (time / max_time)
                data_t1b = data_x * (1 - (time+1) / max_time) + data_y * ((time+1) / max_time)

                data_t0[idx_batch, :, :, :] = data_t0b
                data_t1[idx_batch, :, :, :] = data_t1b
                time[idx_batch] = tb

            data_t0 = np.expand_dims(data_t0, axis=(0))
            data_t1 = np.expand_dims(data_t1, axis=(0))
            data_t0 = torch.tensor(data_t0).float().to(device)
            data_t1 = torch.tensor(data_t1).float().to(device)
            time = torch.tensor(time).float().to(device)

            # forward without gradient
            with torch.no_grad():
                pred_t1 = model(data_t0, time)
                eval_loss += loss_fn(pred_t1, data_t1).item()
        
        eval_loss /= eval_batch_num
        if eval_loss < best_eval_loss:
            best_eval_loss = eval_loss
            torch.save(model.state_dict(), root_folder + "best_model.pth")
            torch.save(optimizer.state_dict(), root_folder + "best_optimizer.pth")
        with open(root_folder + "eval_loss.txt", "a") as f:
            f.write("%d, %.6f\n" % (idx_epoch, eval_loss))
        print("Epoch: %d, Eval Loss: %.6f" % (idx_epoch, eval_loss))

# finish the training
print("Finish the training")

        






    # # save for preview
    # plt.figure(figsize=(12, 3))
    # plt.subplot(1, 4, 1)
    # plt.imshow(data_x[0, :, :], cmap="gray")
    # plt.title("MR")
    # plt.axis("off")

    # plt.subplot(1, 4, 2)
    # plt.imshow(data_y[0, :, :], cmap="gray")
    # plt.title("CT")
    # plt.axis("off")

    # plt.subplot(1, 4, 3)
    # plt.imshow(data_t0[0, 0, :, :], cmap="gray")
    # plt.title("fusion t0")
    # plt.axis("off")

    # plt.subplot(1, 4, 4)
    # plt.imshow(data_t1[0, 0, :, :], cmap="gray")
    # plt.title("fusion t1")
    # plt.axis("off")

    # plt.savefig(root_folder + "preview.png")
    # plt.close()



