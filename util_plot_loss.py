# proj/decoder_Deconv
# proj/decoder_PP_pct100
# proj/decoder_PP_pct20
# proj/decoder_PyramidPooling
# proj/decoder_UNETR
# proj/UNetMONAI_pseudo_pct5
# proj/UNetMONAI_pseudo_pct20

# I will load loss.txt and val_loss.txt from the following directories:

import os
import numpy as np
import matplotlib.pyplot as plt

list_wo_timestamp = [
    "proj/decoder_Deconv",
    "proj/decoder_PP_pct100",
    "proj/decoder_PP_pct20",
    "proj/decoder_PyramidPooling",
    "proj/decoder_UNETR",
]

list_w_timestamp = [
    "proj/decoder_UNetMONAI_pct5",
    "proj/decoder_UNetMONAI_pct20",
    "proj/decoder_UNetMONAI_pct100",
]

list_model = list_wo_timestamp + list_w_timestamp

def plot_loss_from_file(filename, output_tag):
    # we need to extract the epoch and loss
    n_wo_timestamp = len(list_wo_timestamp)
    n_w_timestamp = len(list_w_timestamp)
    n_model = n_wo_timestamp + n_w_timestamp
    max_epoch = 300
    loss_model = np.zeros((n_model, max_epoch))
    vaild_epoch = np.zeros(n_model, dtype=int)

    # each line in list_wo_timestamp is like Epoch 1/300, loss: 4.489071458006995e-07
    for idx, folder in enumerate(list_wo_timestamp):
        loss_file = folder + "/" + filename
        with open(loss_file, "r") as f:
            lines = f.readlines()
            for line in lines:
                epoch, loss = line.split(", ")
                # ValueError: could not convert string to float: 'loss: 7.096514131411594e-06\n'
                epoch = int(epoch.split(" ")[1].split("/")[0])
                loss = float(loss.split(": ")[1])
                loss_model[idx, epoch-1] = loss
                vaild_epoch[idx] = epoch

    # each line in list_w_timestamp is like %2024-03-05 22:15:29% -> Epoch 1/300, loss: 0.0001564910279885503
    for idx, folder in enumerate(list_w_timestamp):
        loss_file = folder + "/" + filename
        with open(loss_file, "r") as f:
            lines = f.readlines()
            for line in lines:
                epoch, loss = line.split(", ")
                # epoch is %2024-03-05 22:15:29% -> Epoch 1/300
                epoch = int(epoch.split(">")[1].split("/")[0].split("Epoch ")[1])
                # ValueError: could not convert string to float: 'loss: 7.096514131411594e-06\n'
                loss = float(loss.split(": ")[1])
                loss_model[n_wo_timestamp+idx, epoch-1] = loss
                vaild_epoch[n_wo_timestamp+idx] = epoch

    # plot the loss, label is the folder name
    plt.figure(figsize=(10, 5), dpi=100)
    for idx in range(n_model):
        label = list_model[idx].split("/")[-1]
        data = loss_model[idx, :vaild_epoch[idx]]
        plt.plot(range(vaild_epoch[idx]), data, label=label, alpha=0.5)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(output_tag + " Loss")
    save_name = "Loss_" + output_tag + "_loss_wo_timestamp.png"
    plt.legend()
    plt.yscale("log")
    plt.savefig(save_name)
    plt.close()
    print(f"Save {save_name}")

plot_loss_from_file("loss.txt", "Train")
plot_loss_from_file("val_loss.txt", "Validation")