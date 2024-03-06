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
    "proj/UNetMONAI_pseudo_pct5",
    "proj/UNetMONAI_pseudo_pct20",
]

# in loss.txt
# each line in list_wo_timestamp is like Epoch 1/300, loss: 4.489071458006995e-07
# we need to extract the epoch and loss

n_wo_timestamp = len(list_wo_timestamp)
max_epoch = 300
loss_wo_timestamp = np.zeros((n_wo_timestamp, max_epoch))
vaild_epoch = np.zeros(n_wo_timestamp, dtype=int)
for idx, folder in enumerate(list_wo_timestamp):
    loss_file = folder + "/loss.txt"
    model_name = folder.split("/")[-1]
    print(f"Loading {loss_file} for {model_name}")
    with open(loss_file, "r") as f:
        lines = f.readlines()
        for line in lines:
            epoch, loss = line.split(", ")
            # ValueError: could not convert string to float: 'loss: 7.096514131411594e-06\n'
            epoch = int(epoch.split(" ")[1].split("/")[0])
            loss = float(loss.split(": ")[1])
            loss_wo_timestamp[idx, epoch-1] = loss
            vaild_epoch[idx] = epoch
            print(f"Epoch {epoch}, loss: {loss}")

print(vaild_epoch)

# plot the loss, label is the folder name
plt.figure(figsize=(10, 5), dpi=100)
for idx in range(n_wo_timestamp):
    label = list_wo_timestamp[idx].split("/")[-1]
    data = loss_wo_timestamp[idx, :vaild_epoch[idx]]
    plt.plot(data, label=label)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
save_name = "training_loss_wo_timestamp.png"
plt.legend()
plt.savefig(save_name)
plt.close()
print(f"Save {save_name}")