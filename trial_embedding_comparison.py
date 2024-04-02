# here we get both mr and ct data, and compare the ViT heads for both
# the encoder is the same, so we can use the same model

from model import output_ViTheads_encoder_MedSAM
from dataset import slice_hdf5_dataset
from util import acquire_data_from_control
from util import remove_data_occupation

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("################### device:", device, "###################")

is_load_model = False

import glob
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader

experiment_name = "trial_embedding_comparison"
root_dir = "proj/" + experiment_name + "/"
if not os.path.exists(root_dir):
    os.makedirs(root_dir)

if is_load_model:
    model = output_ViTheads_encoder_MedSAM(
        img_size=1024,
        patch_size=16,
        in_chans=3,
        out_chans=1,
        out_chans_pretrain=256,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        norm_layer=nn.LayerNorm,
        act_layer=nn.GELU,
        use_abs_pos=True,
        use_rel_pos=False,
        rel_pos_zero_init=True,
        window_size=14,
        global_attn_indexes=(),
        verbose=False,
    )
    model.load_pretrain(pretrain_path = "pretrain/medsam_vit_b.pth")
    model.to(device)
    model.eval()

    dataset_name = "MIMRTL_Brain_subset"
    remove_data_occupation(
        data_folder_name = dataset_name,
        experiment_name = experiment_name,
    )
    case_list = acquire_data_from_control(
            data_folder_name = dataset_name,
            required_case_numbers = 10,
            experiment_name = experiment_name,
    )
    remove_head_tail = 2
    remove_head_tail
    hdf5_file_list = []
    search_prefix = "slice"
    search_affix = ""
    required_keys = ["mr", "ct"]

    for case in case_list:
        found_hdf5 = sorted(glob.glob(case+"/"+search_prefix+"*"+search_affix+".hdf5"))
        # remove the first and last one
        found_hdf5 = found_hdf5[remove_head_tail:-remove_head_tail]
        hdf5_file_list.extend(found_hdf5)

    with open(root_dir+"hdf5_validation_list.txt", "w") as f:
        for item in hdf5_file_list:
            f.write("%s\n" % item)
    # create the dataset and dataloader
    hdf5_dataset = slice_hdf5_dataset(hdf5_file_list, required_keys=required_keys)
    hdf5_dataloader = DataLoader(hdf5_dataset, batch_size=1, shuffle=True)
    n_slice = len(hdf5_dataset)
    total_stat_channel = np.zeros((n_slice, 12, 768))
    total_stat_feature_map = np.zeros((n_slice, 12, 64, 64))

    for idx_batch, data in enumerate(hdf5_dataloader):

        print(f"Processing {idx_batch+1}/{n_slice} samples.")
        mr = data["mr"].float().to(device).squeeze(1)
        ct = data["ct"].float().to(device).squeeze(1)

        with torch.no_grad():
            output_mr = model(mr)
            output_ct = model(ct)
            # outputs are a list containing 
            # after block 0 torch.Size([batch, 64, 64, 768])
            # after block 1 torch.Size([batch, 64, 64, 768])
            # after block 2 torch.Size([batch, 64, 64, 768])
            # after block 3 torch.Size([batch, 64, 64, 768])
            # after block 4 torch.Size([batch, 64, 64, 768])
            # after block 5 torch.Size([batch, 64, 64, 768])
            # after block 6 torch.Size([batch, 64, 64, 768])
            # after block 7 torch.Size([batch, 64, 64, 768])
            # after block 8 torch.Size([batch, 64, 64, 768])
            # after block 9 torch.Size([batch, 64, 64, 768])
            # after block 10 torch.Size([batch, 64, 64, 768])
            # after block 11 torch.Size([batch, 64, 64, 768])
            # after neck x.shape torch.Size([batch, 256, 64, 64])

            stat_channel = np.zeros((12, 768))
            stat_feature_map = np.zeros((12, 64, 64))

            n_embedding = len(output_mr)
            # remove 0th, 1st, and last one
            for i in range(2, n_embedding-1):
                embedding_mr = output_mr[i].squeeze(0)
                embedding_ct = output_ct[i].squeeze(0)
                diff = np.abs(embedding_mr - embedding_ct) # [64, 64, 768]

                diff_channel = np.mean(np.abs(diff), axis=(0, 1)) # [768]
                diff_feature_map = np.mean(np.abs(diff), axis=2) # [64, 64]

                stat_channel[i-2, :] = diff_channel
                stat_feature_map[i-2, :, :] = diff_feature_map

        total_stat_channel[idx_batch, :, :] = stat_channel
        total_stat_feature_map[idx_batch, :, :, :] = stat_feature_map

    # save the data as float32
    total_stat_channel = total_stat_channel.astype(np.float32)
    total_stat_feature_map = total_stat_feature_map.astype(np.float32)
    np.save(root_dir+"total_stat_channel.npy", total_stat_channel)
    np.save(root_dir+"total_stat_feature_map.npy", total_stat_feature_map)

else:
    total_stat_channel = np.load(root_dir+"total_stat_channel.npy", allow_pickle=True)[()]
    total_stat_feature_map = np.load(root_dir+"total_stat_feature_map.npy", allow_pickle=True)[()]


# plot the channel wise diff in mean and std
import matplotlib.pyplot as plt
mean_channel = total_stat_channel.mean(axis=0)
std_channel = total_stat_channel.std(axis=0)

# try 12 subplots, in each plot, plot the mean and use std as error bar
# 12 rows, 1 column, with title as the block number
fig, axs = plt.subplots(12, 1, figsize=(10, 60))
for i in range(12):
    # axs[i].bar(range(768), mean_channel[i, :])
    # plot the curve using thin line
    axs[i].plot(range(768), mean_channel[i, :], color="blue", linewidth=0.5)
    axs[i].set_title(f"Block {i}, mean: {mean_channel[i, :].mean():.4f}")
    axs[i].set_xlabel("Channel")
    axs[i].set_ylabel("Mean")
    # set the y range from 0 to 12
    axs[i].set_ylim(0, 12)

plt.savefig(root_dir+"channel_wise_diff_mean.png")
plt.close()
print("Channel wise diff saved to", root_dir+"channel_wise_diff_mean.png")

# plot the std
fig, axs = plt.subplots(12, 1, figsize=(10, 60))
for i in range(12):
    # axs[i].bar(range(768), mean_channel[i, :])
    # plot the curve using thin line
    axs[i].plot(range(768), std_channel[i, :], color="blue", linewidth=0.5)
    axs[i].set_title(f"Block {i}, std: {std_channel[i, :].mean():.4f}")
    axs[i].set_xlabel("Channel")
    axs[i].set_ylabel("Std")
    # set the y range from 0 to 12
    axs[i].set_ylim(0, 4)

plt.savefig(root_dir+"channel_wise_diff_std.png")
plt.close()
print("Channel wise diff saved to", root_dir+"channel_wise_diff_std.png")





# plot the feature map wise diff in mean and std
mean_feature_map = total_stat_feature_map.mean(axis=0)
std_feature_map = total_stat_feature_map.std(axis=0)

# try 12 subplots, in each plot, plot the mean in 64*64 bars and use std as error bar
# 24 subplots, 4 rows, 6 columns, with title as the block number
fig, axs = plt.subplots(4, 3, figsize=(30, 40))
for i in range(12):
    # force to use 0 to 3 as the color bar range
    image_to_plot = mean_feature_map[i, :, :]

    axs[i//3, i%3].imshow(mean_feature_map[i, :, :], vmin=0, vmax=3)
    axs[i//3, i%3].set_title(f"Block {i} mean: {mean_feature_map[i, :, :].mean():.4f}")
    axs[i//3, i%3].set_xlabel("Width")
    axs[i//3, i%3].set_ylabel("Height")
    # show the color bar
    plt.colorbar(axs[i//3, i%3].imshow(mean_feature_map[i, :, :]))

plt.savefig(root_dir+"feature_map_wise_diff_mean.png")
plt.close()
print("Feature map wise diff saved to", root_dir+"feature_map_wise_diff_mean.png")

# plot the std map
fig, axs = plt.subplots(4, 3, figsize=(30, 40))
for i in range(12):
    # force to use 0 to 3 as the color bar range
    image_to_plot = std_feature_map[i, :, :]

    axs[i//3, i%3].imshow(std_feature_map[i, :, :], vmin=0, vmax=1.5)
    axs[i//3, i%3].set_title(f"Block {i} std: {std_feature_map[i, :, :].mean():.4f}")   
    axs[i//3, i%3].set_xlabel("Width")
    axs[i//3, i%3].set_ylabel("Height")
    # show the color bar
    plt.colorbar(axs[i//3, i%3].imshow(std_feature_map[i, :, :]))

plt.savefig(root_dir+"feature_map_wise_diff_std.png")
plt.close()
print("Feature map wise diff saved to", root_dir+"feature_map_wise_diff_std.png")




        
