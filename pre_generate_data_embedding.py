# data folder:

# data/MIMRTL_Brain
# data/SynthRad_Brain
# data/SynthRad_Pelvis

# for each folder path, this is like:
# /path_to_data_folder/name_of_case/mr.nii.gz
# /path_to_data_folder/name_of_case/ct.nii.gz
# the data range for mr is from 0 to 3000
# the data range for ct is from -1000 to 3000

# for each case, we will:
# 1: Load mr.nii.gz and ct.nii.gz
# 2: Normalise mr and ct to [0, 1]
# 3: Divide the MR into (3, res_x, res_y), according to last dim.
#     e.g. 256*256*148 -> 3*256*256, for 148 slices
#     pad the first and last slice with 0 to make sure each slice is centered
# 4: Input the MR into the MedSAM model encoder, and get the output
# 5: Divide the CT into (1, res_x, res_y), according to last dim.
#     e.g. 256*256*148 -> 1*256*256, for 148 slices
# 6: Save the results into a dict named "MedSAM_embedding"
#     e.g. MedSAM_embedding = {
#         "1": {
#             "mr_embedding_3": (768, res_x, res_y)
#             "mr_embedding_6": (768, res_x, res_y)
#             "mr_embedding_9": (768, res_x, res_y)
#             "mr_embedding_12": (768, res_x, res_y)
#             "mr_embedding_neck": (256, res_x, res_y)
#             "mr": mr_slice, (-1:1, res_x, res_y)
#             "ct": ct_slice, (0, res_x, res_y)
#              },
#         "2": {
#             "mr_embedding_3": (768, res_x, res_y)
#             "mr_embedding_6": (768, res_x, res_y)
#             "mr_embedding_9": (768, res_x, res_y)
#             "mr_embedding_12": (768, res_x, res_y)
#             "mr_embedding_neck": (256, res_x, res_y)
#             "mr": mr_slice, (-1:1, res_x, res_y)
#             "ct": ct_slice, (0, res_x, res_y)
#              },
#         ...
#     }
# 7: Save the dict into a .hdf5 file in the same folder, named as "MedSAM_embedding.hdf5"

import os
import glob
import time
import json
import nibabel as nib
import h5py
import numpy as np


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
import torch.nn as nn
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("The device is: ", device)

# load the config file
cfg_path = "config/MedSAM_encoder.json"
cfg = json.load(open(cfg_path))

# load the model
from model.MedSAM_encoder import MedSAM_encoder
model = MedSAM_encoder(
    img_size=cfg["img_size"],
    patch_size=cfg["patch_size"],
    in_chans=cfg["in_chans"],
    out_chans=cfg["out_chans"],
    out_chans_pretrain=cfg["out_chans_pretrain"],
    embed_dim=cfg["embed_dim"],
    depth=cfg["depth"],
    num_heads=cfg["num_heads"],
    mlp_ratio=cfg["mlp_ratio"],
    qkv_bias=True if cfg["qkv_bias"] == "True" else False,
    norm_layer=nn.LayerNorm if cfg["norm_layer"] == "nn.LayerNorm" else None,
    act_layer=nn.GELU if cfg["act_layer"] == "nn.GELU" else None,
    use_abs_pos=True if cfg["use_abs_pos"] == "True" else False,
    use_rel_pos=False if cfg["use_rel_pos"] == "False" else True,
    rel_pos_zero_init=True if cfg["rel_pos_zero_init"] == "True" else False,
    window_size=cfg["window_size"],
    global_attn_indexes=cfg["global_attn_indexes"],
    # verbose=True,
)
model.load_pretrain(cfg["pretrain_path"])
model.to(device)
print("Model loaded from ", cfg["pretrain_path"])


data_folder_list = [
    ["data/MIMRTL_Brain", 33, 218],
    ["data/MIMRTL_Brain", 219, 404],
    ["data/MIMRTL_Brain", 405, 590],
    ["data/MIMRTL_Brain", 591, 777],
    ["data/SynthRad_Brain", 23, 181],
    ["data/SynthRad_Pelvis", 0, -1],
]

# take the user input to determine which data folder to process
input_user = input("Which data folder to process?\n1: MIMRTL_Brain 33-218\n2: MIMRTL_Brain 219-404\n3: MIMRTL_Brain 405-590\n4: MIMRTL_Brain 591-777\n5: SynthRad_Brain 23-181\n6: SynthRad_Pelvis 0-All\n")
if input_user == "1":
    data_folder_to_process, start_case_num, end_case_num = data_folder_list[0]
elif input_user == "2":
    data_folder_to_process, start_case_num, end_case_num = data_folder_list[1]
elif input_user == "3":
    data_folder_to_process, start_case_num, end_case_num = data_folder_list[2]
elif input_user == "4":
    data_folder_to_process, start_case_num, end_case_num = data_folder_list[3]
elif input_user == "5":
    data_folder_to_process, start_case_num, end_case_num = data_folder_list[4]
elif input_user == "6":
    data_folder_to_process, start_case_num, end_case_num = data_folder_list[5]
else:
    print("Invalid input, exit.")
    exit()

for data_folder in [data_folder_to_process]:
    print(f"Processing {data_folder}")
    folder_name = data_folder.split("/")[-1]
    case_list = sorted(glob.glob(os.path.join(data_folder, "*/")))
    case_list = case_list[start_case_num:end_case_num]
    n_cases = len(case_list)

    for idx_case, case_path in enumerate(case_list):
        start_time = time.time()
        print(f"[{data_folder}][{idx_case+1+start_case_num}/{n_cases+start_case_num}] Processing {case_path}")
        mr_file = nib.load(os.path.join(case_path, "mr.nii.gz"))
        ct_file = nib.load(os.path.join(case_path, "ct.nii.gz"))
        mr_data = mr_file.get_fdata()
        ct_data = ct_file.get_fdata()
        res_x, res_y, res_z = mr_data.shape
        print(f"[{data_folder}][{idx_case+1+start_case_num}/{n_cases+start_case_num}] MR shape: {mr_data.shape}, CT shape: {ct_data.shape}")

        # normalise mr and ct to [0, 1]
        mr_data = np.clip(mr_data, 0, 3000) / 3000
        ct_data = np.clip(ct_data+1000, 0, 4000) / 4000
        print(f"[{data_folder}][{idx_case+1+start_case_num}/{n_cases+start_case_num}] Normalised MR and CT")

        # pad mr_data
        mr_data = np.pad(mr_data, ((0, 0), (0, 0), (1, 1)), mode="constant")
        print(f"[{data_folder}][{idx_case+1+start_case_num}/{n_cases+start_case_num}] Padded MR")

        # create the MedSAM_embedding dict
        MedSAM_embedding = {}

        # divide the MR into (3, res_x, res_y), according to last dim.
        for idx_z in range(1, res_z+1):
            mr_slice = mr_data[:, :, idx_z-1:idx_z+2] # (256, 256, 3)
            mr_slice = torch.from_numpy(mr_slice).float().unsqueeze(0) # (1, 256, 256, 3)
            mr_slice = mr_slice.permute(0, 3, 1, 2).to(device) # (1, 3, 256, 256)
            # interpolate the MR slice to (1, 3, 1024, 1024)
            mr_slice = F.interpolate(mr_slice, size=(1024, 1024), mode="bilinear", align_corners=False)
            print(f"[{data_folder}][{idx_case+1+start_case_num}/{n_cases+start_case_num}][{idx_z}/{res_z}] MR slice shape: {mr_slice.shape}")

            # input the MR into the MedSAM model encoder, and get the output
            with torch.no_grad():
                head_3, head_6, head_9, head_12, head_neck = model(mr_slice)
                # head_3: (1, 768, 64, 64)
                # head_6: (1, 768, 64, 64)
                # head_9: (1, 768, 64, 64)
                # head_12: (1, 768, 64, 64)
                # head_neck: (1, 256, 64, 64)
                head_3 = head_3.detach().cpu().numpy()
                head_6 = head_6.detach().cpu().numpy()
                head_9 = head_9.detach().cpu().numpy()
                head_12 = head_12.detach().cpu().numpy()
                head_neck = head_neck.detach().cpu().numpy()

                # head_3 = head_3.permute(0, 3, 1, 2).detach().cpu().numpy()
                # head_6 = head_6.permute(0, 3, 1, 2).detach().cpu().numpy()
                # head_9 = head_9.permute(0, 3, 1, 2).detach().cpu().numpy()
                # head_12 = head_12.permute(0, 3, 1, 2).detach().cpu().numpy()
                # head_neck = head_neck.permute(0, 3, 1, 2).detach().cpu().numpy()
                print(f"[{data_folder}][{idx_case+1+start_case_num}/{n_cases+start_case_num}][{idx_z}/{res_z}] MR embedding shape: {head_3.shape}, {head_6.shape}, {head_9.shape}, {head_12.shape}, {head_neck.shape}")

            # save the results into a dict named "MedSAM_embedding"
            ct_slice = ct_data[:, :, idx_z-1:idx_z] # (256, 256, 1)
            ct_slice = np.squeeze(ct_slice) # (256, 256)
            ct_slice = np.expand_dims(ct_slice, axis=0) # (1, 256, 256)
            # interpolate the CT slice to (1, 1024, 1024)
            ct_slice = torch.from_numpy(ct_slice).float().unsqueeze(0)
            ct_slice = F.interpolate(ct_slice, size=(1024, 1024), mode="bilinear", align_corners=False)
            ct_slice = ct_slice.detach().cpu().numpy() # (1, 1024, 1024)
            mr_slice = mr_slice.detach().cpu().numpy() # (1, 3, 1024, 1024)
            MedSAM_embedding[str(idx_z)] = {
                "mr_emb":{
                    "head_3": head_3,
                    "head_6": head_6,
                    "head_9": head_9,
                    "head_12": head_12,
                    "head_neck": head_neck,
                },
                "mr": mr_slice,
                "ct": ct_slice,
            }
            print(f"[{data_folder}][{idx_case+1+start_case_num}/{n_cases+start_case_num}][{idx_z}/{res_z}] Saved MR and CT")

        # create the .hdf5 file and compress the MedSAM_embedding dict
        
        with h5py.File(os.path.join(case_path, "MedSAM_embedding_gzip4.hdf5"), "w") as f:
            for key in MedSAM_embedding.keys():
                grp = f.create_group(key)
                grp.create_dataset("mr_emb_head_3", data=MedSAM_embedding[key]["mr_emb"]["head_3"], compression="gzip", compression_opts=4)
                grp.create_dataset("mr_emb_head_6", data=MedSAM_embedding[key]["mr_emb"]["head_6"], compression="gzip", compression_opts=4)
                grp.create_dataset("mr_emb_head_9", data=MedSAM_embedding[key]["mr_emb"]["head_9"], compression="gzip", compression_opts=4)
                grp.create_dataset("mr_emb_head_12", data=MedSAM_embedding[key]["mr_emb"]["head_12"], compression="gzip", compression_opts=4)
                grp.create_dataset("mr_emb_head_neck", data=MedSAM_embedding[key]["mr_emb"]["head_neck"], compression="gzip", compression_opts=4)
                grp.create_dataset("mr", data=MedSAM_embedding[key]["mr"], compression="gzip", compression_opts=4)
                grp.create_dataset("ct", data=MedSAM_embedding[key]["ct"], compression="gzip", compression_opts=4)
                # ratio = 9, file_size = 6.95GB
                # ratio = 4, file_size = 6.96GB

        # with h5py.File(os.path.join(case_path, "MedSAM_embedding_lzf.hdf5"), "w") as f:
        #     for key in MedSAM_embedding.keys():
        #         grp = f.create_group(key)
        #         grp.create_dataset("mr_emb_head_3", data=MedSAM_embedding[key]["mr_emb"]["head_3"], compression="lzf")
        #         grp.create_dataset("mr_emb_head_6", data=MedSAM_embedding[key]["mr_emb"]["head_6"], compression="lzf")
        #         grp.create_dataset("mr_emb_head_9", data=MedSAM_embedding[key]["mr_emb"]["head_9"], compression="lzf")
        #         grp.create_dataset("mr_emb_head_12", data=MedSAM_embedding[key]["mr_emb"]["head_12"], compression="lzf")
        #         grp.create_dataset("mr_emb_head_neck", data=MedSAM_embedding[key]["mr_emb"]["head_neck"], compression="lzf")
        #         grp.create_dataset("mr", data=MedSAM_embedding[key]["mr"], compression="lzf")
        #         grp.create_dataset("ct", data=MedSAM_embedding[key]["ct"], compression="lzf")
        
        print(f"[{data_folder}][{idx_case+1+start_case_num}/{n_cases+start_case_num}] Saved MedSAM_embedding.hdf5")
        end_time = time.time()
        # write to txt file and record the time in seconds
        with open(os.path.join("./"+folder_name+"_time.txt"), "a") as f:
            f.write(f"Case {idx_case+1+start_case_num}/{n_cases+start_case_num}: {end_time-start_time:.4f} seconds\n")
                

# [data/MIMRTL_Brain][1/777][121/124] MR slice shape: torch.Size([1, 3, 1024, 1024])
# [data/MIMRTL_Brain][1/777][121/124] MR embedding shape: (1, 768, 64, 64), (1, 768, 64, 64), (1, 768, 64, 64), (1, 768, 64, 64), (1, 256, 64, 64)
# [data/MIMRTL_Brain][1/777][121/124] Saved MR and CT