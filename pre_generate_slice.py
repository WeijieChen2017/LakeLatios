
import os
import glob
import time
import torch
import nibabel as nib
import h5py
import numpy as np
import torch.nn.functional as F

data_folder_list = [
    "data/MIMRTL_Brain",
    "data/SynthRad_Brain",
    "data/SynthRad_Pelvis",
    "data/MIMRTL_Brain_subset",
]

# take the user input to determine which data folder to process
input_idx = int(input("Which data folder to process? (0-3) "))
data_folder_to_process = data_folder_list[input_idx]
dataset_name = data_folder_to_process.split("/")[-1]
record_txt_name = f"data/{dataset_name}_record.txt"

def log_message(message):
    print(message)
    with open(record_txt_name, "a") as f:
        f.write(message + "\n")

for data_folder in [data_folder_to_process]:
    print(f"Processing {data_folder}")
    folder_name = data_folder.split("/")[-1]
    case_list = sorted(glob.glob(os.path.join(data_folder, "*/")))

    n_cases = len(case_list)

    for idx_case, case_path in enumerate(case_list):
        start_time = time.time()
        print(f"[{data_folder}][{idx_case+1}/{n_cases}] Processing {case_path}")
        mr_file = nib.load(os.path.join(case_path, "mr.nii.gz"))
        ct_file = nib.load(os.path.join(case_path, "ct.nii.gz"))
        mr_data = mr_file.get_fdata()
        ct_data = ct_file.get_fdata()
        res_x, res_y, res_z = mr_data.shape

        # normalise mr and ct to [0, 1]
        mr_data = np.clip(mr_data, 0, 3000) / 3000
        ct_data = np.clip(ct_data+1000, 0, 4000) / 4000
        log_message(f"[{data_folder}][{idx_case+1}/{n_cases}] Normalised MR and CT")

        # pad mr_data
        # mr_data = np.pad(mr_data, ((0, 0), (0, 0), (1, 1)), mode="constant")
        # log_message(f"[{data_folder}][{idx_case+1}/{n_cases}] Padded MR")

        # divide the MR into (3, res_x, res_y), according to last dim.
        for idx_z in range(1, res_z+1):
            mr_slice = np.squeeze(mr_data[:, :, idx_z]) # (256, 256, 3)
            # repeat the mr_slice for 3 times
            mr_slice = np.repeat(mr_slice[:, :, np.newaxis], 3, axis=2)
            mr_slice = torch.from_numpy(mr_slice).float().unsqueeze(0) # (1, 256, 256, 3)
            mr_slice = mr_slice.permute(0, 3, 1, 2) # (1, 3, 256, 256)
            # interpolate the MR slice to (1, 3, 1024, 1024)
            mr_slice = F.interpolate(mr_slice, size=(1024, 1024), mode="bilinear", align_corners=False)

            # save the results into a dict named "MedSAM_embedding"
            if data_folder_to_process == "data/MIMRTL_Brain_subset":
                ct_slice = ct_data[:, :, idx_z-1:idx_z+2] # (256, 256, 3)
                ct_slice = torch.from_numpy(ct_slice).float().unsqueeze(0) # (1, 256, 256, 3)
                ct_slice = ct_slice.permute(0, 3, 1, 2) # (1, 3, 256, 256)
                # interpolate the CT slice to (1, 3, 1024, 1024)
                ct_slice = F.interpolate(ct_slice, size=(1024, 1024), mode="bilinear", align_corners=False)
            else:
                ct_slice = ct_data[:, :, idx_z] # (256, 256, 1)
                ct_slice = np.squeeze(ct_slice) # (256, 256)
                ct_slice = np.expand_dims(ct_slice, axis=0) # (1, 256, 256)
                # interpolate the CT slice to (1, 1024, 1024)
                ct_slice = torch.from_numpy(ct_slice).float().unsqueeze(0)
                ct_slice = F.interpolate(ct_slice, size=(1024, 1024), mode="bilinear", align_corners=False)
            
            ct_slice = ct_slice.detach().cpu().numpy() # (1, 1024, 1024)
            mr_slice = mr_slice.detach().cpu().numpy() # (1, 3, 1024, 1024)
            
            # save the results into a dict named "slice_mr_ct_1024_idxz.hdf5"
            mr_mean = np.mean(mr_slice)
            ct_mean = np.mean(ct_slice)
            if mr_mean > 1e-3 and ct_mean > 1e-3:
                log_message(f"[{data_folder}][{idx_case+1}/{n_cases}][{idx_z}/{res_z}] MR shape: {mr_slice.shape}, CT shape: {ct_slice.shape}, MR mean: {mr_mean}, CT mean: {ct_mean}, Saved MR and CT")
                
                save_name = os.path.join(case_path, f"slice_mr_ct_1024_{idx_z:03d}.hdf5")
                with h5py.File(save_name, "w") as f:
                    f.create_dataset("mr", data=mr_slice, compression="gzip", compression_opts=4)
                    f.create_dataset("ct", data=ct_slice, compression="gzip", compression_opts=4)
                f.close()
            else:
                log_message(f"[{data_folder}][{idx_case+1}/{n_cases}][{idx_z}/{res_z}] Warning: MR mean: {mr_mean}, CT mean: {ct_mean}, not saved")

# [data/MIMRTL_Brain][1/777][121/124] MR slice shape: torch.Size([1, 3, 1024, 1024])
# [data/MIMRTL_Brain][1/777][121/124] MR embedding shape: (1, 768, 64, 64), (1, 768, 64, 64), (1, 768, 64, 64), (1, 768, 64, 64), (1, 256, 64, 64)
# [data/MIMRTL_Brain][1/777][121/124] Saved MR and CT