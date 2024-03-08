# data folder:

# data/MIMRTL_Brain
# data/SynthRad_Brain
# data/SynthRad_Pelvis

# for each folder path, this is like:
# /path_to_data_folder/name_of_case/MedSAM_embedding_gzip4.hdf5

# we load this, and save the data into individual .hdf5 files for keyname in 3 digits, like: pack_000.hdf5, pack_001.hdf5, ...

import os
import glob
import time
import json
import nibabel as nib
import h5py
import numpy as np

data_folder_list = [
    ["data/MIMRTL_Brain", 0, 33],
    ["data/MIMRTL_Brain", 33, 219],
    ["data/MIMRTL_Brain", 219, 405],
    ["data/MIMRTL_Brain", 405, 591],
    ["data/MIMRTL_Brain", 591, 777],
    ["data/SynthRad_Brain", 0, 23],
    ["data/SynthRad_Brain", 23, 181],
    ["data/SynthRad_Pelvis", 0, -1],
]

# take the user input to determine which data folder to process
input_idx = int(input("Which data folder to process? (0-7) "))
data_folder_to_process, start_case_num, end_case_num = data_folder_list[input_idx]

for data_folder in [data_folder_to_process]:
    print(f"Processing {data_folder}")
    folder_name = data_folder.split("/")[-1]
    case_list = sorted(glob.glob(os.path.join(data_folder, "*/")))
    case_list = case_list[start_case_num:end_case_num]
    n_cases = len(case_list)

    for idx_case, case_path in enumerate(case_list):
        start_time = time.time()
        print(f"[{data_folder}][{idx_case+1+start_case_num}/{n_cases+start_case_num}] Processing {case_path}")
        hdf5_path = os.path.join(case_path, "MedSAM_embedding_gzip4.hdf5")
        hdf5_data = h5py.File(hdf5_path, "r")
        print(f"[{data_folder}][{idx_case+1+start_case_num}/{n_cases+start_case_num}] Loaded {hdf5_path}")

        key_list = list(hdf5_data.keys())
        n_keys = len(key_list)
        for idx_key in range(n_keys):
            new_key = f"pack_{int(idx_key):03d}.hdf5"
            new_path = os.path.join(case_path, new_key)
            with h5py.File(new_path, "w") as f:
                grp = f.create_group(key_list[idx_key])
                grp.create_dataset("mr_emb_head_3", data=hdf5_data[key_list[idx_key]]["mr_emb_head_3"], compression="gzip", compression_opts=4)
                grp.create_dataset("mr_emb_head_6", data=hdf5_data[key_list[idx_key]]["mr_emb_head_6"], compression="gzip", compression_opts=4)
                grp.create_dataset("mr_emb_head_9", data=hdf5_data[key_list[idx_key]]["mr_emb_head_9"], compression="gzip", compression_opts=4)
                grp.create_dataset("mr_emb_head_12", data=hdf5_data[key_list[idx_key]]["mr_emb_head_12"], compression="gzip", compression_opts=4)
                grp.create_dataset("mr_emb_head_neck", data=hdf5_data[key_list[idx_key]]["mr_emb_head_neck"], compression="gzip", compression_opts=4)
                grp.create_dataset("mr", data=hdf5_data[key_list[idx_key]]["mr"], compression="gzip", compression_opts=4)
                grp.create_dataset("ct", data=hdf5_data[key_list[idx_key]]["ct"], compression="gzip", compression_opts=4)
            print(f"[{data_folder}][{idx_case+1+start_case_num}/{n_cases+start_case_num}] Saved {new_path}")

        hdf5_data.close()

        end_time = time.time()
        # write to txt file and record the time in seconds
        with open(os.path.join("./"+folder_name+"_time_disassemble.txt"), "a") as f:
            f.write(f"Case {idx_case+1+start_case_num}/{n_cases+start_case_num}: {end_time-start_time:.4f} seconds\n")

        # delete the original hdf5 file
        os.remove(hdf5_path)
        print(f"[{data_folder}][{idx_case+1+start_case_num}/{n_cases+start_case_num}] Deleted {hdf5_path}")
                
