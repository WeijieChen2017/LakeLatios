# given the dataset file, load all the data and check whether the data is valid

import os
import h5py
import glob
import torch
import numpy as np

dataset_name = "MIMRTL_Brain"
data_folder = "data/"+dataset_name
data_list = sorted(glob.glob(data_folder+"/*/slice_*.hdf5"))
print("Searching for data in the following folders:", data_folder)
print("Found", len(data_list), "data files")
input("Press Enter to continue...")

for idx, data_path in enumerate(data_list):
    # print(f"Processing {idx+1}/{len(data_list)}: {data_path}")
    with h5py.File(data_path, 'r') as f:
        # get keys
        keys = list(f.keys())
        # get data
        for key in keys:
            data = f[key][()]
            if np.mean(data) < 1e-3:
                print(f"Warning: {key} in {data_path} has mean {np.mean(data)}")
                # rename the file
                os.system(f"mv {data_path} {data_path+'.invalid'}")
            