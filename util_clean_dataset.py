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

for idx, data_path in enumerate(data_list):
    print(f"Processing {idx+1}/{len(data_list)}: {data_path}")
    with h5py.File(data_path, 'r') as f:
        # get keys
        keys = list(f.keys())
        # get data
        for key in keys:
            data = f[key][()]
            # check whether the data(pytorch tensor) is valid, if not, change the filename to .invalid
            # rule: the num of values < 1e-3 should be less than 1/1000 of the total num of values
            if torch.sum(data < 1e-3) > data.numel() / 1000:
                print(f"---------> Invalid data: {data_path}")
                os.rename(data_path, data_path + ".invalid")

            