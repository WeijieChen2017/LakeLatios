cfg = {
    "data_folder": "data/MIMRTL_Brain",
    "batch_size": 4,
    "training_case" : 27,
    "validation_case" : 5,
    "random_seed": 42,
}



# load all cases and select the first part for tarining and the second part for validation
training_case = cfg["training_case"]
validation_case = cfg["validation_case"]
data_folder = cfg["data_folder"]
import glob
case_list = sorted(glob.glob(data_folder+"/*"))
training_list = case_list[:training_case]
validation_list = case_list[training_case:training_case+validation_case]
# output the training and validation list
for idx, item in enumerate(training_list):
    print(f"Training {idx+1}: {item}")
for idx, item in enumerate(validation_list):
    print(f"Validation {idx+1}: {item}")

n_train = len(training_list)
n_val = len(validation_list)
batch_size = cfg["batch_size"]

import h5py
import numpy as np
import random

for idx_batch, case in enumerate(training_list):
    hdf5_filename = case + "/MedSAM_embedding_gzip4.hdf5"
    data_hdf5 = h5py.File(hdf5_filename, "r")
    n_slice = len(data_hdf5.keys())
    slice_list = list(data_hdf5.keys())

    # shuffle the slice_list
    random.shuffle(slice_list)

    # divide the slice_list into batches
    # create the list like [batch1, batch2, ..., batch_n]
    # batch_n may be less than batch_size
    n_train_batch = n_slice // batch_size
    if n_slice % batch_size != 0:
        n_train_batch += 1
    train_batch_list = []
    for idx in range(n_train_batch):
        if idx == n_train_batch - 1:
            train_batch_list.append(slice_list[idx*batch_size:])
        else:
            train_batch_list.append(slice_list[idx*batch_size:(idx+1)*batch_size])
    
    # load the data
    for idx_train_batch in range(n_train_batch):
        mr = np.concatenate([data_hdf5[slice_name]["mr"][()] for slice_name in train_batch_list[idx_train_batch]], axis=0)
        