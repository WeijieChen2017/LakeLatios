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
        # data_list = [data.cpu().numpy() for data in [data1, data2, data3, data4]]
        # stacked_data = np.concatenate(data_list, axis=0)

        mr = np.concatenate([data_hdf5[train_batch_list[idx_train_batch][str(idx_slice)]]["mr"][()] for idx_slice in range(len(train_batch_list[idx_train_batch]))], axis=0)
        ct = np.concatenate([data_hdf5[train_batch_list[idx_train_batch][str(idx_slice)]]["ct"][()] for idx_slice in range(len(train_batch_list[idx_train_batch]))], axis=0)
        mr_emb_head_3 = np.concatenate([data_hdf5[train_batch_list[idx_train_batch][str(idx_slice)]]["mr_emb_head_3"][()] for idx_slice in range(len(train_batch_list[idx_train_batch]))], axis=0)
        mr_emb_head_6 = np.concatenate([data_hdf5[train_batch_list[idx_train_batch][str(idx_slice)]]["mr_emb_head_6"][()] for idx_slice in range(len(train_batch_list[idx_train_batch]))], axis=0)
        mr_emb_head_9 = np.concatenate([data_hdf5[train_batch_list[idx_train_batch][str(idx_slice)]]["mr_emb_head_9"][()] for idx_slice in range(len(train_batch_list[idx_train_batch]))], axis=0)
        mr_emb_head_12 = np.concatenate([data_hdf5[train_batch_list[idx_train_batch][str(idx_slice)]]["mr_emb_head_12"][()] for idx_slice in range(len(train_batch_list[idx_train_batch]))], axis=0)
        mr_emb_head_neck = np.concatenate([data_hdf5[train_batch_list[idx_train_batch][str(idx_slice)]]["mr_emb_head_neck"][()] for idx_slice in range(len(train_batch_list[idx_train_batch]))], axis=0)
    
        print(f"Training batch {idx_train_batch+1}/{n_train_batch} in case {idx_batch+1}/{n_train}")
        print(f"---> mr.shape: {mr.shape}, ct.shape: {ct.shape}, mr_emb_head_3.shape: {mr_emb_head_3.shape}, mr_emb_head_6.shape: {mr_emb_head_6.shape}, mr_emb_head_9.shape: {mr_emb_head_9.shape}, mr_emb_head_12.shape: {mr_emb_head_12.shape}, mr_emb_head_neck.shape: {mr_emb_head_neck.shape}")