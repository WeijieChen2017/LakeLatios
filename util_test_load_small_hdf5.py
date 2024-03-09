# given three hdf5 files, compare the loading time

import os
import h5py
import glob
import time

file_list = [
    "data/MIMRTL_Brain/00001/"
]
slice_list = sorted(glob.glob(os.path.join(file_list[0], "pack*.hdf5")))

for slice_path in slice_list:
    print(f"Loading {slice_path}", end="")
    start = time.time()
    with h5py.File(slice_path, "r") as f:
        only_key = list(f.keys())[0]
        print(f"Keys: {list(f.keys())}", end="")
        modalities = list(f[only_key].keys())
        print(f"Modalities: {modalities}", end="")
    print(f"Loading {slice_path} takes {time.time() - start:.2f} seconds")
    print()

# Loading data/MIMRTL_Brain/00001/pack_121.hdf5Keys: ['97']Modalities: ['ct', 'mr', 'mr_emb_head_12', 'mr_emb_head_3', 'mr_emb_head_6', 'mr_emb_head_9', 'mr_emb_head_neck']Loading data/MIMRTL_Brain/00001/pack_121.hdf5 takes 0.00 seconds

# Loading data/MIMRTL_Brain/00001/pack_122.hdf5Keys: ['98']Modalities: ['ct', 'mr', 'mr_emb_head_12', 'mr_emb_head_3', 'mr_emb_head_6', 'mr_emb_head_9', 'mr_emb_head_neck']Loading data/MIMRTL_Brain/00001/pack_122.hdf5 takes 0.01 seconds

# Loading data/MIMRTL_Brain/00001/pack_123.hdf5Keys: ['99']Modalities: ['ct', 'mr', 'mr_emb_head_12', 'mr_emb_head_3', 'mr_emb_head_6', 'mr_emb_head_9', 'mr_emb_head_neck']Loading data/MIMRTL_Brain/00001/pack_123.hdf5 takes 0.02 seconds
