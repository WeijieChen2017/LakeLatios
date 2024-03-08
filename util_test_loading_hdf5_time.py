# given three hdf5 files, compare the loading time

import h5py
import time

file_list = [
    "data/MIMRTL_Brain/00001/MedSAM_embedding.hdf5", # 8.84GB
    "data/MIMRTL_Brain/00001/MedSAM_embedding_gzip.hdf5", # 6.96GB
    "data/MIMRTL_Brain/00001/MedSAM_embedding_lzf.hdf5", # 
]

for file_path in file_list:
    print(f"Loading {file_path}")
    start = time.time()
    with h5py.File(file_path, "r") as f:
        print(f"Keys: {list(f.keys())}")
    print(f"Loading {file_path} takes {time.time() - start:.2f} seconds")
    print()



    