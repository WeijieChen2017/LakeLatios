# given three hdf5 files, compare the loading time

import os
import h5py
import glob
import time

file_list = [
    "data/MIMRTL_Brain/00001/*"
]
slice_list = sorted(glob.glob(os.path.join(file_list[0], "pack*.hdf5")))

for file_path in file_list:
    print(f"Loading {file_path}")
    start = time.time()
    with h5py.File(file_path, "r") as f:
        print(f"Keys: {list(f.keys())}")
    print(f"Loading {file_path} takes {time.time() - start:.2f} seconds")
    print()