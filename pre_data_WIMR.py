# data folder:
# data/WIMR/*

# load every case in data_folder/ct.nii.gz
# shift data range by -1000
# save with the same name

import os
import glob
import nibabel as nib

data_folder = "data/WIMR/*"
data_list = sorted(glob.glob(os.path.join(data_folder, "ct.nii.gz")))

for idx, data_path in enumerate(data_list):
    print(f"Processing {idx+1}/{len(data_list)}: {data_path}")
    ct_file = nib.load(data_path)
    ct_data = ct_file.get_fdata()
    ct_data = ct_data - 1000
    new_file = nib.Nifti1Image(ct_data, affine=ct_file.affine, header=ct_file.header)
    nib.save(new_file, data_path)
    print(f"Saved {data_path}")