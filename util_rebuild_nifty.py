# load all nifty in "./data/nifty/CT/"
save_folder = "./results/nifiti_test_PP_pct20_v1/"
pred_folder = "./results/test_PP_pct20_v1/"

import numpy as np
import os
import nibabel as nib
import glob
from skimage.transform import resize

if not os.path.exists(save_folder):
    os.makedirs(save_folder)

# find all nifty files
data_folder = "./data/MR2CT/nifty/"
file_list = sorted(glob.glob(data_folder + "/CT/*.nii.gz"))

# create a list to store all filename
file_name_list = []
for filename in file_list:
    file_name_list.append(filename.split("/")[-1].split(".")[0])
    # print(filename.split("/")[-1].split(".")[0])
n_file = len(file_name_list)

# iterate all the files
for idx, filename in enumerate(file_name_list):
    MR_path = data_folder + "/MR/" + filename + ".nii.gz"
    CT_path = data_folder + "/CT/" + filename + ".nii.gz"
    MR_file = nib.load(MR_path)
    CT_file = nib.load(CT_path)
    MR_data = MR_file.get_fdata()
    CT_data = CT_file.get_fdata()
    # print the progress
    print(f"processing {idx+1}/{n_file}: ", filename)
    print("MR_data.shape: ", MR_data.shape, "CT_data.shape: ", CT_data.shape)

    # load all predictions
    pred_data = np.zeros(CT_data.shape)
    idx_z = CT_data.shape[2]
    for idx in range(idx_z):
        pred_path = pred_folder + filename + "_{:04d}".format(idx) + ".npy"
        # load is 1024x1024, and we need to downsample it to 256x256
        # try to load the prediction
        try:
            pred_hr = np.load(pred_path, allow_pickle=True)
            pred_data[:, :, idx] = resize(pred_hr, (256, 256), anti_aliasing=True) * 4024 - 1024
        except:
            print("file not found: ", pred_path)

    # save the prediction as nifty
    save_path = save_folder + filename + ".nii.gz"
    pred_file = nib.Nifti1Image(pred_data, CT_file.affine, CT_file.header)
    nib.save(pred_file, save_path)
    print("save prediction as nifty: ", save_path)
    print("")

