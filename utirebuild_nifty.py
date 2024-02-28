# load all nifty in "./data/nifty/CT/"
save_folder = "./results/test_PP_v1_nifty/"

import numpy as np
import os
import nibabel as nib
import glob

# find all nifty files
data_folder = "./data/nifty/CT/"
file_list = glob.glob(data_folder + "*.nii.gz")

# create a list to store all filename
file_name_list = []
for filename in file_list:
    file_name_list.append(filename.split("/")[-1].split(".")[0])
    print(filename.split("/")[-1].split(".")[0])

# # iterate all the files
# for filename in file_list:
#     img = nib.load(filename)
#     img_data = img.get_fdata()
#     print("img_data.shape: ", img_data.shape)

#     # save the file
#     savename = os.path.join(save_folder, filename.split("/")[-1].split(".")[0] + ".npy")
#     np.save(savename, img_data)
