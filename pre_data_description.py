# you will load a folder of data, now you need to search all .nii.gz files in the folder
# and output stats of each file using a file tree at the folder path
# then output a overall stats of the folder

folder_path = "data/LumbarSpine/"
import os
import glob
import nibabel as nib
import numpy as np

# get all the nii.gz files in the folder
# the folder is like data/data_folder/path_to_case_num/*.nii.gz

file_list = sorted(glob.glob(os.path.join(folder_path, "*/*.nii.gz")))
file_list = file_list + sorted(glob.glob(os.path.join(folder_path, "*/*.nii")))
print("Found", len(file_list), "files")

folder_level_stat = {}

def compare_stat(basenmae, stat):
    if basename not in folder_level_stat:
        folder_level_stat[basename] = {}
        folder_level_stat[basename]["mean"] = [stat["mean"]]
        folder_level_stat[basename]["std"] = [stat["std"]]
        folder_level_stat[basename]["shape"] = [stat["shape"]]
        folder_level_stat[basename]["max"] = stat["max"]
        folder_level_stat[basename]["min"] = stat["min"]

    else:
        # keys:
        # mean, std, shape, max, min, sum
        if folder_level_stat[basename]["max"] < stat["max"]:
            folder_level_stat[basename]["max"] = stat["max"]
        if folder_level_stat[basename]["min"] > stat["min"]:
            folder_level_stat[basename]["min"] = stat["min"]
        if not stat["shape"] in folder_level_stat[basename]["shape"]:
            folder_level_stat[basename]["shape"].append(stat["shape"])
        folder_level_stat[basename]["std"].append(stat["std"])
        folder_level_stat[basename]["mean"].append(stat["mean"])

for file_path in file_list:
    print("Loading", file_path)
    basename = os.path.basename(file_path)
    case_path = os.path.dirname(file_path)

    # load the data
    data = nib.load(file_path).get_fdata()
    mean = np.mean(data)
    std = np.std(data)
    shape = data.shape
    max = np.max(data)
    min = np.min(data)
    file_stat = {
        "mean": mean,
        "std": std,
        "shape": shape,
        "max": max,
        "min": min,
    }
    compare_stat(basename, file_stat)

    # output the stat in the folder in a txt
    with open(os.path.join(case_path, "stat.txt"), "a") as f:
        f.write(f"File: {basename}\n")
        # write in lines
        f.write(f"mean: {mean}\n")
        f.write(f"std: {std}\n")
        f.write(f"shape: {shape}\n")
        f.write(f"max: {max}\n")
        f.write(f"min: {min}\n\n")

    # if this is a .nii file, load the data, header, and affine, and output as a .nii.gz file
    if file_path.endswith(".nii"):
        print("Outputing", file_path)
        # output the data as a .nii.gz file
        new_file_path = file_path + ".gz"
        old_file = nib.load(file_path)
        new_file = nib.Nifti1Image(data, old_file.affine, old_file.header)
        nib.save(new_file, new_file_path)
        os.system(f"rm {file_path}")
        print("Outputed", new_file_path)

# output the overall stat of the folder
for basename in folder_level_stat.keys():
    mean = np.mean(folder_level_stat[basename]["mean"])
    std = np.mean(folder_level_stat[basename]["std"])
    shape = folder_level_stat[basename]["shape"]
    max = folder_level_stat[basename]["max"]
    min = folder_level_stat[basename]["min"]
    with open(os.path.join(folder_path, "stat.txt"), "a") as f:
        f.write(f"Folder: {basename}\n")
        # write in lines
        f.write(f"mean: {mean}\n")
        f.write(f"std: {std}\n")
        f.write(f"shape: {shape}\n")
        f.write(f"max: {max}\n")
        f.write(f"min: {min}\n\n")

print("Finished")