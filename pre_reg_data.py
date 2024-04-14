import os
import glob
import numpy as np
import nibabel as nib

data_folder = "data/LumbarSpine/"
# find folders in the data folder
folder_list = sorted(glob.glob(data_folder+"*/"))

# for folder in folder_list:
#     print(folder)
#     # find nii files in the folder
#     mr_file = os.path.join(folder, "T1WI.nii.gz")
#     ct_file = os.path.join(folder, "C.nii.gz")
#     nc_file = os.path.join(folder, "NC.nii.gz")

#     mr_file = nib.load(mr_file)
#     ct_file = nib.load(ct_file)
#     nc_file = nib.load(nc_file)

#     mr_data = mr_file.get_fdata()
#     ct_data = ct_file.get_fdata()
#     nc_data = nc_file.get_fdata()

#     proc_mr_data = np.clip(mr_data, 0, 3000)
#     proc_nc_data = nc_data + 1024
#     proc_nc_data = np.clip(proc_nc_data, 0, 3000)
#     proc_ct_data = ct_data + 1024
#     proc_ct_data = np.clip(proc_ct_data, 0, 3000)

#     proc_mr_file = nib.Nifti1Image(proc_mr_data, mr_file.affine, mr_file.header)
#     proc_nc_file = nib.Nifti1Image(proc_nc_data, nc_file.affine, nc_file.header)
#     proc_ct_file = nib.Nifti1Image(proc_ct_data, ct_file.affine, ct_file.header)

#     proc_mr_filename = os.path.join(folder, "proc_T1WI.nii.gz")
#     proc_nc_filename = os.path.join(folder, "proc_NC.nii.gz")
#     proc_ct_filename = os.path.join(folder, "proc_CT.nii.gz")

#     nib.save(proc_mr_file, proc_mr_filename)
#     nib.save(proc_nc_file, proc_nc_filename)
#     nib.save(proc_ct_file, proc_ct_filename)

#     print(f"Saved processed files to {proc_mr_filename}, {proc_nc_filename}, {proc_ct_filename}")

import ants

for folder in folder_list:
    print(folder)
    
    mr_path = os.path.join(folder, "proc_T1WI.nii.gz")
    ct_path = os.path.join(folder, "proc_CT.nii.gz")
    nc_path = os.path.join(folder, "proc_NC.nii.gz")

    mr_ants_img = ants.image_read(mr_path)
    ct_ants_img = ants.image_read(ct_path)
    nc_ants_img = ants.image_read(nc_path)

    new_ct_img = mr_ants_img.new_image_like(mr_ants_img.numpy())
    new_nc_img = mr_ants_img.new_image_like(mr_ants_img.numpy())
    reg_ct_img = ants.registration(fixed=mr_ants_img, moving=ct_ants_img, type_of_transform='SyN')
    reg_nc_img = ants.registration(fixed=mr_ants_img, moving=nc_ants_img, type_of_transform='SyN')

    war_ct_img = ants.apply_transforms(fixed=mr_ants_img, moving=ct_ants_img, transformlist=reg_ct_img['warpedmovout'])
    war_nc_img = ants.apply_transforms(fixed=mr_ants_img, moving=nc_ants_img, transformlist=reg_nc_img['warpedmovout'])

    fwd_ct_img = ants.apply_transforms(fixed=mr_ants_img, moving=ct_ants_img, transformlist=reg_ct_img['fwdtransforms'])
    fwd_nc_img = ants.apply_transforms(fixed=mr_ants_img, moving=nc_ants_img, transformlist=reg_nc_img['fwdtransforms'])

    ants.image_write(fwd_ct_img, os.path.join(folder, "fwd_CT.nii.gz"))
    ants.image_write(fwd_nc_img, os.path.join(folder, "fwd_NC.nii.gz"))
    ants.image_write(war_ct_img, os.path.join(folder, "war_CT.nii.gz"))
    ants.image_write(war_nc_img, os.path.join(folder, "war_NC.nii.gz"))

    print(f"Saved registered files to {os.path.join(folder, 'fwd_CT.nii.gz')}, {os.path.join(folder, 'fwd_NC.nii.gz')}, {os.path.join(folder, 'war_CT.nii.gz')}, {os.path.join(folder, 'war_NC.nii.gz')}")
