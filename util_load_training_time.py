import os
import time
import glob
import numpy as np

directory_path_list = [
    "proj/decoder_Deconv",
    "proj/decoder_PP_pct100",
    "proj/decoder_PP_pct20",
    "proj/decoder_PyramidPooling",
    "proj/decoder_UNetMONAI",
    "proj/decoder_UNetMONAI_pct100",
    "proj/decoder_UNetMONAI_pct20",
    "proj/decoder_UNetMONAI_pct5",
    "proj/decoder_UNETR",
    "proj/decoder_UNETR_pct100",
    "proj/decoder_UNETR_pct20",
    "proj/MR2CT_InstanceUNet",
    "proj/original_conv_output_PP_ViTheads_encoder_MedSAM",
    "proj/original_conv_output_ViTheads_encoder_MedSAM",
    "proj/output_PP_ViTheads_encoder_MedSAM",
    "proj/output_UNETR_IN_ViTheads_encoder_MedSAM",
    "proj/output_UNETR_ViTheads_encoder_MedSAM",
    "proj/prembed_Deconv_pct5_MB",
    "proj/prembed_PP_pct5_MB",
    "proj/prembed_UNETR_pct5_MB",
    "proj/small_prembed_Deconv_pct5_MB",
    "proj/small_prembed_PP_pct5_MB",
    "proj/small_prembed_UNETR_pct5_MB",
    "proj/UNetMONAI_pct100",
    "proj/UNetMONAI_pct100_43",
    "proj/UNetMONAI_pct20",
    "proj/UNetMONAI_pct5",
]

# Dictionary to hold file names and their modified times
file_modified_times = {}

for directory_path in directory_path_list:

    mod_time_list = []
    png_list = sorted(glob.glob(os.path.join(directory_path, "epoch_*.png")))

    for png_path in png_list:
        # Get the filename
        filename = os.path.basename(png_path)
        # Get the file path
        file_path = os.path.join(directory_path, filename)
        # Get the modified time
        mod_time = os.path.getmtime(file_path)
        # Convert the modified time to a more readable format (optional)
        readable_time = time.ctime(mod_time)
    
        # Save the modified time to the dictionary
        mod_time_list.append(readable_time)

    # Save the modified time list to the dictionary
    file_modified_times[directory_path] = mod_time_list

# save the dictionary to a npy file
np.save("file_modified_times.npy", file_modified_times)
