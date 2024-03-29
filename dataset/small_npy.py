import os
import torch
import numpy as np
from torch.utils.data import Dataset

class slice_npy(Dataset):
    def __init__(self, file_dict_list, required_keys, 
                 is_channel_last=False, return_filename=False,
                 init_verbose=False,
                 transform=None):
        """
        Args:
            file_paths (list of str): List of npy file paths.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.file_dict_list = file_dict_list
        self.required_keys = required_keys
        self.transform = transform
        self.is_channel_last = is_channel_last
        self.return_filename = return_filename
        self.init_verbose = init_verbose

        if self.init_verbose:
            print("---> slice_npy dataset initialized.")
            print("---> Number of samples: ", len(self.file_dict_list))
            print("---> Required keys: ", self.required_keys)
            print("---> is_channel_last: ", self.is_channel_last)
            print("---> return_filename: ", self.return_filename)

    def __len__(self):
        return len(self.file_dict_list)

    def __getitem__(self, idx):
        data = {}
        for key in self.required_keys:
            if self.is_channel_last:
                data[key] = np.load(self.file_dict_list[idx][key], allow_pickle=True).transpose(2, 0, 1)
            else:
                data[key] = np.load(self.file_dict_list[idx][key], allow_pickle=True)
        if self.return_filename:
            filename = os.path.basename(self.file_dict_list[idx]["mr"])
            print("$$$$$$$$$$", filename)
            filename = filename.split(".")[0]
            print("$$$$$$$$$$", filename)
            return data, filename
        else:
            return data
