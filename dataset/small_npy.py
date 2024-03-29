import torch
import numpy as np
from torch.utils.data import Dataset

class slice_npy(Dataset):
    def __init__(self, file_dict_list, required_keys, is_channel_last=False, transform=None):
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

    def __len__(self):
        return len(self.file_dict_list)

    def __getitem__(self, idx):
        data = {}
        for key in self.required_keys:
            if self.is_channel_last:
                data[key] = np.load(self.file_dict_list[idx][key], allow_pickle=True).transpose(2, 0, 1)
            else:
                data[key] = np.load(self.file_dict_list[idx][key], allow_pickle=True)

        return data
