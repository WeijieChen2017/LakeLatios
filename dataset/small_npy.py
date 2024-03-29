import torch
import numpy as np
from torch.utils.data import Dataset

class slice_npy(Dataset):
    def __init__(self, file_dict_list, required_keys, transform=None):
        """
        Args:
            file_paths (list of str): List of npy file paths.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.file_dict_list = file_dict_list
        self.required_keys = required_keys
        self.transform = transform

    def __len__(self):
        return len(self.file_dict_list)

    def __getitem__(self, idx):
        data = {}
        for key in self.required_keys:
            data[key] = np.load(self.file_dict_list[idx][key], allow_pickle=True)

        return data
