import h5py
import torch
from torch.utils.data import Dataset, DataLoader

class small_hdf5_dataset(Dataset):
    def __init__(self, file_path_list, required_keys=[], transform=None):
        """
        Args:
            file_paths (list of str): List of HDF5 file paths.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.file_path_list = file_path_list
        self.required_keys = required_keys
        self.transform = transform

    def __len__(self):
        return len(self.file_path_list)

    def __getitem__(self, idx):
        with h5py.File(self.file_path_list[idx], 'r') as f:
            data = {}
            only_key = list(f.keys())[0]
            for key in self.required_keys:
                data[key] = f[only_key][key][()]

        return data

class slice_hdf5_dataset(Dataset):
    def __init__(self, file_path_list, required_keys=[], training_verbose=False, transform=None):
        """
        Args:
            file_paths (list of str): List of HDF5 file paths.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.file_path_list = file_path_list
        self.required_keys = required_keys
        self.training_verbose = training_verbose
        self.transform = transform

    def __len__(self):
        return len(self.file_path_list)

    def __getitem__(self, idx):
        with h5py.File(self.file_path_list[idx], 'r') as f:
            data = {}
            for key in self.required_keys:
                data[key] = f[key][()]

        if self.training_verbose:
            print(f"idx {idx}, {self.file_path_list[idx]}")
            for key in self.required_keys:
                print(f"key {key}, data[key].shape {data[key].shape}")
        return data

# # Usage
# file_paths = ['path/to/file1.hdf5', 'path/to/file2.hdf5', 'path/to/file3.hdf5']  # Your HDF5 files
# dataset = HDF5Dataset(file_paths)
# dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# # Now you can iterate over dataloader in your training loop
# for batch_data in dataloader:
#     # Your training process here
#     pass
