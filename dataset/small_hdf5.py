import h5py
import torch
from torch.utils.data import Dataset, DataLoader

class small_hdf5_dataset(Dataset):
    def __init__(self, file_paths, transform=None):
        """
        Args:
            file_paths (list of str): List of HDF5 file paths.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.file_paths = file_paths
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        with h5py.File(self.file_paths[idx], 'r') as f:
            # Assuming data is stored under the 'data' key. Adjust as needed.
            data = f['data'][()]
        
        if self.transform:
            data = self.transform(data)
        
        # Convert data to PyTorch tensor or any other processing you need
        data = torch.tensor(data, dtype=torch.float32)

        return data

# # Usage
# file_paths = ['path/to/file1.hdf5', 'path/to/file2.hdf5', 'path/to/file3.hdf5']  # Your HDF5 files
# dataset = HDF5Dataset(file_paths)
# dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# # Now you can iterate over dataloader in your training loop
# for batch_data in dataloader:
#     # Your training process here
#     pass
