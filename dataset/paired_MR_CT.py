import glob
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms

class DynamicNormalize:
    def __init__(self):
        self.mean = 0
        self.std = 1
        self.n_samples = 0

    def update_stats(self, tensor):
        # Calculate the mean and std for the current tensor
        batch_mean = tensor.mean()
        batch_std = tensor.std()

        # Update the overall mean and std using a running average
        self.mean = (self.mean * self.n_samples + batch_mean) / (self.n_samples + 1)
        self.std = (self.std * self.n_samples + batch_std) / (self.n_samples + 1)
        self.n_samples += 1

    def __call__(self, tensor):
        # Update stats with the current image
        self.update_stats(tensor)
        
        # Normalize the current image
        return (tensor - self.mean) / (self.std + 1e-6)


# create customized dataset
# folder structure:
# - data
#   - train
#     - 0001.npy
#     - 0002.npy
#     - ...
#   - val
#     - 0011.npy
#     - 0012.npy
#     - ...
#   - test
#     - 0021.npy
#     - 0022.npy
#     - ...

class PairedMRCTDataset(Dataset):
    def __init__(self, path_MR, path_CT, stage="train", transform=None):
        self.path_MR = path_MR
        self.path_CT = path_CT
        self.transform = transform
        self.list_MR = sorted(glob.glob(self.path_MR+"/"+stage+"/*.npy"))
        self.list_CT = sorted(glob.glob(self.path_CT+"/"+stage+"/*.npy"))
        # check whether the length of the two lists are the same
        assert len(self.list_MR) == len(self.list_CT)
        # check whether the file names are the same
        for i in range(len(self.list_MR)):
            assert self.list_MR[i].split("/")[-1] == self.list_CT[i].split("/")[-1]
        self.data_path = list(zip(self.list_MR, self.list_CT))
        self.dynamic_normalizer = DynamicNormalize()  # Instantiate the dynamic normalizer

    def __len__(self):
        return len(self.list_MR)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        MR = np.load(self.data_path[idx][0], allow_pickle=True)
        CT = np.load(self.data_path[idx][1], allow_pickle=True)
        sample = {'MR': MR, 'CT': CT}
        if self.transform:
            sample = self.transform(sample)

        # Apply dynamic normalization to your data
        MR_normalized = self.dynamic_normalizer(sample['MR'])
        CT_normalized = self.dynamic_normalizer(sample['CT'])
        sample = {'MR': MR_normalized, 'CT': CT_normalized}

        return sample
    
# define the training and validation transform
# resize to 1024x1024
# random horizontal flip
# random vertical flip
# random rotation
# random intensity shift
# normalize to 0 mean and 1 std
train_transform = transforms.Compose([
    transforms.Resize((1024, 1024)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(30),
    transforms.ToTensor(),
])

# define the test transform
# resize to 1024x1024
# normalize to 0 mean and 1 std
val_transform = transforms.Compose([
    transforms.Resize((1024, 1024)),
    transforms.ToTensor(),
])
