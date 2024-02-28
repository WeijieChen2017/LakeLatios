import glob
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms

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

# >>> import numpy as np
# >>> data = np.load("02187_0021.npy", allow_pickle=True)
# >>> print(data.shape)
# (3, 256, 256)
# >>> print(np.amax(data))
# 0.6369983510077
# >>> print(np.ain(data))
# Traceback (most recent call last):
#   File "<stdin>", line 1, in <module>
#   File "/shares/mimrtl/Users/Winston/anaconda3/envs/mimrtl/lib/python3.9/site-packages/numpy/__init__.py", line 303, in __getattr__
#     raise AttributeError("module {!r} has no attribute "
# AttributeError: module 'numpy' has no attribute 'ain'
# >>> print(np.amin(data))
# 0.0
# >>> exit()

def paird_random_augmentation(mr, ct):
    # C, H, W
    # transforms.RandomHorizontalFlip(),
    if np.random.rand() > 0.5:
        mr = torch.flip(mr, [2])
        ct = torch.flip(ct, [2])
    # transforms.RandomVerticalFlip(),
    if np.random.rand() > 0.5:
        mr = torch.flip(mr, [1])
        ct = torch.flip(ct, [1])
    # transforms.RandomRotation(30),
    if np.random.rand() > 0.5:
        angle = np.random.randint(0, 360)
        mr = transforms.functional.rotate(mr, angle)
        ct = transforms.functional.rotate(ct, angle)
    return mr, ct



class PairedMRCTDataset_train(Dataset):
    def __init__(self, path_MR, path_CT, stage="train", transform=None, subset_fraction=1.0, out_channels=1):
        self.path_MR = path_MR
        self.path_CT = path_CT
        self.subset_fraction = subset_fraction
        self.transform = transform
        self.out_channels = out_channels
        self.list_MR_full = sorted(glob.glob(self.path_MR+"/"+stage+"/*.npy"))
        self.list_CT_full = sorted(glob.glob(self.path_CT+"/"+stage+"/*.npy"))
        # check whether the length of the two lists are the same
        assert len(self.list_MR_full) == len(self.list_CT_full)
        # check whether the file names are the same
        for i in range(len(self.list_MR_full)):
            assert self.list_MR_full[i].split("/")[-1] == self.list_CT_full[i].split("/")[-1]

        # Selecting a subset of data
        total_samples = len(self.list_MR_full)
        step = int(1 / subset_fraction)
        indices = range(0, total_samples, step)  # Taking every nth index

        self.list_MR = [self.list_MR_full[i] for i in indices]
        self.list_CT = [self.list_CT_full[i] for i in indices]
        self.data_path = list(zip(self.list_MR, self.list_CT))

        # Optionally, track the selected samples/indices
        self.used_samples = indices  # or self.list_MR to track by file paths

    def save_used_samples(self, filename):
        with open(filename, "w") as f:
            for item in self.used_samples:
                f.write("%s\n" % item)

    def __len__(self):
        return len(self.list_MR)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        MR = np.load(self.data_path[idx][0], allow_pickle=True)
        CT = np.load(self.data_path[idx][1], allow_pickle=True)

        # convert to tensor
        MR = torch.from_numpy(MR).float()
        CT = torch.from_numpy(CT).float()
        # add first dimension
        MR = MR.unsqueeze(0)
        CT = CT.unsqueeze(0)

        # resize from 3x256x256 to 3x1024x1024
        MR = transforms.functional.resize(MR, (1024, 1024), interpolation=2, antialias=True)
        CT = transforms.functional.resize(CT, (1024, 1024), interpolation=2, antialias=True)

        # # H, W, C -> C, H, W
        # MR = MR.transpose((2, 0, 1))
        # CT = CT.transpose((2, 0, 1))
        # have pre-normalized to 0 mean and 1 std
        # have converted to C, H, W

        # transform
        if self.transform:
            MR = self.transform(MR)
            CT = self.transform(CT)
        # random augmentation
        MR, CT = paird_random_augmentation(MR, CT)
        # squeeze the first dimension
        MR = MR.squeeze(0)
        CT = CT.squeeze(0)
        
        if self.out_channels == 1:
            # select CT in the middle slice from 3x1024x1024 to 1024x1024
            CT = CT[1, :, :]
            CT = CT.unsqueeze(0)

        # normalize to 0 mean and 1 std
        MR = transforms.functional.normalize(MR, mean=0.0, std=1.0)
        CT = transforms.functional.normalize(CT, mean=0.0, std=1.0)

        sample = {"MR": MR, "CT": CT}

        return sample


class PairedMRCTDataset_test(Dataset):
    def __init__(self, path_MR, path_CT, stage="train", transform=None, subset_fraction=1.0, out_channels=1):
        self.path_MR = path_MR
        self.path_CT = path_CT
        self.subset_fraction = subset_fraction
        self.transform = transform
        self.out_channels = out_channels
        self.list_MR_full = sorted(glob.glob(self.path_MR+"/"+stage+"/*.npy"))
        self.list_CT_full = sorted(glob.glob(self.path_CT+"/"+stage+"/*.npy"))
        # check whether the length of the two lists are the same
        assert len(self.list_MR_full) == len(self.list_CT_full)
        # check whether the file names are the same
        for i in range(len(self.list_MR_full)):
            assert self.list_MR_full[i].split("/")[-1] == self.list_CT_full[i].split("/")[-1]

        # Selecting a subset of data
        total_samples = len(self.list_MR_full)
        step = int(1 / subset_fraction)
        indices = range(0, total_samples, step)  # Taking every nth index

        self.list_MR = [self.list_MR_full[i] for i in indices]
        self.list_CT = [self.list_CT_full[i] for i in indices]
        self.data_path = list(zip(self.list_MR, self.list_CT))

        # Optionally, track the selected samples/indices
        self.used_samples = indices  # or self.list_MR to track by file paths

    def save_used_samples(self, filename):
        with open(filename, "w") as f:
            for item in self.used_samples:
                f.write("%s\n" % item)

    def __len__(self):
        return len(self.list_MR)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        MR = np.load(self.data_path[idx][0], allow_pickle=True)
        CT = np.load(self.data_path[idx][1], allow_pickle=True)

        # convert to tensor
        MR = torch.from_numpy(MR).float()
        CT = torch.from_numpy(CT).float()
        # add first dimension
        MR = MR.unsqueeze(0)
        CT = CT.unsqueeze(0)

        # resize from 3x256x256 to 3x1024x1024
        MR = transforms.functional.resize(MR, (1024, 1024), interpolation=2, antialias=True)
        CT = transforms.functional.resize(CT, (1024, 1024), interpolation=2, antialias=True)

        # transform
        if self.transform:
            MR = self.transform(MR)
            CT = self.transform(CT)
        # random augmentation
        # MR, CT = paird_random_augmentation(MR, CT)
        # squeeze the first dimension
        MR = MR.squeeze(0)
        CT = CT.squeeze(0)
        
        if self.out_channels == 1:
            # select CT in the middle slice from 3x1024x1024 to 1024x1024
            CT = CT[1, :, :]
            CT = CT.unsqueeze(0)

        # normalize to 0 mean and 1 std
        MR = transforms.functional.normalize(MR, mean=0.0, std=1.0)
        CT = transforms.functional.normalize(CT, mean=0.0, std=1.0)

        sample = {"MR": MR, "CT": CT}
        filename = self.data_path[idx][0].split("/")[-1]

        return sample, filename
