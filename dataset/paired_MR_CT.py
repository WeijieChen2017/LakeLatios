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
    def __init__(self, path_MR, path_CT, stage="train", transform=None, subset_fraction=1.0):
        self.path_MR = path_MR
        self.path_CT = path_CT
        self.subset_fraction = subset_fraction
        self.transform = transform
        list_MR_full = sorted(glob.glob(self.path_MR+"/"+stage+"/*.npy"))
        list_CT_full = sorted(glob.glob(self.path_CT+"/"+stage+"/*.npy"))
        # check whether the length of the two lists are the same
        assert len(self.list_MR) == len(self.list_CT)
        # check whether the file names are the same
        for i in range(len(self.list_MR)):
            assert self.list_MR[i].split("/")[-1] == self.list_CT[i].split("/")[-1]
        self.data_path = list(zip(self.list_MR, self.list_CT))

        # Selecting a subset of data
        total_samples = len(list_MR_full)
        step = int(1 / subset_fraction)
        indices = range(0, total_samples, step)  # Taking every nth index

        self.list_MR = [list_MR_full[i] for i in indices]
        self.list_CT = [list_CT_full[i] for i in indices]
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
        # select CT in the middle slice from 3x1024x1024 to 1024x1024
        CT = CT[1, :, :]
        CT = CT.unsqueeze(0)

        sample = {"MR": MR, "CT": CT}

        return sample
    
# define the training and validation transform
# resize to 1024x1024
# random horizontal flip
# random vertical flip
# random rotation
# H, W, C -> C, H, W
# has pre-normalized to 0 mean and 1 std
# trainval_transform = transforms.Compose([
#     transforms.Resize((1024, 1024), interpolation=2),
#     transforms.ToTensor(),
# ])

# define the test transform
# resize to 1024x1024
# normalize to 0 mean and 1 std
# test_transform = transforms.Compose([
#     transforms.Resize((1024, 1024)),
#     transforms.ToTensor(),
# ])
