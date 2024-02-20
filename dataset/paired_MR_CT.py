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

    def __len__(self):
        return len(self.list_MR)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        MR = np.load(self.data_path[idx][0], allow_pickle=True)
        CT = np.load(self.data_path[idx][1], allow_pickle=True)
        # H, W, C -> C, H, W
        MR = MR.transpose((2, 0, 1))
        CT = CT.transpose((2, 0, 1))
        # convert to tensor
        MR = torch.from_numpy(MR).float()
        CT = torch.from_numpy(CT).float()
        # transform
        if self.transform:
            MR = self.transform(MR)
            CT = self.transform(CT)
        # random augmentation
        MR, CT = paird_random_augmentation(MR, CT)
        sample = {"MR": MR, "CT": CT}

        return sample
    
# define the training and validation transform
# resize to 1024x1024
# random horizontal flip
# random vertical flip
# random rotation
# H, W, C -> C, H, W
# has pre-normalized to 0 mean and 1 std
trainval_transform = transforms.Compose([
    transforms.Resize((1024, 1024)),
    transforms.ToTensor(),
])

# define the test transform
# resize to 1024x1024
# normalize to 0 mean and 1 std
test_transform = transforms.Compose([
    transforms.Resize((1024, 1024)),
    transforms.ToTensor(),
])
