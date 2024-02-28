# this is to load a checkpoint and print out all keys
import torch
from collections import OrderedDict
checkpoint = torch.load("./proj/decoder_PyramidPooling/best_model.pth")
# print it with structure
for key in checkpoint.keys():
    print(key)
    