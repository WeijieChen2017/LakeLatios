# input ViT_heads: list with 15 heads of ndarray (B, 64, 64, embed_dim)
# input save_path: 
# output 15 images of 
# intensity 
# data distribution
# cosine similarity

import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib import cm

def viz_ViT_heads(ViT_heads, save_path):

    # save all intensity images in one figure
    fig, axs = plt.subplots(3, 5, figsize=(20, 12))
    for i in range(3):
        for j in range(5):
            data = ViT_heads[i*5+j]
            # take mean over the last dimension
            data = np.squeeze(np.mean(data, axis=-1))
            axs[i, j].imshow(data, cmap='gray')
            axs[i, j].axis('off')
            axs[i, j].set_title(f'head {i*5+j}')
    plt.savefig(save_path+"/intensity.png")
    plt.close()

    # plot the data distribution in one column
    fig, axs = plt.subplots(15, 1, figsize=(5, 30))
    for i in range(15):
        data = ViT_heads[i]
        data = np.squeeze(np.mean(data, axis=-1))
        axs[i].hist(data.flatten(), bins=100)
        axs[i].set_title(f'head {i}')
    plt.savefig(save_path+"/data_distribution.png")
    plt.close()

    


