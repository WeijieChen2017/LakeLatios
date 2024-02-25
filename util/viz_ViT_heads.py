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
    # head_names = [f'patch_emb', f'pos_emb', f'head {i}' for i in range(12), f'neck']
    head_names = ['patch_emb', 'pos_emb', 'head 0', 'head 1', 'head 2', 'head 3', 'head 4', 'head 5', 'head 6', 'head 7', 'head 8', 'head 9', 'head 10', 'head 11', 'neck']
    for i in range(3):
        for j in range(5):
            data = ViT_heads[i*5+j]
            # take mean over the last dimension
            data = np.squeeze(np.mean(data, axis=-1))
            axs[i, j].imshow(data, cmap='gray')
            axs[i, j].axis('off')
            axs[i, j].set_title(head_names[i*5+j])
    plt.savefig(save_path+"/intensity.png")
    plt.close()

    # plot the data distribution in one column
    # let the subplot uniformly distributed in the vertical direction
    fig, axs = plt.subplots(15, 1, figsize=(5, 40))
    for i in range(15):
        data = ViT_heads[i]
        data = np.squeeze(np.mean(data, axis=-1))
        axs[i].hist(data.flatten(), bins=200)
        axs[i].set_title(head_names[i])
        # set the range from -1 to 1
        axs[i].set_xlim(-2, 2)
        # set the y axis to log scale
        axs[i].set_yscale('log')
        # adjust subplot
        plt.subplots_adjust(hspace=0.5)
        # tight layout
        plt.tight_layout()
    plt.savefig(save_path+"/data_distribution.png")
    plt.close()

    # # plot the cosine similarity in one column
    # for i in range(15):

    #     data = ViT_heads[i]
    #     # 64, 64, emb to 64*64, emb
    #     data = np.reshape(data, (64*64, -1))
    #     px_cos_sim = np.zeros((64*64, 64*64))
    #     for j in range(64*64):
    #         for k in range(64*64):
    #             px_cos_sim[j, k] = np.dot(data[j], data[k]) / (np.linalg.norm(data[j]) * np.linalg.norm(data[k]))
        
    #     fig, axs = plt.subplots(1, 1, figsize=(5, 5))
    #     axs.imshow(px_cos_sim, cmap='gray')
    #     axs.set_title(head_names[i])
    #     plt.savefig(save_path+f"/cosine_similarity_{i}.png")
    #     plt.close()

    
def viz_ViT_heads_zneck_z12_z9_z6_z3_out(ViT_heads, save_path):

    # save all intensity images in one figure
    fig, axs = plt.subplots(3, 2, figsize=(20, 12))
    # head_names = [f'patch_emb', f'pos_emb', f'head {i}' for i in range(12), f'neck']
    head_names = ['neck', 'head 12', 'head 9', 'head 6', 'head 3', 'out']
    for i in range(3):
        for j in range(2):
            data = ViT_heads[i*3+j]
            print(head_names[i*3+j], data.shape)
            # take mean over the last dimension
            data = np.squeeze(np.mean(data, axis=-1))
            axs[i, j].imshow(data, cmap='gray')
            axs[i, j].axis('off')
            axs[i, j].set_title(head_names[i*3+j])
    plt.savefig(save_path+"/intensity.png")
    plt.close()

    # plot the data distribution in one column
    # let the subplot uniformly distributed in the vertical direction
    fig, axs = plt.subplots(6, 1, figsize=(5, 30))
    for i in range(6):
        data = ViT_heads[i]
        data = np.squeeze(np.mean(data, axis=-1))
        axs[i].hist(data.flatten(), bins=200)
        axs[i].set_title(head_names[i])
        # set the range from -1 to 1
        axs[i].set_xlim(-2, 2)
        # set the y axis to log scale
        axs[i].set_yscale('log')
        # adjust subplot
        plt.subplots_adjust(hspace=0.5)
        # tight layout
        plt.tight_layout()
    plt.savefig(save_path+"/data_distribution.png")
    plt.close()