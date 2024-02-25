# here we load the pretrained model and finetune it
import argparse

if __name__ == "__main__":

    # run the parser to get the cfg path
    parser = argparse.ArgumentParser(description="Load configuration file path.")
    # Add an argument for the configuration file path with a default value
    parser.add_argument("--cfg_path", type=str, default="viz.json", help="Path to the configuration file.")
    
    # Parse the command-line arguments
    args = parser.parse_args()

    # Use args.cfg_path to access the configuration file path
    print("Configuration path:", args.cfg_path)

    # set GPU
    import os
    import json
    # load the cfg
    # cfg_path = "config_0221.json"
    cfg_path = "config/"+args.cfg_path
    cfg = json.load(open(cfg_path))
    print(cfg)
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root_dir = cfg["root_dir"]
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)

    # load libraries
    import torch.nn as nn
    import torch.optim as optim
    from torchvision import transforms
    from torch.utils.data import DataLoader
    from matplotlib import pyplot as plt

    from model import output_ViTheads_encoder_MedSAM
    from model import decoder_PyramidPooling_encoder_MedSAM
    from model import decoder_UNETR_encoder_MedSAM
    from dataset import PairedMRCTDataset_train
    from util import viz_ViT_heads
    from util import viz_ViT_heads_zneck_z12_z9_z6_z3_out
    from util import viz_ViT_heads_zneck_z12_z9_z6_z3_d12_d9_d6_d3

    # load the random seed
    random_seed = cfg["random_seed"]
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    import random
    import numpy as np
    np.random.seed(random_seed)
    random.seed(random_seed)

    # load the model as the "model_name"
    if cfg["model_name"] == "decoder_PyramidPooling_encoder_MedSAM":
        model = decoder_PyramidPooling_encoder_MedSAM(
            img_size=cfg["img_size"],
            patch_size=cfg["patch_size"],
            in_chans=cfg["in_chans"],
            out_chans=cfg["out_chans"],
            out_chans_pretrain=cfg["out_chans_pretrain"],
            embed_dim=cfg["embed_dim"],
            depth=cfg["depth"],
            num_heads=cfg["num_heads"],
            mlp_ratio=cfg["mlp_ratio"],
            qkv_bias=True if cfg["qkv_bias"] == "True" else False,
            norm_layer=nn.LayerNorm if cfg["norm_layer"] == "nn.LayerNorm" else None,
            act_layer=nn.GELU if cfg["act_layer"] == "nn.GELU" else None,
            use_abs_pos=True if cfg["use_abs_pos"] == "True" else False,
            use_rel_pos=False if cfg["use_rel_pos"] == "False" else True,
            rel_pos_zero_init=True if cfg["rel_pos_zero_init"] == "True" else False,
            window_size=cfg["window_size"],
            global_attn_indexes=cfg["global_attn_indexes"],
            verbose=True,
        )
    else:
        raise ValueError("model_name not found !")

    model.load_pretrain(cfg["pretrain_path"])
    model.to(device)

    # load the dataset
    # train_transform = transforms.Compose([
    #     transforms.Resize((1024, 1024)),
    # ])
    # val_transform = transforms.Compose([
    #     transforms.Resize((1024, 1024)),
    # ])
    dataset_train = PairedMRCTDataset_train(
        path_MR=cfg["data_path_MR"],
        path_CT=cfg["data_path_CT"],
        stage="train", 
        subset_fraction=cfg["subset_fraction"],
        #  transform=train_transform,
        out_channels=cfg["out_chans"],
    )
    dataset_val = PairedMRCTDataset_train(
        path_MR=cfg["data_path_MR"],
        path_CT=cfg["data_path_CT"],
        stage="val", 
        subset_fraction=cfg["subset_fraction"],
        #  transform=val_transform,
        out_channels=cfg["out_chans"],
    )
    dataset_train.save_used_samples(root_dir+"used_samples_train.txt")
    dataset_val.save_used_samples(root_dir+"used_samples_val.txt")

    train_loader = DataLoader(
        dataset_train, 
        batch_size=cfg["batch_size"], 
        shuffle=True, 
        num_workers=cfg["num_workers"], 
        pin_memory=True
    )
    val_loader = DataLoader(
        dataset_val, 
        batch_size=cfg["batch_size"], 
        shuffle=False, 
        num_workers=cfg["num_workers"], 
        pin_memory=True
    )

    # train the model
    best_val_loss = 1e10
    n_train = len(train_loader)
    n_val = len(val_loader)
    n_train_batch = n_train
    n_val_batch = n_val
    # output the parameters
    print(f"n_train: {n_train}, n_val: {n_val}, n_train_batch: {n_train_batch}, n_val_batch: {n_val_batch}")

    # train the model
    for epoch in range(cfg["epochs"]):

        # training
        model.train()
        batch_idx = 0
        for batch in train_loader:
            MR = batch["MR"]
            CT = batch["CT"]
            MR = MR.to(device)
            CT = CT.to(device)
           
            with torch.set_grad_enabled(True):
                _, ViT_heads_MR = model(MR) # B, 64, 64, Embed_dim
                _, ViT_heads_CT = model(CT) # B, 64, 64, Embed_dim

            save_folder_MR = root_dir+f"epoch{epoch}_batch{batch_idx}_MR"
            save_folder_CT = root_dir+f"epoch{epoch}_batch{batch_idx}_CT"
            if not os.path.exists(save_folder_MR):
                os.makedirs(save_folder_MR)
            if not os.path.exists(save_folder_CT):
                os.makedirs(save_folder_CT)
            # save ViT_heads
            # np.save(save_folder_MR+"/ViT_heads.npy", ViT_heads_MR)
            # np.save(save_folder_CT+"/ViT_heads.npy", ViT_heads_CT)
            # print length
            print(f"ViT_heads_MR: {len(ViT_heads_MR)}, ViT_heads_CT: {len(ViT_heads_CT)}")
            viz_ViT_heads_zneck_z12_z9_z6_z3_d12_d9_d6_d3(ViT_heads_MR, save_folder_MR)
            viz_ViT_heads_zneck_z12_z9_z6_z3_d12_d9_d6_d3(ViT_heads_CT, save_folder_CT)

            batch_idx += 1