# here we load the pretrained model and finetune it


import argparse

if __name__ == "__main__":



    # run the parser to get the cfg path
    parser = argparse.ArgumentParser(description="Load configuration file path.")
    # Add an argument for the configuration file path with a default value
    parser.add_argument("--cfg_path", type=str, default="PyramidPooling.json", help="Path to the configuration file.")
    
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
    # torch.autograd.set_detect_anomaly(True)

    # load libraries
    import torch.nn as nn
    import torch.optim as optim
    from torchvision import transforms
    from torch.utils.data import DataLoader
    from matplotlib import pyplot as plt

    from model import decoder_UNETR_encoder_MedSAM
    from model import decoder_Deconv_encoder_MedSAM
    from model import decoder_PyramidPooling_encoder_MedSAM
    from dataset import PairedMRCTDataset_train

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
            # verbose=True,
        )
    elif cfg["model_name"] == "decoder_UNETR_encoder_MedSAM":
        model = decoder_UNETR_encoder_MedSAM(
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
        )
    elif cfg["model_name"] == "decoder_Deconv_encoder_MedSAM":
        model = decoder_Deconv_encoder_MedSAM(
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
            BN=True if cfg["batch_size"] >= 8 else False,
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
    )
    dataset_val = PairedMRCTDataset_train(
        path_MR=cfg["data_path_MR"],
        path_CT=cfg["data_path_CT"],
        stage="val", 
        subset_fraction=cfg["subset_fraction"],
        #  transform=val_transform,
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

    # create the optimizer
    optimizer = optim.Adam(model.parameters(), lr=cfg["lr"])

    # create the loss function using MAE loss
    loss_function = nn.L1Loss()

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
            optimizer.zero_grad()
            epoch_loss = 0
            with torch.set_grad_enabled(True):
                pred = model(MR)
                loss = loss_function(pred, CT)
                loss.backward()
                optimizer.step()
                text_loss = loss.item()
                epoch_loss += text_loss
                # print the loss every print_step, with current batch over the whole batch
                if (batch_idx+1) % cfg["print_batch_step"] == 0:
                    print(f"Epoch {epoch+1}/{cfg['epochs']} Batch {batch_idx+1}/{n_train_batch}, loss: {text_loss}")
            batch_idx += 1

            epoch_loss /= len(train_loader)

        # plot images
        if (epoch+1) % cfg["plot_step"] == 0:
            fig, ax = plt.subplots(1, 3, figsize=(15, 5))
            ax[0].imshow(MR[0, 1, :, :].cpu().detach().numpy(), cmap="gray")
            ax[0].set_title("MR")
            ax[1].imshow(CT[0, 0, :, :].cpu().detach().numpy(), cmap="gray")
            ax[1].set_title("CT")
            ax[2].imshow(pred[0, 0, :, :].cpu().detach().numpy(), cmap="gray")
            ax[2].set_title("pred")
            plt.savefig(root_dir+f"epoch_{epoch+1}.png")
            plt.close()

        # print the loss
        print(f"Epoch {epoch+1}/{cfg['epochs']}, loss: {epoch_loss}")
        # save the loss
        with open(root_dir+"loss.txt", "a") as f:
            f.write(f"Epoch {epoch+1}/{cfg['epochs']}, loss: {epoch_loss}\n")
        # save the model and optimizer
        if (epoch+1) % cfg["save_step"] == 0:
            torch.save(model.state_dict(), root_dir+f"model_{epoch+1}.pth")
            torch.save(optimizer.state_dict(), root_dir+f"optimizer_{epoch+1}.pth")
        
        # validation
        if (epoch+1) % cfg["eval_step"] == 0:
            model.eval()
            with torch.no_grad():
                val_loss = 0
                epoch_val_loss = 0
                for idx_batch_val, batch in enumerate(val_loader):
                    MR = batch["MR"]
                    CT = batch["CT"]
                    MR = MR.to(device)
                    CT = CT.to(device)
                    pred = model(MR)
                    text_val_loss = loss_function(pred, CT).item()
                    epoch_val_loss += text_val_loss
                    if (batch_idx+1) % cfg["print_batch_step"] == 0:
                        print(f"Epoch {epoch+1}/{cfg['epochs']} Batch {idx_batch_val+1}/{n_val_batch}, loss: {text_loss}")
                val_loss = epoch_val_loss / len(val_loader)

            print(f"Epoch {epoch+1}/{cfg['epochs']}, loss: {loss.item()}, val_loss: {val_loss}")
            with open(root_dir+"val_loss.txt", "a") as f:
                f.write(f"Epoch {epoch+1}/{cfg['epochs']}, val_loss: {val_loss}\n")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), root_dir+f"best_model.pth")
                torch.save(optimizer.state_dict(), root_dir+f"best_optimizer.pth")
                print("Model was saved !")
            else:
                print("Model was not saved !")