# here we load the pretrained model and finetune it

import time
import argparse

if __name__ == "__main__":

    # ------------------- load the configuration file path -------------------
    # run the parser to get the cfg path
    parser = argparse.ArgumentParser(description="Load configuration file path.")
    # Add an argument for the configuration file path with a default value
    parser.add_argument("--cfg_path", type=str, default="case_UNet_pct1.json", help="Path to the configuration file.")
    
    # Parse the command-line arguments
    args = parser.parse_args()

    # Use args.cfg_path to access the configuration file path
    print("Configuration path:", args.cfg_path)

    # ------------------- set the GPU -------------------
    import os
    import json
    import random
    # load the cfg
    # cfg_path = "config_0221.json"
    cfg_path = "config/"+args.cfg_path
    cfg = json.load(open(cfg_path))
    print(cfg)
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("################### device:", device, "###################")
    if "cross_validation" not in cfg:
        root_dir = cfg["root_dir"]
    else:
        # generate an random 3-digit number according to current time as the cross_validation
        random_id = int(time.time() % 1000)
        root_dir = cfg["root_dir"] + f"cv{random_id}/"
    print("------------------- root_dir:", root_dir, "-------------------")
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)
    # torch.autograd.set_detect_anomaly(True)

    # load libraries
    import torch.nn as nn
    import torch.optim as optim
    from matplotlib import pyplot as plt

    from model import decoder_PyramidPooling_encoder_MedSAM
    from model import decoder_UNETR_encoder_MedSAM
    from model import decoder_Deconv_encoder_MedSAM
    from model import UNet_MONAI

    from dataset import slice_hdf5_dataset

    # ------------------- random seeds -------------------
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

    # ------------------- create the model -------------------
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
    elif cfg["model_name"] == "UNet_MONAI":
        model = UNet_MONAI(
            spatial_dims=cfg["spatial_dims"],
            in_channels=cfg["in_channels"],
            out_channels=cfg["out_channels"],
            channels=cfg["channels"],
            strides=cfg["strides"],
            kernel_size=cfg["kernel_size"],
            up_kernel_size=cfg["up_kernel_size"],
            num_res_units=cfg["num_res_units"],
            act=cfg["act"], 
            norm=cfg["norm"],
            dropout=cfg["dropout"],
            bias=cfg["bias"],
            adn_ordering=cfg["adn_ordering"],
        )
    else:
        raise ValueError("model_name not found !")

    model.to(device)

    # ------------------- load the dataset -------------------
    # load all cases and select the first part for tarining and the second part for validation
    training_case = cfg["training_case"]
    validation_case = cfg["validation_case"]
    data_folder = cfg["data_folder"]
    import glob
    case_list = sorted(glob.glob(data_folder+"/*"))
    if cfg["shuffle_case"] == "True":
        random.shuffle(case_list)
    training_list = case_list[:training_case]
    validation_list = case_list[training_case:training_case+validation_case]
    # save the training and validation list into the root_dir as a txt file
    with open(root_dir+"training_list.txt", "w") as f:
        for item in training_list:
            f.write("%s\n" % item)
    with open(root_dir+"validation_list.txt", "w") as f:
        for item in validation_list:
            f.write("%s\n" % item)
    # output the training and validation list
    for idx, item in enumerate(training_list):
        print(f"Training {idx+1}: {item}")
    for idx, item in enumerate(validation_list):
        print(f"Validation {idx+1}: {item}")
    # load all hdf5 in the training and validation list
    # in each folder, there are pack_000.hdf5, pack_001.hdf5, ...
    hdf5_training_list = []
    hdf5_validation_list = []
    # check whether there is a key in the cfg
    if "file_affix" not in cfg:
        search_affix = ""
    else:
        search_affix = cfg["file_affix"]
    if "file_prefix" not in cfg:
        search_prefix = ""
    else:
        search_prefix = cfg["file_prefix"]
    for case in training_list:
        hdf5_training_list.extend(sorted(glob.glob(case+"/"+search_prefix+"*"+search_affix+".hdf5")))
    for case in validation_list:
        hdf5_validation_list.extend(sorted(glob.glob(case+"/"+search_prefix+"*"+search_affix+".hdf5")))
    # save the training and validation hdf5 list into the root_dir as a txt file
    with open(root_dir+"hdf5_training_list.txt", "w") as f:
        for item in hdf5_training_list:
            f.write("%s\n" % item)
    with open(root_dir+"hdf5_validation_list.txt", "w") as f:
        for item in hdf5_validation_list:
            f.write("%s\n" % item)
    # create the dataset and dataloader
    training_dataset = slice_hdf5_dataset(hdf5_training_list, required_keys=cfg["required_keys"])
    validation_dataset = slice_hdf5_dataset(hdf5_validation_list, required_keys=cfg["required_keys"])

    from torch.utils.data import DataLoader
    training_dataloader = DataLoader(training_dataset, batch_size=cfg["batch_size"], shuffle=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=cfg["batch_size"], shuffle=True)
    
    # ------------------- training setting -------------------
    # create the optimizer
    optimizer = optim.Adam(model.parameters(), lr=cfg["lr"])

    # create the loss function using MAE loss
    loss_function = nn.L1Loss()

    # train the model
    best_val_loss = 1e10
    n_train = len(training_list)
    n_val = len(validation_list)
    batch_size = cfg["batch_size"]
    # output the parameters
    print(f"n_train: {n_train}, n_val: {n_val}), batch_size: {batch_size}")

    # ------------------- start training -------------------
    # train the model
    for epoch in range(cfg["epochs"]):

        epoch_loss = []
        # training
        model.train()
        # use training_dataloader to load the data
        for idx_batch, data in enumerate(training_dataloader):
            # data is a dict with keys in cfg["required_keys"], and the values are tensors
            # [4, 1, 3, 1024, 1024], so squeeze the second dimension
            # load the data
            mr = data["mr"].float().to(device).squeeze(1)
            ct = data["ct"].float().to(device).squeeze(1)

            optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                pred = model(mr)
                loss = loss_function(pred, ct)
                loss.backward()
                optimizer.step()
                text_loss = loss.item()
                epoch_loss.append(text_loss)

        # plot images
        if (epoch+1) % cfg["plot_step"] == 0:
            fig, ax = plt.subplots(1, 3, figsize=(15, 5))
            ax[0].imshow(mr[0, 1, :, :].cpu().detach().numpy(), cmap="gray")
            ax[0].set_title("MR")
            ax[1].imshow(ct[0, 0, :, :].cpu().detach().numpy(), cmap="gray")
            ax[1].set_title("CT")
            ax[2].imshow(pred[0, 0, :, :].cpu().detach().numpy(), cmap="gray")
            ax[2].set_title("pred")
            plt.savefig(root_dir+f"epoch_{epoch+1}.png")
            plt.close()

        epoch_loss = np.mean(np.asarray(epoch_loss))
        # print the loss
        print(f"Epoch {epoch+1}/{cfg['epochs']}, loss: {epoch_loss}")
        # save the loss
        time_stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        with open(root_dir+"loss.txt", "a") as f:
            f.write(f"%{time_stamp}% -> Epoch {epoch+1}/{cfg['epochs']}, loss: {epoch_loss}\n")
        # save the model and optimizer
        if (epoch+1) % cfg["save_step"] == 0:
            torch.save(model.state_dict(), root_dir+f"model_{epoch+1}.pth")
            torch.save(optimizer.state_dict(), root_dir+f"optimizer_{epoch+1}.pth")
        
        # validation
        if (epoch+1) % cfg["eval_step"] == 0:
            model.eval()
            val_loss = []
            
            for idx_batch, data in enumerate(validation_dataloader):
                
                mr = data["mr"].float().to(device).squeeze(1)
                ct = data["ct"].float().to(device).squeeze(1)

                with torch.set_grad_enabled(False):
                    pred = model(mr)
                    loss = loss_function(pred, ct)
                    val_loss.append(loss.item())

            val_loss = np.mean(np.asarray(val_loss))
            print(f"Epoch {epoch+1}/{cfg['epochs']}, val_loss: {val_loss}")
            time_stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            with open(root_dir+"val_loss.txt", "a") as f:
                f.write(f"%{time_stamp}% -> Epoch {epoch+1}/{cfg['epochs']}, val_loss: {val_loss}\n")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), root_dir+f"best_model.pth")
                torch.save(optimizer.state_dict(), root_dir+f"best_optimizer.pth")
                print("Model was saved !")
            else:
                print("Model was not saved !")