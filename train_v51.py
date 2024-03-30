# here we load the pretrained model and finetune it

import time
import argparse

if __name__ == "__main__":

    # ------------------- load the configuration file path -------------------
    # run the parser to get the cfg path
    parser = argparse.ArgumentParser(description="Load configuration file path.")
    # Add an argument for the configuration file path with a default value
    parser.add_argument("--cfg_path", type=str, default="t1w_ViTAutoEnc.json", help="Path to the configuration file.")
    
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
    # print(cfg)
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("################### device:", device, "###################")
    if "cross_validation" not in cfg:
        root_dir = cfg["root_dir"]
    else:
        # generate an random 3-digit number according to current time as the cross_validation
        random_id = int(time.time() % 10000)
        root_dir = cfg["root_dir"] + f"_cv{random_id}/"
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
    from model import MONAI_ViTAutoEnc
    from model import MONAI_UNETR
    from model import UNet_MONAI

    from dataset import slice_npy

    # ------------------- random seeds -------------------
    # load the random seed
    if "cross_validation" not in cfg:
        random_seed = cfg["random_seed"]
    else:
        random_seed = random_id
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    import random
    import numpy as np
    np.random.seed(random_seed)
    random.seed(random_seed)

    # ------------------- verbose -------------------
    # model_verbose
    if "model_verbose" in cfg:
        model_verbose = True if cfg["model_verbose"] == "True" else False
    # training_verbose
    if "training_verbose" in cfg:
        training_verbose = True if cfg["training_verbose"] == "True" else False
    # show the configuration
    print("################### Configuration ###################")
    for key, value in cfg.items():
        print(f"{key}: {value}")
    print("################### Verbose ###################")
    print(f"model_verbose: {model_verbose}")
    print(f"training_verbose: {training_verbose}")

    # ------------------- create the model -------------------
    if "last_channel_num" in cfg:
        last_channel_num = cfg["last_channel_num"]
    else:
        last_channel_num = 32

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
            verbose=model_verbose,
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
            verbose=model_verbose,
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
            verbose=model_verbose,
            last_channel_num=last_channel_num,
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
    elif cfg["model_name"] == "MONAI_ViTAutoEnc":
        model = MONAI_ViTAutoEnc(
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
            verbose=model_verbose,
        )
    elif cfg["model_name"] == "MONAI_UNETR":
        model = MONAI_UNETR(
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
            verbose=model_verbose,
            last_channel_num=last_channel_num,
        )
    else:
        raise ValueError("model_name not found !")

    if cfg["model_name"] in ["decoder_PyramidPooling_encoder_MedSAM", "decoder_UNETR_encoder_MedSAM", "decoder_Deconv_encoder_MedSAM", "MONAI_ViTAutoEnc", "MONAI_UNETR"]:
        model.load_pretrain(cfg["pretrain_path"])
    model.to(device)

    # ------------------- load the dataset -------------------
    # load all cases and select the first part for tarining and the second part for validation

    data_folder = cfg["data_folder"]
    import glob
    case_list = sorted(glob.glob(data_folder+"/*mr*"))
    required_keys = cfg["required_keys"]
    file_dict_list = []
    for idx, case in enumerate(case_list):
        file_dict = {}
        for key in required_keys:
            file_dict[key] = case.replace("mr", key)
        file_dict_list.append(file_dict)
    # split the case_list into training and validation
    n_case = len(case_list)
    train_case = cfg["train_case"]
    val_case = cfg["val_case"]
    test_case = cfg["test_case"]
    training_list = file_dict_list[:train_case]
    validation_list = file_dict_list[train_case:train_case+val_case]
    testing_list = file_dict_list[train_case+val_case:]

    # save the training and validation list into the root_dir as a txt file
    with open(root_dir+"training_list.txt", "w") as f:
        for item in training_list:
            f.write("%s\n" % item)
    with open(root_dir+"validation_list.txt", "w") as f:
        for item in validation_list:
            f.write("%s\n" % item)
    with open(root_dir+"testing_list.txt", "w") as f:
        for item in testing_list:
            f.write("%s\n" % item)
    # output the training and validation list
    for idx, item in enumerate(training_list):
        print(f"Training {idx+1}: {item}")
    for idx, item in enumerate(validation_list):
        print(f"Validation {idx+1}: {item}")
    for idx, item in enumerate(testing_list):
        print(f"Testing {idx+1}: {item}")
    # output dataset division into a txt file
    with open(root_dir+"dataset_division.txt", "w") as f:
        f.write(f"Training: {train_case}\n")
        f.write(f"Validation: {val_case}\n")
        f.write(f"Testing: {test_case}\n")

    training_transform = True if cfg["training_transform"] == "True" else False

    # create the dataset and dataloader
    training_dataset = slice_npy(training_list, required_keys=cfg["required_keys"], is_channel_last=True, 
                                 transform=training_transform, output_size=1024)
    validation_dataset = slice_npy(validation_list, required_keys=cfg["required_keys"], is_channel_last=True)
    testing_dataset = slice_npy(testing_list, required_keys=cfg["required_keys"], is_channel_last=True)
    from torch.utils.data import DataLoader
    training_dataloader = DataLoader(training_dataset, batch_size=cfg["batch_size"], shuffle=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=cfg["batch_size"], shuffle=True)
    testing_dataloader = DataLoader(testing_dataset, batch_size=cfg["batch_size"], shuffle=True)
    
    # ------------------- training setting -------------------
    # create the optimizer
    if "optimizer" in cfg:
        if cfg["optimizer"] == "Adam":
            optimizer = optim.Adam(model.parameters(), lr=cfg["lr"])
        elif cfg["optimizer"] == "AdamW":
            optimizer = optim.AdamW(model.parameters(), lr=cfg["lr"])
        elif cfg["optimizer"] == "SGD":
            optimizer = optim.SGD(model.parameters(), lr=cfg["lr"])
        else:
            raise ValueError("optimizer not found !")
    else:
        optimizer = optim.Adam(model.parameters(), lr=cfg["lr"])

    # create the loss function using MAE loss
    loss_function = nn.L1Loss()

    # train the model
    best_val_loss = 1e10
    n_train = len(training_list)
    n_val = len(validation_list)
    batch_size = cfg["batch_size"]
    if "display_loss_per_batch" in cfg:
        display_step = cfg["display_loss_per_batch"]
    else:
        display_step = 1000
    # output the parameters
    n_epoch = cfg["epochs"]
    print(f"n_train: {n_train}, n_val: {n_val}), batch_size: {batch_size}")

    # ------------------- start training -------------------
    # train the model
    for epoch in range(n_epoch):

        epoch_loss = []
        # training
        model.train()
        # use training_dataloader to load the data
        display_loss = 0.0

        for idx_batch, data in enumerate(training_dataloader):
            # data is a dict with keys in cfg["required_keys"], and the values are tensors
            # [4, 1, 3, 1024, 1024], so squeeze the second dimension
            # load the data
            mr = data["mr"].float().to(device)
            ct = data["ct"].float().to(device)
            print(mr.shape, ct.shape)

            optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                pred = model(mr)
                loss = loss_function(pred, ct)
                loss.backward()
                # save the gradient into training verbose
                # # grad clipping
                # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                text_loss = loss.item()
                display_loss += text_loss

                print(f"Epoch {epoch+1}/{n_epoch}, batch {idx_batch+1}/{len(training_dataloader)}, loss: {text_loss}")
                if training_verbose:
                    # write the loss into a txt file
                    with open(root_dir+"training_verbose.txt", "a") as f:
                        f.write(f"Epoch {epoch+1}/{cfg['epochs']}, batch {idx_batch+1}/{len(training_dataloader)}, loss: {text_loss}\n")
                    # print(f"Epoch {epoch+1}/{cfg['epochs']}, batch {idx_batch+1}/{len(training_dataloader)}, loss: {text_loss}")
                    # pause the program
                    # input("Press Enter to continue...")
                        
            # display the loss
            if (idx_batch+1) % display_step == 0:
                print(f"Epoch {epoch+1}/{n_epoch}, batch {idx_batch+1}/{len(training_dataloader)}, loss: {display_loss/display_step}")
                display_loss = 0.0
                with open(root_dir+"training_verbose.txt", "a") as f:
                    f.write(f"Epoch {epoch+1}/{n_epoch}, batch {idx_batch+1}/{len(training_dataloader)}, loss: {display_loss/display_step}\n")
            

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
                
                mr = data["mr"].float().to(device)
                ct = data["ct"].float().to(device)

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
        
        # testing
        if (epoch+1) % cfg["test_step"] == 0:
            model.eval()
            test_loss = []
            for idx_batch, data in enumerate(testing_dataloader):
                mr = data["mr"].float().to(device)
                ct = data["ct"].float().to(device)

                with torch.set_grad_enabled(False):
                    pred = model(mr)
                    loss = loss_function(pred, ct)
                    test_loss.append(loss.item())
            
            test_loss = np.mean(np.asarray(test_loss))
            print(f"Epoch {epoch+1}/{cfg['epochs']}, test_loss: {test_loss}")
            time_stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            with open(root_dir+"test_loss.txt", "a") as f:
                f.write(f"%{time_stamp}% -> Epoch {epoch+1}/{cfg['epochs']}, test_loss: {test_loss}\n")