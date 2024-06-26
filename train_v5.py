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

    from dataset import slice_hdf5_dataset
    from util import acquire_data_from_control

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

    model.load_pretrain(cfg["pretrain_path"])
    model.to(device)

    # ------------------- load the dataset -------------------
    # load all cases and select the first part for tarining and the second part for validation

    data_folder = cfg["data_folder"]
    import glob
    case_list = sorted(glob.glob(data_folder+"/*"))
    data_folder_name = cfg["data_folder"].split("/")[-1]
    experiment_name = root_dir.split("/")[-2]
    if "shuffle_case" in cfg:
        if cfg["shuffle_case"] == "True":
            random.shuffle(case_list)
    if "training_case" in cfg and "validation_case" in cfg and "testing_case" in cfg:
        training_num = cfg["training_case"]
        validation_num = cfg["validation_case"]
        testing_num = cfg["testing_case"]
    elif "training_ratio" in cfg and "validation_ratio" in cfg and "testing_ratio" in cfg:
        training_ratio = cfg["training_ratio"]
        validation_ratio = cfg["validation_ratio"]
        testing_ratio = cfg["testing_ratio"]
        training_num = int(len(case_list)*training_ratio)
        validation_num = int(len(case_list)*validation_ratio)
        testing_num = int(len(case_list)*testing_ratio)
    else:
        raise ValueError("training_case and validation_case or training_ratio and validation_ratio not found !")

    training_list = acquire_data_from_control(
        data_folder_name = data_folder_name,
        required_case_numbers = training_num,
        experiment_name = experiment_name,
    )

    validation_list = acquire_data_from_control(
        data_folder_name = data_folder_name,
        required_case_numbers = validation_num,
        experiment_name = experiment_name,
    )

    testing_list = acquire_data_from_control(
        data_folder_name = data_folder_name,
        required_case_numbers = testing_num,
        experiment_name = experiment_name,
    )

    

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
    if "remove_head_tail" not in cfg:
        remove_head_tail = 1
    else:
        remove_head_tail = cfg["remove_head_tail"]
    for case in training_list:
        found_hdf5 = sorted(glob.glob(case+"/"+search_prefix+"*"+search_affix+".hdf5"))
        # remove the first and last one
        found_hdf5 = found_hdf5[remove_head_tail:-remove_head_tail]
        hdf5_training_list.extend(found_hdf5)
    for case in validation_list:
        found_hdf5 = sorted(glob.glob(case+"/"+search_prefix+"*"+search_affix+".hdf5"))
        # remove the first and last one
        found_hdf5 = found_hdf5[remove_head_tail:-remove_head_tail]
        hdf5_validation_list.extend(found_hdf5)
    # save the training and validation hdf5 list into the root_dir as a txt file
    with open(root_dir+"hdf5_training_list.txt", "w") as f:
        for item in hdf5_training_list:
            f.write("%s\n" % item)
    with open(root_dir+"hdf5_validation_list.txt", "w") as f:
        for item in hdf5_validation_list:
            f.write("%s\n" % item)
    # create the dataset and dataloader
    training_dataset = slice_hdf5_dataset(hdf5_training_list, required_keys=cfg["required_keys"], 
                                          training_verbose=training_verbose, training_verbose_file=root_dir+"training_verbose.txt")
    validation_dataset = slice_hdf5_dataset(hdf5_validation_list, required_keys=cfg["required_keys"],
                                            training_verbose=training_verbose, training_verbose_file=root_dir+"training_verbose.txt")
    testing_dataset = slice_hdf5_dataset(hdf5_validation_list, required_keys=cfg["required_keys"],
                                            training_verbose=training_verbose, training_verbose_file=root_dir+"training_verbose.txt")

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
            mr = data["mr"].float().to(device).squeeze(1)
            ct = data["ct"].float().to(device).squeeze(1)

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
                # if text_loss is nan, pause the program
                if np.isnan(text_loss):
                    print("text_loss is nan !")
                    print(f"idx_batch: {idx_batch}, mr.shape: {mr.shape}, ct.shape: {ct.shape}, pred.shape: {pred.shape}")
                    print(f"mr: {mr}")
                    print(f"ct: {ct}")
                    print(f"pred: {pred}")
                    input("Press Enter to continue...")
                epoch_loss.append(text_loss)
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

        # ------------------- testing -------------------
        if (epoch+1) % cfg["test_step"] == 0:
            model.eval()
            test_loss = []
            for idx_batch, data in enumerate(testing_dataloader):
                mr = data["mr"].float().to(device).squeeze(1)
                ct = data["ct"].float().to(device).squeeze(1)

                with torch.set_grad_enabled(False):
                    pred = model(mr)
                    loss = loss_function(pred, ct)
                    test_loss.append(loss.item())
            test_loss = np.mean(np.asarray(test_loss))
            print(f"Epoch {epoch+1}/{cfg['epochs']}, test_loss: {test_loss}")
            time_stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            with open(root_dir+"test_loss.txt", "a") as f:
                f.write(f"%{time_stamp}% -> Epoch {epoch+1}/{cfg['epochs']}, test_loss: {test_loss}\n")
            