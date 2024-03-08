# here we load the pretrained model and finetune it

import time
import argparse

if __name__ == "__main__":

    # run the parser to get the cfg path
    parser = argparse.ArgumentParser(description="Load configuration file path.")
    # Add an argument for the configuration file path with a default value
    parser.add_argument("--cfg_path", type=str, default="prembed_UNETR_pct5_SynthRad_Brain.json", help="Path to the configuration file.")
    
    # Parse the command-line arguments
    args = parser.parse_args()

    # Use args.cfg_path to access the configuration file path
    print("Configuration path:", args.cfg_path)

    # set GPU
    import os
    import json
    import h5py
    import random
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
    from matplotlib import pyplot as plt

    from model import decoder_UNETR

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
    if cfg["model_name"] == "decoder_UNETR":
        model = decoder_UNETR(
            img_size = cfg["img_size"],
            out_chans = cfg["out_chans"],
            verbose = True if cfg["verbose"] == "True" else False,
        )
    else:
        raise ValueError("model_name not found !")

    model.to(device)

    # load all cases and select the first part for tarining and the second part for validation
    training_case = cfg["training_case"]
    validation_case = cfg["validation_case"]
    data_folder = cfg["data_folder"]
    import glob
    case_list = sorted(glob.glob(data_folder+"/*"))
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

    # train the model
    for epoch in range(cfg["epochs"]):

        epoch_loss = []
        # training
        model.train()
        for idx_batch, case in enumerate(training_list):
            hdf5_filename = case + "/MedSAM_embedding_gzip4.hdf5"
            data_hdf5 = h5py.File(hdf5_filename, "r")
            n_slice = len(data_hdf5.keys())
            slice_list = list(data_hdf5.keys())

            # shuffle the slice_list
            random.shuffle(slice_list)

            # divide the slice_list into batches
            # create the list like [batch1, batch2, ..., batch_n]
            # batch_n may be less than batch_size
            n_train_batch = n_slice // batch_size
            if n_slice % batch_size != 0:
                n_train_batch += 1
            train_batch_list = []
            for idx in range(n_train_batch):
                if idx == n_train_batch - 1:
                    train_batch_list.append(slice_list[idx*batch_size:])
                else:
                    train_batch_list.append(slice_list[idx*batch_size:(idx+1)*batch_size])
            
            

            # load the data
            for idx_train_batch in range(n_train_batch):
                mr = torch.from_numpy(np.concatenate([data_hdf5[slice_name]["mr"][()] for slice_name in train_batch_list[idx_train_batch]], axis=0)).float().to(device)
                ct = torch.from_numpy(np.concatenate([data_hdf5[slice_name]["ct"][()] for slice_name in train_batch_list[idx_train_batch]], axis=0)).float().to(device)
                mr_emb_head_3 = torch.from_numpy(np.concatenate([data_hdf5[slice_name]["mr_emb_head_3"][()] for slice_name in train_batch_list[idx_train_batch]], axis=0)).float().to(device)
                mr_emb_head_6 = torch.from_numpy(np.concatenate([data_hdf5[slice_name]["mr_emb_head_6"][()] for slice_name in train_batch_list[idx_train_batch]], axis=0)).float().to(device)
                mr_emb_head_9 = torch.from_numpy(np.concatenate([data_hdf5[slice_name]["mr_emb_head_9"][()] for slice_name in train_batch_list[idx_train_batch]], axis=0)).float().to(device)
                mr_emb_head_12 = torch.from_numpy(np.concatenate([data_hdf5[slice_name]["mr_emb_head_12"][()] for slice_name in train_batch_list[idx_train_batch]], axis=0)).float().to(device)
                mr_emb_head_neck = torch.from_numpy(np.concatenate([data_hdf5[slice_name]["mr_emb_head_neck"][()] for slice_name in train_batch_list[idx_train_batch]], axis=0)).float().to(device)
                # print(mr.size(), ct.size(), mr_emb_head_3.size(), mr_emb_head_6.size(), mr_emb_head_9.size(), mr_emb_head_12.size(), mr_emb_head_neck.size())

                optimizer.zero_grad()
                with torch.set_grad_enabled(True):
                    pred = model(mr, mr_emb_head_3, mr_emb_head_6, mr_emb_head_9, mr_emb_head_12, mr_emb_head_neck)
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
            for idx_batch, case in enumerate(validation_list):
                hdf5_filename = case + "/MedSAM_embedding_gzip4.hdf5"
                data_hdf5 = h5py.File(hdf5_filename, "r")
                n_slice = len(data_hdf5.keys())
                slice_list = list(data_hdf5.keys())

                # shuffle the slice_list
                random.shuffle(slice_list)

                # divide the slice_list into batches
                # create the list like [batch1, batch2, ..., batch_n]
                # batch_n may be less than batch_size
                n_val_batch = n_slice // batch_size
                if n_slice % batch_size != 0:
                    n_val_batch += 1
                val_batch_list = []
                for idx in range(n_val_batch):
                    if idx == n_val_batch - 1:
                        val_batch_list.append(slice_list[idx*batch_size:])
                    else:
                        val_batch_list.append(slice_list[idx*batch_size:(idx+1)*batch_size])
                
                # load the data
                for idx_val_batch in range(n_val_batch):
                    mr = torch.from_numpy(np.concatenate([data_hdf5[slice_name]["mr"][()] for slice_name in val_batch_list[idx_val_batch]], axis=0)).float().to(device)
                    ct = torch.from_numpy(np.concatenate([data_hdf5[slice_name]["ct"][()] for slice_name in val_batch_list[idx_val_batch]], axis=0)).float().to(device)
                    mr_emb_head_3 = torch.from_numpy(np.concatenate([data_hdf5[slice_name]["mr_emb_head_3"][()] for slice_name in val_batch_list[idx_val_batch]], axis=0)).float().to(device)
                    mr_emb_head_6 = torch.from_numpy(np.concatenate([data_hdf5[slice_name]["mr_emb_head_6"][()] for slice_name in val_batch_list[idx_val_batch]], axis=0)).float().to(device)
                    mr_emb_head_9 = torch.from_numpy(np.concatenate([data_hdf5[slice_name]["mr_emb_head_9"][()] for slice_name in val_batch_list[idx_val_batch]], axis=0)).float().to(device)
                    mr_emb_head_12 = torch.from_numpy(np.concatenate([data_hdf5[slice_name]["mr_emb_head_12"][()] for slice_name in val_batch_list[idx_val_batch]], axis=0)).float().to(device)
                    mr_emb_head_neck = torch.from_numpy(np.concatenate([data_hdf5[slice_name]["mr_emb_head_neck"][()] for slice_name in val_batch_list[idx_val_batch]], axis=0)).float().to(device)

                    with torch.set_grad_enabled(False):
                        pred = model(mr, mr_emb_head_3, mr_emb_head_6, mr_emb_head_9, mr_emb_head_12, mr_emb_head_neck)
                        loss = loss_function(pred, ct)
                        val_loss.append(loss.item())

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