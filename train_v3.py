# here we load the pretrained model and finetune it

import time
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

        # training
        model.train()
        batch_idx = 0
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
                # data_list = [data.cpu().numpy() for data in [data1, data2, data3, data4]]
                # stacked_data = np.concatenate(data_list, axis=0)
                mr = [data_hdf5[train_batch_list[idx_train_batch[str(idx_slice)]]]["mr"][()] for idx_slice in range(len(train_batch_list[idx_train_batch]))]
                mr = np.concatenate(mr, axis=0)
                ct = [data_hdf5[train_batch_list[idx_train_batch[str(idx_slice)]]]["ct"][()] for idx_slice in range(len(train_batch_list[idx_train_batch]))]
                ct = np.concatenate(ct, axis=0)
                mr_emb_head_3 = [data_hdf5[train_batch_list[idx_train_batch[str(idx_slice)]]]["mr_emb_head_3"][()] for idx_slice in range(len(train_batch_list[idx_train_batch]))]
                mr_emb_head_3 = np.concatenate(mr_emb_head_3, axis=0)
                mr_emb_head_6 = [data_hdf5[train_batch_list[idx_train_batch[str(idx_slice)]]]["mr_emb_head_6"][()] for idx_slice in range(len(train_batch_list[idx_train_batch]))]
                mr_emb_head_6 = np.concatenate(mr_emb_head_6, axis=0)
                mr_emb_head_9 = [data_hdf5[train_batch_list[idx_train_batch[str(idx_slice)]]]["mr_emb_head_9"][()] for idx_slice in range(len(train_batch_list[idx_train_batch]))]
                mr_emb_head_9 = np.concatenate(mr_emb_head_9, axis=0)
                mr_emb_head_12 = [data_hdf5[train_batch_list[idx_train_batch[str(idx_slice)]]]["mr_emb_head_12"][()] for idx_slice in range(len(train_batch_list[idx_train_batch]))]
                mr_emb_head_12 = np.concatenate(mr_emb_head_12, axis=0)
                mr_emb_head_neck = [data_hdf5[train_batch_list[idx_train_batch[str(idx_slice)]]]["mr_emb_head_neck"][()] for idx_slice in range(len(train_batch_list[idx_train_batch]))]
                mr_emb_head_neck = np.concatenate(mr_emb_head_neck, axis=0)
                




            for idx_slice in range(n_slice):
                key_name = slice_list[idx_slice]
                # grp.create_dataset("mr_emb_head_3", data=MedSAM_embedding[key]["mr_emb"]["head_3"], compression="gzip", compression_opts=4)
                # grp.create_dataset("mr_emb_head_6", data=MedSAM_embedding[key]["mr_emb"]["head_6"], compression="gzip", compression_opts=4)
                # grp.create_dataset("mr_emb_head_9", data=MedSAM_embedding[key]["mr_emb"]["head_9"], compression="gzip", compression_opts=4)
                # grp.create_dataset("mr_emb_head_12", data=MedSAM_embedding[key]["mr_emb"]["head_12"], compression="gzip", compression_opts=4)
                # grp.create_dataset("mr_emb_head_neck", data=MedSAM_embedding[key]["mr_emb"]["head_neck"], compression="gzip", compression_opts=4)
                # grp.create_dataset("mr", data=MedSAM_embedding[key]["mr"], compression="gzip", compression_opts=4)
                # grp.create_dataset("ct", data=MedSAM_embedding[key]["ct"], compression="gzip", compression_opts=4)
                mr = torch.tensor(data_hdf5[key_name]["mr"][()]).unsqueeze(0).to(device)
                ct = torch.tensor(data_hdf5[key_name]["ct"][()]).unsqueeze(0).to(device)
                mr_emb_head_3 = torch.tensor(data_hdf5[key_name]["mr_emb_head_3"][()]).unsqueeze(0).to(device)
                mr_emb_head_6 = torch.tensor(data_hdf5[key_name]["mr_emb_head_6"][()]).unsqueeze(0).to(device)
                mr_emb_head_9 = torch.tensor(data_hdf5[key_name]["mr_emb_head_9"][()]).unsqueeze(0).to(device)
                mr_emb_head_12 = torch.tensor(data_hdf5[key_name]["mr_emb_head_12"][()]).unsqueeze(0).to(device)
                mr_emb_head_neck = torch.tensor(data_hdf5[key_name]["mr_emb_head_neck"][()]).unsqueeze(0).to(device)
            
                optimizer.zero_grad()
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