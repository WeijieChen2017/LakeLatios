# here we load the pretrained model and finetune it

# set GPU
import os
import json
# load the cfg
cfg_path = "config_0219.json"
cfg = json.load(open(cfg_path))
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

from model import decoder_UNETR_encoder_MedSAM
from dataset import PairedMRCTDataset_train

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
model.load_pretrain(cfg["pretrain_path"])

# load the dataset
train_transform = transforms.Compose([
    transforms.Resize((1024, 1024)),
    transforms.ToTensor(),
])
val_transform = transforms.Compose([
    transforms.Resize((1024, 1024)),
    transforms.ToTensor(),
])
dataset_train = PairedMRCTDataset_train(
     path_MR=cfg["data_path_MR"],
     path_CT=cfg["data_path_CT"],
     stage="train", 
     transform=train_transform,
)
dataset_val = PairedMRCTDataset_train(
     path_MR=cfg["data_path_MR"],
     path_CT=cfg["data_path_CT"],
     stage="val", 
     transform=val_transform,
)
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
for epoch in range(cfg["epochs"]):

    # training
    model.train()
    for batch in train_loader:
        MR = batch["MR"]
        CT = batch["CT"][:, 1, :, :]
        MR = MR.to(device)
        CT = CT.to(device)
        optimizer.zero_grad()
        with torch.set_grad_enabled(True):
            pred = model(MR)
            loss = loss_function(pred, CT)
            loss.backward()
            optimizer.step()
        text_loss = loss.item()

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
    print(f"Epoch {epoch+1}/{cfg['epochs']}, loss: {text_loss}")
    # save the loss
    with open(root_dir+"loss.txt", "a") as f:
        f.write(f"Epoch {epoch+1}/{cfg['epochs']}, loss: {text_loss}\n")
    # save the model and optimizer
    if (epoch+1) % cfg["save_step"] == 0:
        torch.save(model.state_dict(), root_dir+f"model_{epoch+1}.pth")
        torch.save(optimizer.state_dict(), root_dir+f"optimizer_{epoch+1}.pth")
    
    # validation
    if (epoch+1) % cfg["eval_step"] == 0:
        model.eval()
        with torch.no_grad():
            val_loss = 0
            for batch in val_loader:
                MR = batch["MR"]
                CT = batch["CT"][:, 1, :, :]
                MR = MR.to(device)
                CT = CT.to(device)
                pred = model(MR)
                val_loss += loss_function(pred, CT).item()
            val_loss /= len(val_loader)
        print(f"Epoch {epoch+1}/{cfg['epochs']}, loss: {loss.item()}, val_loss: {val_loss}")
        with open(root_dir+"val_loss.txt", "a") as f:
            f.write(f"Epoch {epoch+1}/{cfg['epochs']}, val_loss: {val_loss}\n")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model.pth")
            torch.save(optimizer.state_dict(), "best_optimizer.pth")
            print("Model was saved !")
        else:
            print("Model was not saved !")
