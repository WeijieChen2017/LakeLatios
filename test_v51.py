import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
import os
import json
import argparse
import numpy as np
from dataset import slice_npy

from model import decoder_PyramidPooling_encoder_MedSAM
from model import decoder_UNETR_encoder_MedSAM
from model import decoder_Deconv_encoder_MedSAM
from model import UNet_MONAI

def load_model(cfg, device):
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
            qkv_bias=cfg["qkv_bias"] == "True",
            norm_layer=nn.LayerNorm if cfg["norm_layer"] == "nn.LayerNorm" else None,
            act_layer=nn.GELU if cfg["act_layer"] == "nn.GELU" else None,
            use_abs_pos=cfg["use_abs_pos"] == "True",
            use_rel_pos=cfg["use_rel_pos"] != "False",
            rel_pos_zero_init=cfg["rel_pos_zero_init"] == "True",
            window_size=cfg["window_size"],
            global_attn_indexes=cfg["global_attn_indexes"],
            # verbose=True,
        )
    # Add the other model options here as in your training script
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
            qkv_bias=cfg["qkv_bias"] == "True",
            norm_layer=nn.LayerNorm if cfg["norm_layer"] == "nn.LayerNorm" else None,
            act_layer=nn.GELU if cfg["act_layer"] == "nn.GELU" else None,
            use_abs_pos=cfg["use_abs_pos"] == "True",
            use_rel_pos=cfg["use_rel_pos"] != "False",
            rel_pos_zero_init=cfg["rel_pos_zero_init"] == "True",
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
            qkv_bias=cfg["qkv_bias"] == "True",
            norm_layer=nn.LayerNorm if cfg["norm_layer"] == "nn.LayerNorm" else None,
            act_layer=nn.GELU if cfg["act_layer"] == "nn.GELU" else None,
            use_abs_pos=cfg["use_abs_pos"] == "True",
            use_rel_pos=cfg["use_rel_pos"] != "False",
            rel_pos_zero_init=cfg["rel_pos_zero_init"] == "True",
            window_size=cfg["window_size"],
            global_attn_indexes=cfg["global_attn_indexes"],
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
    
    # Load the encoder
    # model.load_pretrain(cfg["pretrain_path"])
    # Load the decoder
    # model.load_pretrain(cfg["decoder_path"], remove_prefix="") # this did not load all the weights
    model.load_from_checkpoint(cfg["decoder_path"])

    model.to(device)
    return model

def test_model(model, test_loader, device, cfg):
    model.eval()
    loss_function = nn.L1Loss()
    total_loss = 0.0
    # output the evaluation info
    n_test = len(test_loader)
    print(f"Start testing on {n_test} samples.")
    with torch.no_grad():
        for data, filename_mr in test_loader:
            filename = filename_mr[0]+"_pred"
            print(f"Processing {filename}", end="")

            MR = data["mr"].to(device)
            CT = data["ct"].to(device)
            pred = model(MR)
            loss = loss_function(pred, CT)

            # change the channel order
            MR = MR.permute(0, 2, 3, 1)
            CT = CT.permute(0, 2, 3, 1)
            pred = pred.permute(0, 2, 3, 1)

            MR = MR.detach().cpu().numpy()
            CT = CT.detach().cpu().numpy()
            pred = pred.detach().cpu().numpy()
            # mr shape (1, 512, 512, 3) ct shape (1, 512, 512, 1) pred shape (1, 512, 512, 1)
            # print("mr shape", MR.shape, "ct shape", CT.shape, "pred shape", pred.shape)
            # if modality == "MR":
            #     img_data = np.clip(img_data, 0, 3000)
            #     img_data = img_data / 3000
            # elif modality == "CT":
            #     img_data = np.clip(img_data, -1024, 3000)
            #     img_data = img_data / 4024
            loss = loss.item() * 4024
            print(f" ---> Loss: {loss}")
            total_loss += loss
            # iterate all the filenames in the batch

            idx_batch = 0

            save_name = os.path.join(cfg["root_dir"], filename)
            save_data = np.squeeze(pred[idx_batch, :, :, :])# convert back to HU
            np.save(save_name, save_data)
            # print(f"Saved prediction for {filename[idx_batch]} with loss {loss}")

            # plot input, ground truth and prediction
            plt.figure(figsize=(15, 5), dpi=100)
            plt.subplot(1, 3, 1)
            img_MR = np.squeeze(MR[idx_batch, :, :, 1])
            plt.imshow(img_MR, cmap="gray")
            plt.title("MR")
            plt.axis("off")

            plt.subplot(1, 3, 2)
            img_CT = np.squeeze(CT[idx_batch, :, :, :])
            plt.imshow(img_CT, cmap="gray")
            plt.title("CT")
            plt.axis("off")

            plt.subplot(1, 3, 3)
            img_pred = np.squeeze(pred[idx_batch, :, :, :])
            plt.imshow(img_pred, cmap="gray")
            plt.title("Prediction")
            plt.axis("off")

            plt.savefig(os.path.join(cfg["root_dir"], filename+ ".png"))
            plt.close()

            # write the loss to a file
            with open(os.path.join(cfg["root_dir"], "_test_loss.txt"), "a") as f:
                f.write(f"{filename}, {loss}\n")
    avg_loss = total_loss / len(test_loader)
    print(f"Test Loss: {avg_loss}")
    # write the average loss to a file
    with open(os.path.join(cfg["root_dir"], "_test_loss.txt"), "a") as f:
        f.write(f"Average Loss: {avg_loss}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load configuration for testing.")
    parser.add_argument("--cfg_path", type=str, default="config/t1w_UNet.json", help="Path to the configuration file.")
    args = parser.parse_args()
    cfg = json.load(open(args.cfg_path))
    
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

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    model = load_model(cfg, device)
    
    file_list_dict = [
        {'mr': 'data/t1w/6_mr_0.npy', 'ct': 'data/t1w/6_ct_0.npy'},
        {'mr': 'data/t1w/6_mr_1.npy', 'ct': 'data/t1w/6_ct_1.npy'},
        {'mr': 'data/t1w/6_mr_10.npy', 'ct': 'data/t1w/6_ct_10.npy'},
        # {'mr': 'data/t1w/6_mr_11.npy', 'ct': 'data/t1w/6_ct_11.npy'},
        # {'mr': 'data/t1w/6_mr_12.npy', 'ct': 'data/t1w/6_ct_12.npy'},
        # {'mr': 'data/t1w/6_mr_13.npy', 'ct': 'data/t1w/6_ct_13.npy'},
        # {'mr': 'data/t1w/6_mr_14.npy', 'ct': 'data/t1w/6_ct_14.npy'},
        {'mr': 'data/t1w/6_mr_2.npy', 'ct': 'data/t1w/6_ct_2.npy'},
        {'mr': 'data/t1w/6_mr_3.npy', 'ct': 'data/t1w/6_ct_3.npy'},
        {'mr': 'data/t1w/6_mr_4.npy', 'ct': 'data/t1w/6_ct_4.npy'},
        {'mr': 'data/t1w/6_mr_5.npy', 'ct': 'data/t1w/6_ct_5.npy'},
        {'mr': 'data/t1w/6_mr_6.npy', 'ct': 'data/t1w/6_ct_6.npy'},
        {'mr': 'data/t1w/6_mr_7.npy', 'ct': 'data/t1w/6_ct_7.npy'},
        {'mr': 'data/t1w/6_mr_8.npy', 'ct': 'data/t1w/6_ct_8.npy'},
        {'mr': 'data/t1w/6_mr_9.npy', 'ct': 'data/t1w/6_ct_9.npy'},
        {'mr': 'data/t1w/5_mr_0.npy', 'ct': 'data/t1w/5_ct_0.npy'},
        {'mr': 'data/t1w/5_mr_1.npy', 'ct': 'data/t1w/5_ct_1.npy'},
        {'mr': 'data/t1w/5_mr_10.npy', 'ct': 'data/t1w/5_ct_10.npy'},
        {'mr': 'data/t1w/5_mr_11.npy', 'ct': 'data/t1w/5_ct_11.npy'},
        {'mr': 'data/t1w/5_mr_12.npy', 'ct': 'data/t1w/5_ct_12.npy'},
        {'mr': 'data/t1w/5_mr_2.npy', 'ct': 'data/t1w/5_ct_2.npy'},
        {'mr': 'data/t1w/5_mr_3.npy', 'ct': 'data/t1w/5_ct_3.npy'},
        {'mr': 'data/t1w/5_mr_4.npy', 'ct': 'data/t1w/5_ct_4.npy'},
        {'mr': 'data/t1w/5_mr_5.npy', 'ct': 'data/t1w/5_ct_5.npy'},
        {'mr': 'data/t1w/5_mr_6.npy', 'ct': 'data/t1w/5_ct_6.npy'},
        {'mr': 'data/t1w/5_mr_7.npy', 'ct': 'data/t1w/5_ct_7.npy'},
        {'mr': 'data/t1w/5_mr_8.npy', 'ct': 'data/t1w/5_ct_8.npy'},
        {'mr': 'data/t1w/5_mr_9.npy', 'ct': 'data/t1w/5_ct_9.npy'},
    ]

    print(file_list_dict)

    test_dataset = slice_npy(file_list_dict, required_keys=cfg["required_keys"], is_channel_last=True, return_filename=True, init_verbose=True, transform=None)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=cfg["num_workers"], pin_memory=True)
    
    # Create a directory to save the predictions
    if not os.path.exists(cfg["root_dir"]):
        os.makedirs(cfg["root_dir"])

    # Test the model
    test_model(model, test_loader, device, cfg)
