import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
import os
import json
import argparse
import numpy as np

from model import decoder_PyramidPooling_encoder_MedSAM, decoder_UNETR_encoder_MedSAM, decoder_Deconv_encoder_MedSAM
from dataset import PairedMRCTDataset_test  # Assuming the same dataset class can be used for testing

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
    else:
        raise ValueError("model_name not found !")
    
    # Load the encoder
    model.load_pretrain(cfg["pretrain_path"])
    # Load the decoder
    model.load_pretrain(cfg["decoder_path"], remove_prefix="")

    model.load_state_dict(torch.load(cfg["best_model_path"]))
    model.to(device)
    return model

def test_model(model, test_loader, device, cfg):
    model.eval()
    loss_function = nn.L1Loss()
    total_loss = 0.0
    with torch.no_grad():
        for batch in test_loader:
            MR, filename = batch["MR"].to(device)
            CT, _ = batch["CT"].to(device)
            pred = model(MR)
            loss = loss_function(pred, CT)
            # save the prediction in the save_dir
            save_name = os.path.join(cfg["root_dir"], filename)
            pred = pred.squeeze(0).detach().cpu().numpy()
            np.save(save_name, pred)
            print(f"Saved prediction for {filename}")
            total_loss += loss.item()
            # write the loss to a file
            with open(os.path.join(cfg["root_dir"], "loss.txt"), "a") as f:
                f.write(f"{filename}, {loss.item()}\n")
    avg_loss = total_loss / len(test_loader)
    print(f"Test Loss: {avg_loss}")
    # write the average loss to a file
    with open(os.path.join(cfg["root_dir"], "loss.txt"), "a") as f:
        f.write(f"Average Loss: {avg_loss}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load configuration for testing.")
    parser.add_argument("--cfg_path", type=str, default="config/test_PP_v1.json", help="Path to the configuration file.")
    args = parser.parse_args()
    cfg = json.load(open(args.cfg_path))
    
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    model = load_model(cfg, device)
    
    # Assuming the dataset class for testing is similar to training
    dataset_test = PairedMRCTDataset_test(
        path_MR=cfg["data_path_MR"],
        path_CT=cfg["data_path_CT"],
        stage="test",
        subset_fraction=cfg["subset_fraction"],
    )
    
    test_loader = DataLoader(dataset_test, batch_size=cfg["batch_size"], shuffle=False, num_workers=cfg["num_workers"], pin_memory=True)
    
    # Create a directory to save the predictions
    if not os.path.exists(cfg["root_dir"]):
        os.makedirs(cfg["root_dir"])

    # Test the model
    test_model(model, test_loader, device, cfg)
