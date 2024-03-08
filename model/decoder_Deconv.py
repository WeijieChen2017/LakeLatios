import torch
import torch.nn as nn

from typing import Optional, Tuple, Type
from .image_encoder import PatchEmbed, Block, LayerNorm2d
from .conv_modules import AdjustedYellowBlock as yellow_block
from .conv_modules import AdjustedBlueBlock as blue_block
from .conv_modules import AdjustedGreenBlock as green_block

# x.shape torch.Size([2, 3, 1024, 1024])
# after patch_embed x.shape torch.Size([2, 64, 64, 768])
# after pos_embed x.shape torch.Size([2, 64, 64, 768])
# after block 0 torch.Size([2, 64, 64, 768])
# after block 1 torch.Size([2, 64, 64, 768])
# after block 2 torch.Size([2, 64, 64, 768])
# after block 3 torch.Size([2, 64, 64, 768])
# after block 4 torch.Size([2, 64, 64, 768])
# after block 5 torch.Size([2, 64, 64, 768])
# after block 6 torch.Size([2, 64, 64, 768])
# after block 7 torch.Size([2, 64, 64, 768])
# after block 8 torch.Size([2, 64, 64, 768])
# after block 9 torch.Size([2, 64, 64, 768])
# after block 10 torch.Size([2, 64, 64, 768])
# after block 11 torch.Size([2, 64, 64, 768])
# after neck x.shape torch.Size([2, 256, 64, 64])
# torch.Size([2, 256, 64, 64])

class decoder_Deconv(nn.Module):
    def __init__(
        self,
        img_size: int = 1024,
        out_chans: int = 1,
        verbose: bool = False,
        BN: bool = False,
    ) -> None:
        """
        Args:
            img_size (int): Input image size.
            patch_size (int): Patch size.
            in_chans (int): Number of input image channels.
            embed_dim (int): Patch embedding dimension.
            depth (int): Depth of ViT.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_abs_pos (bool): If True, use absolute positional embeddings.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            window_size (int): Window size for window attention blocks.
            global_attn_indexes (list): Indexes for blocks using global attention.
        """
        super().__init__()
        self.img_size = img_size
        self.verbose = verbose

        #([2, 256, 64, 64])
        self.deconv = nn.Sequential(
            blue_block(256, 256, 2, BN), # 64px -> 128px
            yellow_block(256, 128, 2, BN), # 128px -> 128px
            blue_block(128, 128, 2, BN), # 128px -> 256px
            yellow_block(128, 64, 2, BN), # 256px -> 256px
            blue_block(64, 64, 2, BN), # 256px -> 512px
            yellow_block(64, 32, 2, BN), # 512px -> 512px
            blue_block(32, 32, 2, BN), # 512px -> 1024px
            yellow_block(32, out_chans, 2, BN), # 1024px -> 1024px
            nn.Conv2d(out_chans, out_chans, kernel_size=1, bias=False),
        )

        self._init_weights()
    
    def load_from_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        self.load_state_dict(checkpoint)
        print(f"load from checkpoint {checkpoint_path}")
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if self.verbose:
                    print("init conv2d for", m)
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                if self.verbose:
                    print("init linear for", m)
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                if self.verbose:
                    print("init layernorm for", m)
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            elif isinstance(m, nn.InstanceNorm2d):
                if self.verbose:
                    print("init instancenorm for", m)
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
        print("init weights done")

    def forward(self, x_neck: torch.Tensor) -> torch.Tensor:
        
        x = self.deconv(x_neck)
        if self.verbose:
            print("after deconv x.shape", x.shape)

        return x
