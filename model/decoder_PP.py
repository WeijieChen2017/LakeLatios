import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv_modules import AdjustedDWConv as DWConv
from .conv_modules import AdjustedYellowBlock as yellow_block
from .conv_modules import AdjustedBlueBlock as blue_block

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

class decoder_PP(nn.Module):
    def __init__(
        self,
        img_size: int = 1024,
        out_chans: int = 1,
        verbose: bool = False,
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

        # build decoder

        self.z_neck_block = nn.Sequential(
            yellow_block(256, 256),
        )

        self.z12_block = nn.Sequential(
            yellow_block(768, 256),
        )
        self.z9_block = nn.Sequential(
            # blue_block(768, 256),
            yellow_block(768, 256),
        )
        self.z6_block = nn.Sequential(
            blue_block(768, 256),
            # blue_block(256, 128),
            yellow_block(256, 128),
        )
        self.z3_block = nn.Sequential(
            blue_block(768, 256),
            blue_block(256, 128),
            # blue_block(128, 64),
            yellow_block(128, 64),
        )

        self.catconv = nn.Sequential(
            DWConv(256+256+256+128+64, 32),
            yellow_block(32, 32),
        )
        
        self.decoder_x = nn.Sequential(
            yellow_block(3, 32),
            yellow_block(32, 32),
        )        

        self.decoder_out = nn.Sequential(
            yellow_block(64, 32),
            yellow_block(32, out_chans),
            nn.Conv2d(out_chans, out_chans, kernel_size=1, bias=False),
        )

        self._freeze_backbone()
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

    def forward(self, 
                x: torch.Tensor,
                head_3: torch.Tensor,
                head_6: torch.Tensor,
                head_9: torch.Tensor,
                head_12: torch.Tensor,
                x_neck: torch.Tensor) -> torch.Tensor:
        
        if self.verbose:
            print("x.shape", x.shape, "mean and std:", x.mean(), x.std())

        zx = self.decoder_x(x)
        if self.verbose:
            print("zx.shape", zx.shape, "mean and std:", zx.mean(), zx.std())

        if self.verbose:
            print("head_3.shape", head_3.shape, "mean and std:", head_3.mean(), head_3.std())
            print("head_6.shape", head_6.shape, "mean and std:", head_6.mean(), head_6.std())
            print("head_9.shape", head_9.shape, "mean and std:", head_9.mean(), head_9.std())
            print("head_12.shape", head_12.shape, "mean and std:", head_12.mean(), head_12.std())
            print("x_neck.shape", x_neck.shape, "mean and std:", x_neck.mean(), x_neck.std())

        z3 = self.z3_block(head_3) # B, 256, 1024, 1024
        z6 = self.z6_block(head_6) # B, 256, 512, 512
        z9 = self.z9_block(head_9) # B, 256, 256, 256
        z12 = self.z12_block(head_12) # B, 256, 128, 128
        zneck = self.z_neck_block(x_neck) # B, 256, 128, 128
        if self.verbose:
            print("after z_block, z3.shape", z3.shape, "mean and std:", z3.mean(), z3.std())
            print("after z_block, z6.shape", z6.shape, "mean and std:", z6.mean(), z6.std())
            print("after z_block, z9.shape", z9.shape, "mean and std:", z9.mean(), z9.std())
            print("after z_block, z12.shape", z12.shape, "mean and std:", z12.mean(), z12.std())
            print("after z_neck_block, zneck.shape", zneck.shape, "mean and std:", zneck.mean(), zneck.std())

        # z12 and zneck 64px to 1024px
        z12 = F.interpolate(z12, scale_factor=4, mode="bilinear", align_corners=False)
        zneck = F.interpolate(zneck, scale_factor=4, mode="bilinear", align_corners=False)
        if self.verbose:
            print("after interpolation, z12.shape", z12.shape, "zneck.shape", zneck.shape, "mean and std:", z12.mean(), z12.std(), zneck.mean(), zneck.std())
        # z9 128px to 1024px
        z9 = F.interpolate(z9, scale_factor=4, mode="bilinear", align_corners=False)
        if self.verbose:
            print("after interpolation, z9.shape", z9.shape, "mean and std:", z9.mean(), z9.std())
        # z6 256px to 1024px
        z6 = F.interpolate(z6, scale_factor=2, mode="bilinear", align_corners=False)
        if self.verbose:
            print("after interpolation, z6.shape", z6.shape, "mean and std:", z6.mean(), z6.std())
        # z3 512px to 1024px
        # z3 = F.interpolate(z3, scale_factor=2, mode="bilinear", align_corners=False)
        # if self.verbose:
            # print("after interpolation, z3.shape", z3.shape)

        out = torch.cat([zneck, z12, z9, z6, z3], dim=1) # B, 1024px, 256*5ch
        if self.verbose:
            print("after cat, out.shape", out.shape, "mean and std:", out.mean(), out.std())
        out = self.catconv(out)
        if self.verbose:
            print("after catconv, out.shape", out.shape, "mean and std:", out.mean(), out.std())
        
        # out 512px to 1024px
        out = F.interpolate(out, scale_factor=4, mode="bilinear", align_corners=False)
        if self.verbose:
            print("after interpolation, out.shape", out.shape, "mean and std:", out.mean(), out.std())
        out = torch.cat([out, zx], dim=1)
        if self.verbose:
            print("after cat, out.shape", out.shape, "mean and std:", out.mean(), out.std())
        out = self.decoder_out(out)
        if self.verbose:
            print("after decoder_out, out.shape", out.shape, "mean and std:", out.mean(), out.std())

        return out