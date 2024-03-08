import torch
import torch.nn as nn

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

class decoder_UNETR(nn.Module):
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
            blue_block(768, 256),
        )
        self.z6_block = nn.Sequential(
            blue_block(768, 256),
            blue_block(256, 128),
        )
        self.z3_block = nn.Sequential(
            blue_block(768, 256),
            blue_block(256, 128),
            blue_block(128, 64),
        )

        self.decoder_12 = nn.Sequential(
            yellow_block(512, 512),
            yellow_block(512, 256),
            green_block(256, 256),
        )

        self.decoder_9 = nn.Sequential(
            yellow_block(512, 256),
            yellow_block(256, 128),
            green_block(128, 128),
        )

        self.decoder_6 = nn.Sequential(
            yellow_block(256, 128),
            yellow_block(128, 64),
            green_block(64, 64),
        )

        self.decoder_3 = nn.Sequential(
            yellow_block(128, 64),
            yellow_block(64, 32),
            green_block(32, 32),
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

        self._init_weights()

    def load_pretrain(self, pretrain_path, remove_prefix="image_encoder."):
        pretrain_dict = torch.load(pretrain_path, map_location="cpu")
        model_dict = self.state_dict()
        pretrain_dict = {k[len(remove_prefix):]: v for k, v in pretrain_dict.items() if k[len(remove_prefix):] in model_dict}
        model_dict.update(pretrain_dict)
        self.load_state_dict(model_dict)
        if self.verbose:
            print(f"load pretrain from {pretrain_path}")

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
            # elif isinstance(m, nn.InstanceNorm2d):
            #     if self.verbose:
            #         print("init instancenorm for", m)
            #     nn.init.constant_(m.bias, 0)
            #     nn.init.constant_(m.weight, 1.0)
        print("init weights done")

    def forward(self, 
                x: torch.Tensor,
                head_3: torch.Tensor,
                head_6: torch.Tensor,
                head_9: torch.Tensor,
                head_12: torch.Tensor,
                x_neck: torch.Tensor) -> torch.Tensor:

        if self.verbose:
            print("x.shape", x.shape)

        zx = self.decoder_x(x)
        if self.verbose:
            print("zx.shape", zx.shape)

        if self.verbose:
            print("Head shapes: ", head_3.shape, head_6.shape, head_9.shape, head_12.shape, x_neck.shape)
        z3 = self.z3_block(head_3) # B, 256, 1024, 1024
        z6 = self.z6_block(head_6) # B, 256, 512, 512
        z9 = self.z9_block(head_9) # B, 256, 256, 256
        z12 = self.z12_block(head_12) # B, 256, 128, 128
        zneck = self.z_neck_block(x_neck) # B, 256, 128, 128
        if self.verbose:
            print("after z_block, z3.shape", z3.shape, "z6.shape", z6.shape, "z9.shape", z9.shape, "z12.shape", z12.shape, "zneck.shape", zneck.shape)

        out = torch.cat([zneck, z12], dim=1) # B, 64px, 512ch
        if self.verbose:
            print("after cat zneck, z12, out.shape", out.shape)
        out = self.decoder_12(out) # B, 64px, 256ch
        if self.verbose:
            print("after decoder_12 out.shape", out.shape)

        out = torch.cat([out, z9], dim=1) # B, 128px, 512ch
        if self.verbose:
            print("after cat out, z9, out.shape", out.shape)
        out = self.decoder_9(out) # B, 128px, 128ch
        if self.verbose:
            print("after decoder_9 out.shape", out.shape)
        
        out = torch.cat([out, z6], dim=1) # B, 256px, 256ch
        if self.verbose:
            print("after cat out, z6, out.shape", out.shape)
        out = self.decoder_6(out) # B, 256px, 64ch
        if self.verbose:
            print("after decoder_6 out.shape", out.shape)

        out = torch.cat([out, z3], dim=1) # B, 512px, 128ch
        if self.verbose:
            print("after cat out, z3, out.shape", out.shape)
        out = self.decoder_3(out) # B, 512px, 32ch
        if self.verbose:
            print("after decoder_3 out.shape", out.shape)

        out = torch.cat([out, zx], dim=1) # B, 1024px, 64ch
        if self.verbose:
            print("after cat out, zx, out.shape", out.shape)
        out = self.decoder_out(out) # B, 1024px, 1ch
        if self.verbose:
            print("after decoder_out out.shape", out.shape)
            # ViT_heads.append(out.permute(0, 2, 3, 1).cpu().detach().numpy())

        return out
    
# x.shape torch.Size([2, 3, 1024, 1024])
# zx.shape torch.Size([2, 32, 1024, 1024])
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
# z3.shape torch.Size([2, 64, 64, 768]) z6.shape torch.Size([2, 64, 64, 768]) z9.shape torch.Size([2, 64, 64, 768]) z12.shape torch.Size([2, 64, 64, 768])
# after z_block, z3.shape torch.Size([2, 64, 512, 512]) z6.shape torch.Size([2, 128, 256, 256]) z9.shape torch.Size([2, 256, 128, 128]) z12.shape torch.Size([2, 256, 64, 64]) zneck.shape torch.Size([2, 256, 64, 64])
# after cat zneck, z12, out.shape torch.Size([2, 512, 64, 64])
# after decoder_12 out.shape torch.Size([2, 256, 128, 128])
# after cat out, z9, out.shape torch.Size([2, 512, 128, 128])
# after decoder_9 out.shape torch.Size([2, 128, 256, 256])
# after cat out, z6, out.shape torch.Size([2, 256, 256, 256])
# after decoder_6 out.shape torch.Size([2, 64, 512, 512])
# after cat out, z3, out.shape torch.Size([2, 128, 512, 512])
# after decoder_3 out.shape torch.Size([2, 32, 1024, 1024])
# after cat out, zx, out.shape torch.Size([2, 64, 1024, 1024])
# after decoder_out out.shape torch.Size([2, 1, 1024, 1024])
# torch.Size([2, 1, 1024, 1024])