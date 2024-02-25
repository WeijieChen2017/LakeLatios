import torch
import torch.nn as nn
import torch.nn.functional as F

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

class decoder_UNETR_encoder_MedSAM(nn.Module):
    def __init__(
        self,
        img_size: int = 1024,
        patch_size: int = 16,
        in_chans: int = 3,
        out_chans: int = 1,
        out_chans_pretrain: int = 256,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        act_layer: Type[nn.Module] = nn.GELU,
        use_abs_pos: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        window_size: int = 0,
        global_attn_indexes: Tuple[int, ...] = (),
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
        self.global_attn_indexes = global_attn_indexes
        self.verbose = verbose

        # ENCODER modules
        self.patch_embed = PatchEmbed(
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            in_chans=in_chans,
            embed_dim=embed_dim,
        )

        self.pos_embed: Optional[nn.Parameter] = None
        if use_abs_pos:
            # Initialize absolute positional embedding with pretrain image size.
            self.pos_embed = nn.Parameter(
                torch.zeros(
                    1, img_size // patch_size, img_size // patch_size, embed_dim
                )
            )

        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                norm_layer=norm_layer,
                act_layer=act_layer,
                use_rel_pos=use_rel_pos,
                rel_pos_zero_init=rel_pos_zero_init,
                window_size=window_size if i not in global_attn_indexes else 0,
                input_size=(img_size // patch_size, img_size // patch_size),
            )
            self.blocks.append(block)

        self.neck = nn.Sequential(
            nn.Conv2d(
                embed_dim,
                out_chans_pretrain,
                kernel_size=1,
                bias=False,
            ),
            LayerNorm2d(out_chans_pretrain),
            nn.Conv2d(
                out_chans_pretrain,
                out_chans_pretrain,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            LayerNorm2d(out_chans_pretrain),
        )

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

        self._freeze_backbone()
        self._init_weights()

    def load_pretrain(self, pretrain_path, remove_prefix="image_encoder."):
        pretrain_dict = torch.load(pretrain_path, map_location="cpu")
        model_dict = self.state_dict()
        pretrain_dict = {k[len(remove_prefix):]: v for k, v in pretrain_dict.items() if k[len(remove_prefix):] in model_dict}
        model_dict.update(pretrain_dict)
        self.load_state_dict(model_dict)
        if self.verbose:
            print(f"load pretrain from {pretrain_path}")
    
    def _freeze_backbone(self):
        for param in self.patch_embed.parameters():
            param.requires_grad = False
        for param in self.blocks.parameters():
            param.requires_grad = False
        for param in self.neck.parameters():
            param.requires_grad = False
        # pos_embed is not a parameter
        if self.pos_embed is not None:
            self.pos_embed.requires_grad = False
        print("freeze backbone done")
    
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
        print("init weights done")

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        if self.verbose:
            print("x.shape", x.shape)

        zx = self.decoder_x(x)
        if self.verbose:
            print("zx.shape", zx.shape)

        x = self.patch_embed(x)
        if self.verbose:
            print("after patch_embed x.shape", x.shape)

        if self.pos_embed is not None:
            x = x + self.pos_embed
        if self.verbose:
            print("after pos_embed x.shape", x.shape)

        ViT_heads = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if self.verbose:
                print("after block", i, x.shape)
            if i in self.global_attn_indexes:
                ViT_heads.append(x)

        x = self.neck(x.permute(0, 3, 1, 2))
        if self.verbose:
            print("after neck x.shape", x.shape)

        # z3, z6, z9, z12 = ViT_heads
        [z3, z6, z9, z12] = ViT_heads
        if self.verbose:
            print("z3.shape", z3.shape, "z6.shape", z6.shape, "z9.shape", z9.shape, "z12.shape", z12.shape)
        z3 = self.z3_block(z3.permute(0, 3, 1, 2)) # B, 256, 1024, 1024
        z6 = self.z6_block(z6.permute(0, 3, 1, 2)) # B, 256, 512, 512
        z9 = self.z9_block(z9.permute(0, 3, 1, 2)) # B, 256, 256, 256
        z12 = self.z12_block(z12.permute(0, 3, 1, 2)) # B, 256, 128, 128
        zneck = self.z_neck_block(x) # B, 256, 128, 128
        if self.verbose:
            print("after z_block, z3.shape", z3.shape, "z6.shape", z6.shape, "z9.shape", z9.shape, "z12.shape", z12.shape, "zneck.shape", zneck.shape)

        ViT_heads = [
            zneck.cpu().detach().numpy(), 
            z12.cpu().detach().numpy(), 
            z9.cpu().detach().numpy(), 
            z6.cpu().detach().numpy(), 
            z3.cpu().detach().numpy()
        ]

        out = torch.cat([zneck, z12], dim=1) # B, 64px, 512ch
        if self.verbose:
            print("after cat zneck, z12, out.shape", out.shape)
        out = self.decoder_12(out) # B, 64px, 256ch
        if self.verbose:
            print("after decoder_12 out.shape", out.shape)
        ViT_heads.append(out.cpu().detach().numpy())

        out = torch.cat([out, z9], dim=1) # B, 128px, 512ch
        if self.verbose:
            print("after cat out, z9, out.shape", out.shape)
        out = self.decoder_9(out) # B, 128px, 128ch
        if self.verbose:
            print("after decoder_9 out.shape", out.shape)
        ViT_heads.append(out.cpu().detach().numpy())
        
        out = torch.cat([out, z6], dim=1) # B, 256px, 256ch
        if self.verbose:
            print("after cat out, z6, out.shape", out.shape)
        out = self.decoder_6(out) # B, 256px, 64ch
        if self.verbose:
            print("after decoder_6 out.shape", out.shape)
        ViT_heads.append(out.cpu().detach().numpy())

        out = torch.cat([out, z3], dim=1) # B, 512px, 128ch
        if self.verbose:
            print("after cat out, z3, out.shape", out.shape)
        out = self.decoder_3(out) # B, 512px, 32ch
        if self.verbose:
            print("after decoder_3 out.shape", out.shape)
        ViT_heads.append(out.cpu().detach().numpy())

        out = torch.cat([out, zx], dim=1) # B, 1024px, 64ch
        if self.verbose:
            print("after cat out, zx, out.shape", out.shape)
        out = self.decoder_out(out) # B, 1024px, 1ch
        if self.verbose:
            print("after decoder_out out.shape", out.shape)
        ViT_heads.append(out.cpu().detach().numpy())

        return out, ViT_heads
    
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