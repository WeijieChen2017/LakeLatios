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

class decoder_Deconv_encoder_MedSAM(nn.Module):
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

        #([2, 256, 64, 64])
        self.deconv = nn.Sequential(
            blue_block(256, 256, BN), # 64px -> 128px
            yellow_block(256, 128, BN), # 128px -> 128px
            blue_block(128, 128, BN), # 128px -> 256px
            yellow_block(128, 64, BN), # 256px -> 256px
            blue_block(64, 64, BN), # 256px -> 512px
            yellow_block(64, 32, BN), # 512px -> 512px
            blue_block(32, 32, BN), # 512px -> 1024px
            yellow_block(32, out_chans, BN), # 1024px -> 1024px
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

        x = self.patch_embed(x)
        if self.verbose:
            print("after patch_embed x.shape", x.shape)

        if self.pos_embed is not None:
            x = x + self.pos_embed
        if self.verbose:
            print("after pos_embed x.shape", x.shape)

        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if self.verbose:
                print("after block", i, x.shape)

        x = self.neck(x.permute(0, 3, 1, 2))
        if self.verbose:
            print("after neck x.shape", x.shape)
        
        x = self.deconv(x)
        if self.verbose:
            print("after deconv x.shape", x.shape)

        return x
