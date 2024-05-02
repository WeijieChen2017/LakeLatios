import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, Tuple, Type
from timm.models.layers import trunc_normal_
from torch.nn.init import normal_



from .MSDeformAttn import MSDeformAttn
from .image_encoder import PatchEmbed, Block, LayerNorm2d
from .ViTadapter_modules import SpatialPriorModule, Extractor, Injector, deform_inputs
from .ViTadapter_modules import PatchEmbed_adapter, LayerNorm, InteractionBlock_frozenViT

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



class decoder_ViTadapter_encoder_MedSAM(nn.Module):
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
        # adapter parameters
        conv_inplane=64,
        out_indices=(0, 1, 2, 3),
        use_final_norm = True,
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
        self.use_final_norm = use_final_norm

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

        # build adapter modules
        self.level_embed = nn.Parameter(torch.zeros(3, embed_dim))
        self.spm = SpatialPriorModule(inplanes=conv_inplane, embed_dim=embed_dim,
                                      out_indices=out_indices)
        self.interactions = nn.Sequential(*[
            InteractionBlock_frozenViT(dim=embed_dim, num_heads=deform_num_heads, n_points=n_points,
                             init_values=init_values, drop_path=self.drop_path_rate,
                             norm_layer=nn.LayerNorm, with_cffn=with_cffn,
                             cffn_ratio=cffn_ratio, deform_ratio=deform_ratio,
                             extra_extractor=((True if i == len(
                                 interaction_indexes) - 1 else False) and use_extra_extractor),
                             with_cp=with_cp)
            for i in range(len(interaction_indexes))
        ])
        self.patch_embed_adapter = PatchEmbed_adapter(img_size, patch_size, in_chans, embed_dim)
        num_patches_adapter = self.patch_embed_adapter.num_patches
        self.num_patches = num_patches_adapter
        self.pos_embed_adapter = nn.Parameter(torch.zeros(1, num_patches_adapter + 1, embed_dim))
        self.pos_drop_adapter = nn.Identity()
        self.cls_token_adapter = nn.Parameter(torch.zeros(1, 1, embed_dim))

        if len(out_indices) == 4:
            self.up = nn.ConvTranspose2d(embed_dim, embed_dim, 2, 2)
            if self.use_final_norm:
                self.norm1 = LayerNorm(embed_dim)
            self.up.apply(self._init_weights)

        if self.use_final_norm:
            self.norm2 = LayerNorm(embed_dim)
            self.norm3 = LayerNorm(embed_dim)
            self.norm4 = LayerNorm(embed_dim)

        self.spm.apply(self._init_weights)
        self.interactions.apply(self._init_weights)
        self.apply(self._init_deform_weights)
        normal_(self.level_embed)



        # encoder init
        self._freeze_backbone()
        self._init_weights()

    def _init_weights_adapter(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm) or isinstance(m, LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def _init_deform_weights(self, m):
        if isinstance(m, MSDeformAttn):
            m._reset_parameters()

    def _add_level_embed(self, c2, c3, c4):
        c2 = c2 + self.level_embed[0]
        c3 = c3 + self.level_embed[1]
        c4 = c4 + self.level_embed[2]
        return c2, c3, c4


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


        # adapter forward
        deform_inputs1, deform_inputs2 = deform_inputs(x)
        x = x.to(self.dtype)
        # SPM forward
        if len(self.out_indices) == 4:
            c1, c2, c3, c4 = self.spm(x)
        else:
            c2, c3, c4 = self.spm(x)

        c2, c3, c4 = self._add_level_embed(c2, c3, c4)
        c = torch.cat([c2, c3, c4], dim=1)

        # adapter Patch Embedding forward
        x_a, H, W = self.patch_embed_a(x)
        bs, n, dim = x_a.shape
        # pos_embed = self._get_pos_embed(self.pos_embed[:, 1:], H, W)
        x_a = self.pos_drop_adapter(x_a + self.pos_embed_adapter[:, 1:])

        # encoder forward
        x = self.patch_embed(x)

        if self.pos_embed is not None:
            x = x + self.pos_embed

        for i, blk in enumerate(self.blocks):
            x = blk(x)
        # adapter output

        # Split & Reshape
        c2 = c[:, 0:c2.size(1), :]
        c3 = c[:, c2.size(1):c2.size(1) + c3.size(1), :]
        c4 = c[:, c2.size(1) + c3.size(1):, :]

        c2 = c2.transpose(1, 2).view(bs, dim, H * 2, W * 2).contiguous()
        c3 = c3.transpose(1, 2).view(bs, dim, H, W).contiguous()
        c4 = c4.transpose(1, 2).view(bs, dim, H // 2, W // 2).contiguous()
        if len(self.out_indices) == 4:
            c1 = self.up(c2) + c1

        if self.add_vit_feature:
            if len(self.out_indices) == 4:
                x3 = x_a.transpose(1, 2).view(bs, dim, H, W).contiguous()
                x1 = F.interpolate(x3, scale_factor=4, mode='bilinear', align_corners=False)
                x2 = F.interpolate(x3, scale_factor=2, mode='bilinear', align_corners=False)
                x4 = F.interpolate(x3, scale_factor=0.5, mode='bilinear', align_corners=False)
                c1, c2, c3, c4 = c1 + x1, c2 + x2, c3 + x3, c4 + x4
            else:
                x3 = x_a.transpose(1, 2).view(bs, dim, H, W).contiguous()
                x2 = F.interpolate(x3, scale_factor=2, mode='bilinear', align_corners=False)
                x4 = F.interpolate(x3, scale_factor=0.5, mode='bilinear', align_corners=False)
                c2, c3, c4 = c2 + x2, c3 + x3, c4 + x4

        # Final Norm
        if self.use_final_norm:
            if len(self.out_indices) == 4:
                f1 = self.norm1(c1.float()).contiguous()
                f2 = self.norm2(c2.float()).contiguous()
                f3 = self.norm3(c3.float()).contiguous()
                f4 = self.norm4(c4.float()).contiguous()
                return [f1, f2, f3, f4]
            else:
                f2 = self.norm2(c2.float()).contiguous()
                f3 = self.norm3(c3.float()).contiguous()
                f4 = self.norm4(c4.float()).contiguous()
                return [f2, f3, f4]
        else:
            return [c1.float().contiguous(),
                    c2.float().contiguous(),
                    c3.float().contiguous(),
                    c4.float().contiguous()]


        return x