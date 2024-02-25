import torch
import torch.nn as nn

class AdjustedDWConv(nn.Module):
    def __init__(self, in_chans, out_chans, BN=False):
        super(AdjustedDWConv, self).__init__()

        self.norm_layer = nn.BatchNorm2d if BN else nn.InstanceNorm2d
        self.conv = nn.Sequential(
            # Depthwise convolution
            nn.Conv2d(in_chans, in_chans, kernel_size=3, padding=1, groups=in_chans, bias=False),
            self.norm_layer(in_chans),
            nn.GELU(),
            
            # Pointwise convolution
            nn.Conv2d(in_chans, out_chans, kernel_size=1, bias=False),
            self.norm_layer(out_chans),
            nn.GELU(),
        )

    def forward(self, x):
        return self.conv(x)
    
class AdjustedYellowBlock(nn.Module):
    def __init__(self, in_chans, out_chans, n_blocks=2, BN=False):
        super(AdjustedYellowBlock, self).__init__()
        self.skip = in_chans == out_chans
        self.blocks = nn.ModuleList()

        self.norm_layer = nn.BatchNorm2d if BN else nn.InstanceNorm2d
        for _ in range(n_blocks):
            self.blocks.append(nn.Sequential(
                nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1, bias=False),
                self.norm_layer(out_chans),
                nn.GELU(),
            ))
            in_chans = out_chans  # Adjust for the first loop, subsequent loops have matching in and out channels

    def forward(self, x):
        out = x
        for block in self.blocks:
            out = block(out)
        
        if self.skip:
            out += x
        return out
    

class AdjustedGreenBlock(nn.Module):
    def __init__(self, in_chans, out_chans, BN=False):
        super(AdjustedGreenBlock, self).__init__()

        self.norm_layer = nn.BatchNorm2d if BN else nn.InstanceNorm2d
        self.blocks = nn.Sequential(
            nn.ConvTranspose2d(in_chans, out_chans, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            self.norm_layer(out_chans),
            nn.GELU(),
        )

    def forward(self, x):
        return self.blocks(x)
    
class AdjustedBlueBlock(nn.Module):
    def __init__(self, in_chans, out_chans):
        super(AdjustedBlueBlock, self).__init__()
        # Initialize the adjusted green_block with GELU and InstanceNorm
        self.green = AdjustedGreenBlock(in_chans, out_chans)
        # Initialize the adjusted yellow_block with GELU and InstanceNorm
        self.yellow = AdjustedYellowBlock(out_chans, out_chans)
        # Sequentially combine the two blocks
        self.blocks = nn.Sequential(
            self.green,
            self.yellow,
        )

    def forward(self, x):
        # Pass the input through the sequential blocks
        return self.blocks(x)
    


# class yellow_block(nn.Module):
#     def __init__(self, in_chans, out_chans, n_blocks=2):
#         super(yellow_block, self).__init__()
#         self.skip = in_chans == out_chans
#         self.block_list = []
#         self.block_list.append(nn.Sequential(
#             nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1, bias=False),
#             LayerNorm2d(out_chans),
#             nn.GELU(),
#         ))
#         for i in range(n_blocks-1):
#             self.block_list.append(nn.Sequential(
#                 nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1, bias=False),
#                 LayerNorm2d(out_chans),
#                 nn.GELU(),
#             ))
#         self.blocks = nn.Sequential(*self.block_list)

#     def forward(self, x):
#         if self.skip:
#             return x + self.blocks(x)
#         else:
#             return self.blocks(x)

# class green_block(nn.Module):
#     def __init__(self, in_chans, out_chans):
#         super(green_block, self).__init__()
#         self.blocks = nn.Sequential(
#             # use convtranspose2d to upsample
#             nn.ConvTranspose2d(in_chans, out_chans, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
#             LayerNorm2d(out_chans),
#             nn.GELU(),
#         )

#     def forward(self, x):
#         return self.blocks(x)

# class blue_block(nn.Module):
#     def __init__(self, in_chans, out_chans):
#         super(blue_block, self).__init__()
#         self.green = green_block(in_chans, out_chans)
#         self.yellow = yellow_block(out_chans, out_chans)
#         self.blocks = nn.Sequential(
#             self.green,
#             self.yellow,
#         )

#     def forward(self, x):
#         return self.blocks(x)