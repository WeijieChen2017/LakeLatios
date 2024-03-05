import torch
from torch import nn
from monai.networks.nets import UNet

class UNet_MONAI(nn.Module):
    def __init__(self, 
                spatial_dims=2,
                in_channels=3,
                out_channels=1, 
                channels=(16, 32, 64, 128, 256),
                strides=(2, 2, 2, 2),
                kernel_size=3, 
                up_kernel_size=3, 
                num_res_units=0, 
                act='PRELU', 
                norm='INSTANCE', 
                dropout=0.0, 
                bias=True, 
                adn_ordering='NDA'):
        super(UNet_MONAI, self).__init__()

        self.spatial_dims = spatial_dims
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.channels = channels
        self.strides = strides
        self.kernel_size = kernel_size
        self.up_kernel_size = up_kernel_size
        self.num_res_units = num_res_units
        self.act = act
        self.norm = norm
        self.dropout = dropout
        self.bias = bias
        self.adn_ordering = adn_ordering
    
        self.model = UNet(
            spatial_dims=self.spatial_dims, 
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            channels=self.channels,
            strides=self.strides,
            kernel_size=self.kernel_size,
            up_kernel_size=self.up_kernel_size,
            num_res_units=self.num_res_units,
            act=self.act,
            norm=self.norm,
            dropout=self.dropout,
            bias=self.bias,
            adn_ordering=self.adn_ordering
        )

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

    def load_from_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        self.load_state_dict(checkpoint)
        print(f"load from checkpoint {checkpoint_path}")
    
    def load_pretrain(self, pretrain_path):
        pass

    def forward(self, x):
        x = self.model(x)
        return x
    
    