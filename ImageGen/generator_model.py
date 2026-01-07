"""

Generator class for generating images. This generates fake images
that is used to trick the discriminator. 


Class ConvBlock - Creates a ConvBlock that is used in Generating a fake image.

Class Generator - Handles Generation via multiple ConvBlocks, from downsampling to then upsampling the newly fake image.


Version 2.0

Programmed by Lov2



"""

import torch
import torch.nn as nn

# Building convBlock. 
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, down=True, use_act=True, **kwargs):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, padding_mode="reflect", **kwargs)
            if down
            else nn.ConvTranspose2d(in_channels, out_channels, **kwargs),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace = True) if use_act else nn.Identity()
        )
        
    # Creates a convBlock.
    def forward(self, x):
        return self.conv(x)
    
#Residual Block which holds two different convBlocks.    
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            ConvBlock(channels, channels, kernel_size = 3, padding=1),
            ConvBlock(channels, channels, use_act=False, kernel_size = 3, padding = 1),
        )
        
    # Creates a residual block
    def forward(self, x):
        return x + self.block(x)
    
# Generator class, used for generation of fake images.     
class Generator(nn.Module):
    def __init__(self, img_channels, num_features = 64, num_residuals=9):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(img_channels, num_features, kernel_size=7, stride=1, padding=3, padding_mode="reflect"),
            nn.ReLU(inplace = True),
        )
        # Downsamples/encodes the images, ready for fake generation. 
        self.down_blocks = nn.ModuleList(
            [
                ConvBlock(num_features, num_features *2 , kernel_size = 3, stride = 2, padding = 1),
                ConvBlock(num_features*2, num_features *4 , kernel_size = 3, stride = 2, padding = 1),
            ]
        )
        # Learns the residual mapping between the input and output
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(num_features*4) for _ in range(num_residuals)]
        )
        # Upsamples the images, creating a fake image in the process
        self.up_blocks = nn.ModuleList(
            [
                ConvBlock(num_features*4, num_features*2, down=False, kernel_size=3, stride=2, padding=1, output_padding = 1),
                ConvBlock(num_features*2, num_features*1, down=False, kernel_size=3, stride=2, padding=1, output_padding = 1),
            ]
        )
        # Last sample to fully create fake image.
        self.last = nn.Conv2d(num_features*1, img_channels, kernel_size=7, stride=1, padding=3, padding_mode="reflect")
        
    # Creates a fake image.    
    def forward(self, x):
        x = self.initial(x)
        for layer in self.down_blocks:
            x = layer(x)
        x = self.residual_blocks(x)
        for layer in self.up_blocks:
            x = layer(x)
        return torch.tanh(self.last(x))