"""
Discriminator model. Used to predict whether images are fake or real.


class Block - This is what the discriminator uses for image prediction. It uses a Conv LeakyReLu blocks for use in predicting fake or real images via sampling.

class Discriminator - Handles image prediction, creates conv Blocks for the amount of features and then uses them to sample the image, returning the now 'fake' image

Version 1.0

Programmed by Lov2

"""


import torch
import torch.nn as nn

# Block class 
class Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        
        # Create Conv block.
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4,stride, 1, bias=True, padding_mode = "reflect"),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )
        
    # returns a conv block.    
    def forward(self, x):
        return self.conv(x)
    
    
    
# Discriminator class, predicts how likely the generated image is to have come from the target image collection.    
class Discriminator(nn.Module):
    
    # Init class
    def __init__(self, in_channels=3, features=[64,128,256,512]):
        super().__init__()
        #Building block.
        self.inital = nn.Sequential(
            nn.Conv2d(
                in_channels,
                features[0],
                kernel_size=4,
                stride=2,
                padding=1,
                padding_mode="reflect",
            ),
            nn.LeakyReLU(0.2)
        )
            
        layers = []
        in_channels = features[0]
        
        # Looks at the number of features, and creates blocks using the block class for downsampling. 
        for feature in features[1:]:
            layers.append(Block(in_channels, feature, stride=1 if features==features[-1] else 2))
            in_channels = feature
        layers.append(nn.Conv2d(in_channels, 1, kernel_size = 4, stride=1, padding=1, padding_mode = "reflect"))
        self.model = nn.Sequential(*layers)
        
        
    # Sends the image back when in training. Predicting whether it is fake or real
    def forward(self, x):
        x = self.inital(x)
        return torch.sigmoid(self.model(x)) 