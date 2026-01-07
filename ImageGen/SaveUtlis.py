"""

File used to load and save training model checkpoints.


def save_checkpoint - Saves the model's weights to a .tar file

def load_checkpoints - Loads the .tar files as the models weights. 

Version 2.0

Programmed by Lov2


"""


import random, torch, os, numpy as np
import torch.nn as nn
import config
import copy


# Saves checkpoint weights.
def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    print("=> Saving Checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)
    
    
# Loads checkpoint weights 
def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    
    for param_groups in optimizer.param_groups:
        param_groups["ir"] = lr