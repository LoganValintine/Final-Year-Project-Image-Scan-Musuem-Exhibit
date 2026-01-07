"""
Config file. Handles everything the CycleGAN model uses. 

Version 2.0

Programmed by Lov2


"""

import torch
import albumentations as A
import os
from albumentations.pytorch import ToTensorV2
from PIL import Image


# Checks for device, if there isnt a GPU with 'Cuda' it uses the CPU (Much slower!)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Directory paths for the dataSets
TRAIN_DIR = "ImageGen/archive/MaleFemale"
TRAIN_DIRY2O = "ImageGen/archive/YoungOld"


# How many images size of images taken for training
BATCH_SIZE = 1

# Learning rate. 2e-4 is 0.0002
LEARNING_RATE = 2e-4

# Lambda Identity. Controls the strength of the identity loss. 0 = identity loss is disablwed
LAMBDA_IDENTITY = 0.0

# Controls the strength of the cylce consistency loss
LAMBDA_CYCLE = 10

# Number of workers (in this case gpu/cpus that are working)
NUM_WORKERS = 2

# Number of EPOCHS or training iterations. 5 is too low but works for small dataset training
NUM_EPOCHS = 100

#Loads and saves the model, set to False to disable,
LOAD_MODEL = False
SAVE_MODEL = False

# Directory for checkpoints when saving the model
CHECKPOINT_GEN_M = "genm.pth.tar"
CHECKPOINT_GEN_F = "genf.pth.tar"
CHECKPOINT_CRITIC_M = "criticm.pth.tar"
CHECKPOINT_CRITIC_F = "criticf.pth.tar"
CHECKPOINT_GEN_Y = "geny.pth.tar"
CHECKPOINT_GEN_O = "geno.pth.tar"
CHECKPOINT_CRITIC_Y = "criticy.pth.tar"
CHECKPOINT_CRITIC_O = "critico.pth.tar"
CHECKPOINT_DIR = "ImageGen/Checkpoints"

# Directory for saving
SAVE_DIRECTORY = "ImageGen/Saved_images"



"""
transforms images via normalisation and readies them for being touched via tensor. 

A.Resize - Resizes the image to 256 for W x H
A.Normalize - Turns the image into a numpy array. This allows it to be then transformed into a tensor image
ToTensorV2 - Turns image into a tensor image (e.g .png now becomes a set of array values.)


"""

transforms = A.Compose(
    [
        A.Resize(width=64, height=64),
        #A.HorizontalFlip(p=0.5)
        A.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5], max_pixel_value=255),
        ToTensorV2(),
    ],
    additional_targets={"image0": "image"},
)

# Performs transformation for the inputted image

transformInput = A.Compose(
    [
        A.Resize(width=64, height=64),
        #A.HorizontalFlip(p=0.5)
        A.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5], max_pixel_value=255),
        ToTensorV2(),
    ]
)

# Function to ensure a path exsits 

def validate_path():
    required_paths = [TRAIN_DIR, TRAIN_DIRY2O, SAVE_DIRECTORY]
    
    for path in required_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f" [config.py] Directory path is missing or cannot be found. Path in question: {path}")
        
        
# Function to crop images

def crop_image(pil_img, face_box, expand_ratio =2.0):
    
    x,y,w,h = face_box
    center_x = x + w // 2
    center_y = y + h // 2
    size = int(max(w,h) * expand_ratio)
    
    left = max(center_x - size // 2, 0)
    right = max(center_x + size // 2, pil_img.width)
    top = max(center_y - size // 2, 0)
    bottom = max(center_y + size // 2, pil_img.height)
    
    return pil_img.crop((left, top, bottom, right))