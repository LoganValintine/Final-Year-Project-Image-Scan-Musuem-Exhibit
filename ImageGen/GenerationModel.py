"""
Handles using the gender model (Masc -> Fem, Fem -> masc)


def preprocess - transforms the image from the gui into a np array
def chosen_model - loads the model based on selection via the GUI
def main - Handles everything, preprocess and gets the chosen model, then uses the Generator class to generate a fake image based on the chosen model using saved weights from training
def __inverse_main - Reverses the normalisation process. Allowing the image to be better translated back into a image able to be shown.


Programmed by Lov2

Version 1.0 

"""


import torch
import config
import numpy as np
import os
import torch.optim as optim
from PIL import Image
from generator_model import Generator
from SaveUtlis import load_checkpoint
from torchvision import transforms

# Preprocesses an image   
def preprocess(image_path, transform=None):
    
    
    input_image = np.array(Image.open(image_path).convert("RGB"))  
    
    if transform:
        augmentations = transform(image=input_image)
        input_image = augmentations(image=input_image)["image"]
        
    return input_image
    
        
    
# Creates a generator, and loads the model selected's checkpoints
def choosen_model(selection):
    
    gen = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)
    
    opt_gen = optim.Adam(
        list(gen.parameters()) + list(gen.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )
    
    
    
    if selection == "masc":
        
        load_checkpoint(os.path.join(config.CHECKPOINT_DIR,config.CHECKPOINT_GEN_M), gen, opt_gen, config.LEARNING_RATE)
        
    elif selection == "fem":
        
        load_checkpoint(os.path.join(config.CHECKPOINT_DIR,config.CHECKPOINT_GEN_F), gen, opt_gen, config.LEARNING_RATE)
        
    elif selection == "old":
        load_checkpoint(os.path.join(config.CHECKPOINT_DIR,config.CHECKPOINT_GEN_O), gen, opt_gen, config.LEARNING_RATE)
    
    elif selection == "young":
        
        load_checkpoint(os.path.join(config.CHECKPOINT_DIR,config.CHECKPOINT_GEN_Y), gen, opt_gen, config.LEARNING_RATE)
    
    return gen


# Performs image generation
def main(selection, image_path, transform=None):
    
    
    image = preprocess(image_path, transform)
    
    gen = choosen_model(selection)
      
    
    transform = config.transformInput(image=image)
    
    inputimg = transform['image']
    inputimg = inputimg.unsqueeze(0).to(config.DEVICE)
    
    
    with torch.no_grad():
        outputImg = gen(inputimg)
    
    outputImg = __inverse_norm(outputImg)
    
    
    # Assume image is (H, W, C) and values are in [0,1] range
    img = outputImg

    img = img.squeeze()

    print("Image shape before transpose:", img.shape)
    
    if img.ndim == 2:
        pil_image = Image.fromarray((img * 255).astype(np.uint8))
    elif img.ndim == 3:
        if img.shape[2] in [1,3]:
            img = (img * 255).astype(np.uint8)
            pil_image = Image.fromarray(img)
        else:
            raise ValueError(f"Unexpcted channel size")
    
    
    
    return pil_image
     
   
        
# Performs inverse normalisation        
def __inverse_norm(image):
    if isinstance(image, torch.Tensor):
        # Check if it's a 3D or 4D tensor
        if image.dim() == 3:  # (C, H, W)
            image = image.cpu().permute(1, 2, 0).numpy()  # (H, W, C)
        elif image.dim() == 4:  # (N, C, H, W)
            image = image.cpu().permute(0, 2, 3, 1).numpy()  # (N, H, W, C)
        else:
            raise ValueError(f"Expected 3D (C, H, W) or 4D (N, C, H, W) tensor, but got shape {image.shape}")

        # Reverse Normalize
        NORM_MEAN = np.array([0.5 , 0.5, 0.5])
        NORM_STD = np.array([0.5, 0.5, 0.5])
        
        image = (image * NORM_STD) + NORM_MEAN # Element-wise operation
        image = np.clip(image, a_min=0.0, a_max=1.0)
        
       
        

    return image