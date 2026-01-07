"""

Young to old training model. This uses the discrimantor and generator 
to create fake images of young-> old transformation and vice versa

def train_fn - carries out the training process, gathering the loss in image whilst performing the generation and discrimination of the images. This is carried out via 'Cuda' Autocast
def main - Creates the generator, discriminator, dataset, dataloader and handles saving/loading the weights, runs the train function based on the epochs

Version 1.0

Programmed by Lov2

"""

import torch
import os
import torch.nn as nn
import torch.optim as optim
import config
from tqdm import tqdm
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from discriminator_model import Discriminator
from generator_model import Generator
from torchvision import transforms
from SaveUtlis import save_checkpoint, load_checkpoint
from OYDataset import OldYoungDataset

import sys
sys.path.append("ImageGen")

project_dir = os.path.dirname(os.path.abspath(__file__))  # Absolute path to the script's directory
young_path = os.path.join(project_dir, 'ImageGen', 'archive', 'YoungOld', 'Young')

# Training cylce, iterates through the discriminator and generator using different images
def train_fn(disc_O, disc_Y, gen_Y, gen_O, loader, opt_disc, opt_gen, l1, mse, d_scaler, g_scaler):

        O_reals = 0
        O_fakes = 0
        # Tracks progress
        loop = tqdm(loader, leave=True)
        
        # Iterates through dataset
        for idx, (young, old) in enumerate(loop):
            young = young.to(config.DEVICE)
            old = old.to(config.DEVICE)

            # Train Discriminators M and F
            with torch.amp.autocast(device_type="cuda"):
                fake_old = gen_O(young)
                D_O_real = disc_O(old)
                D_O_fake = disc_O(fake_old.detach())
                O_reals += D_O_real.mean().item()
                O_fakes += D_O_fake.mean().item()
                
                # Finds loss using mse
                D_O_real_loss = mse(D_O_real, torch.ones_like(D_O_real))
                D_O_fake_loss = mse(D_O_fake, torch.zeros_like(D_O_fake))
                D_O_loss = D_O_real_loss + D_O_fake_loss

                fake_young = gen_Y(old)
                D_Y_real = disc_Y(young)
                D_Y_fake = disc_Y(fake_young.detach())
                D_Y_real_loss = mse(D_Y_real, torch.ones_like(D_Y_real))
                D_Y_fake_loss = mse(D_Y_fake, torch.zeros_like(D_Y_fake))
                D_Y_loss = D_Y_real_loss + D_Y_fake_loss


                # Put it together
                D_loss = (D_O_loss + D_Y_loss) / 2


            opt_disc.zero_grad()
            d_scaler.scale(D_loss).backward()
            d_scaler.step(opt_disc)
            d_scaler.update()

            # Train Generators M and F
        
            with torch.amp.autocast(device_type="cuda"):
                D_O_fake = disc_O(fake_old)
                D_Y_fake = disc_Y(fake_young)
                loss_G_O = mse(D_O_fake, torch.ones_like(D_O_fake))
                loss_G_Y = mse(D_Y_fake, torch.ones_like(D_Y_fake))

                # Cycle loss
                cycle_young = gen_Y(fake_young)
                cycle_old = gen_O(fake_young)
                cycle_old_loss = l1(old, cycle_old)
                cycle_young_loss = l1(young, cycle_young)

                # identity loss
                identity_old = gen_O(old)
                identity_young = gen_Y(young)
                identity_old_loss = l1(old, identity_old)
                identity_young_loss = l1(young, identity_young)


                # Add All together
                G_loss = (
                    loss_G_Y
                    + loss_G_O
                    + cycle_young_loss * config.LAMBDA_CYCLE
                    + cycle_old_loss * config.LAMBDA_CYCLE
                    + identity_old_loss * config.LAMBDA_IDENTITY
                    + identity_young_loss * config.LAMBDA_IDENTITY
                )

            opt_gen.zero_grad()
            g_scaler.scale(G_loss).backward()
            g_scaler.step(opt_gen)
            g_scaler.update()


            # Saves images every 20 iteration. e.g 0, 20, 40, 60 etc.
            if idx % 20 == 0:
                save_image(fake_old * 0.5 + 0.5, os.path.join(config.SAVE_DIRECTORY, f"old_{idx}.png"))
                save_image(fake_young * 0.5 + 0.5, os.path.join(config.SAVE_DIRECTORY,f"young_{idx}.png"))

            loop.set_postfix(O_real = O_reals / (idx + 1), O_fake = O_fakes/(idx+1))
            
            
# Main file that runs training and creates datasets/dataloader
def main():
    
    # Initalises Discriminator and Generator.
    disc_O = Discriminator(in_channels=3).to(config.DEVICE)
    disc_Y = Discriminator(in_channels=3).to(config.DEVICE)
    gen_Y = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)
    gen_O = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)

    opt_disc = optim.Adam(
        list(disc_O.parameters()) + list(disc_Y.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    opt_gen = optim.Adam(
        list(gen_Y.parameters()) + list(gen_O.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    L1 = nn.L1Loss()
    mse = nn.MSELoss()

    # Used to load a models weights. 
    if config.LOAD_MODEL:
        load_checkpoint(config.CHECKPOINT_GEN_O, gen_O, opt_gen, config.LEARNING_RATE)
        load_checkpoint(config.CHECKPOINT_GEN_Y, gen_Y, opt_gen, config.LEARNING_RATE)
        load_checkpoint(config.CHECKPOINT_CRITIC_O, disc_O, opt_disc, config.LEARNING_RATE)
        load_checkpoint(config.CHECKPOINT_CRITIC_Y, disc_Y, opt_disc, config.LEARNING_RATE)

    
    # Ensure paths exist before creating dataset
    
    config.validate_path() 
    
    # Creates training Dataset.
    dataset = oldyoungDataset(
        root_old=os.path.join(config.TRAIN_DIRY2O, "Old"),
        root_young=os.path.join(config.TRAIN_DIRY2O, "Young"),
        transform=config.transforms,
    )
    
    print(f"Dataset Legnth: { len(dataset)}")
    # Creates dataloader
    loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )
    g_scaler = torch.amp.GradScaler()
    d_scaler = torch.amp.GradScaler()

    # Iterates through the whole dataset to produce more and more accurate results based on number of epochs. 
    for epoch in range(config.NUM_EPOCHS):
        train_fn(disc_O, disc_Y, gen_Y, gen_O, loader, opt_disc, opt_gen, L1, mse, d_scaler, g_scaler,)
        
        
        # Saves the model's weight every time epoch is fully ran
        if config.SAVE_MODEL:

            save_checkpoint(gen_O, opt_gen, filename=os.path.join(config.CHECKPOINT_DIR,config.CHECKPOINT_GEN_O))
            save_checkpoint(gen_Y, opt_gen, filename=os.path.join(config.CHECKPOINT_DIR,config.CHECKPOINT_GEN_Y))
            save_checkpoint(disc_O, opt_disc, filename=os.path.join(config.CHECKPOINT_DIR,config.CHECKPOINT_CRITIC_O))
            save_checkpoint(disc_Y, opt_disc, filename=os.path.join(config.CHECKPOINT_DIR,config.CHECKPOINT_CRITIC_Y))

if __name__ =="__main__":

    main()