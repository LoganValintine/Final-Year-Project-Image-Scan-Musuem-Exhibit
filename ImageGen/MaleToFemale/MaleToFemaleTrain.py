"""

Male to female training model. This uses the discrimantor and generator 
to create fake images of male-> female transformation and vice versa

WARNING : THIS WONT WORK ON PYCHARM/OTHER IDES. FOR THEM TO WORK IT NEEDS TO HAVE A RUNNING CONFIG WHERE lov2majorproject -> WORKING DIRECTORY

def train_fn - carries out the training process, gathering the loss in image whilst performing the generation and discrimination of the images. This is carried out via 'Cuda' Autocast

def main - Creates the generator, discriminator, dataset, dataloader and handles saving/loading the weights, runs the train function based on the epochs

Version 2.0

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
from MFDataset import MaleFemaleDataset

import sys
sys.path.append('ImageGen')
# Training cylce, iterates through the discriminator and generator using different images
def train_fn(disc_M, disc_F, gen_F, gen_M, loader, opt_disc, opt_gen, l1, mse, d_scaler, g_scaler):

        M_reals = 0
        M_fakes = 0
        # Tracks progress
        loop = tqdm(loader, leave=True)
        
        # Iterates through dataset
        for idx, (female, male) in enumerate(loop):
            female = female.to(config.DEVICE)
            male = male.to(config.DEVICE)

            # Train Discriminators M and F
            with torch.amp.autocast(device_type=config.DEVICE):
                fake_male = gen_M(female)
                D_M_real = disc_M(male)
                D_M_fake = disc_M(fake_male.detach())
                M_reals += D_M_real.mean().item()
                M_fakes += D_M_fake.mean().item()
                
                # Finds loss using mse
                D_M_real_loss = mse(D_M_real, torch.ones_like(D_M_real))
                D_M_fake_loss = mse(D_M_fake, torch.zeros_like(D_M_fake))
                D_M_loss = D_M_real_loss + D_M_fake_loss

                fake_female = gen_F(male)
                D_F_real = disc_F(female)
                D_F_fake = disc_F(fake_female.detach())
                D_F_real_loss = mse(D_F_real, torch.ones_like(D_F_real))
                D_F_fake_loss = mse(D_F_fake, torch.zeros_like(D_F_fake))
                D_F_loss = D_F_real_loss + D_F_fake_loss


                # Put it together
                D_loss = (D_M_loss + D_F_loss) / 2


            opt_disc.zero_grad()
            d_scaler.scale(D_loss).backward()
            d_scaler.step(opt_disc)
            d_scaler.update()

            # Train Generators M and F
        
            with torch.amp.autocast(device_type=config.DEVICE):
                D_M_fake = disc_M(fake_male)
                D_F_fake = disc_F(fake_female)
                loss_G_M = mse(D_M_fake, torch.ones_like(D_M_fake))
                loss_G_F = mse(D_F_fake, torch.ones_like(D_F_fake))

                # Cycle loss
                cycle_female = gen_F(fake_male)
                cycle_male = gen_M(fake_female)
                cycle_male_loss = l1(male, cycle_male)
                cycle_female_loss = l1(female, cycle_female)

                # identity loss
                identity_male = gen_M(male)
                identity_female = gen_F(female)
                identity_male_loss = l1(male, identity_male)
                identity_female_loss = l1(female, identity_female)


                # Add All together
                G_loss = (
                    loss_G_F
                    + loss_G_M
                    + cycle_female_loss * config.LAMBDA_CYCLE
                    + cycle_male_loss * config.LAMBDA_CYCLE
                    + identity_male_loss * config.LAMBDA_IDENTITY
                    + identity_female_loss * config.LAMBDA_IDENTITY
                )

            opt_gen.zero_grad()
            g_scaler.scale(G_loss).backward()
            g_scaler.step(opt_gen)
            g_scaler.update()


            # Saves images every 20 iteration. e.g 0, 20, 40, 60 etc.
            if idx % 20 == 0:
                save_image(fake_male * 0.5 + 0.5, os.path.join(config.SAVE_DIRECTORY, f"male_{idx}.png"))
                save_image(fake_female * 0.5 + 0.5, os.path.join(config.SAVE_DIRECTORY,f"female_{idx}.png"))

            loop.set_postfix(M_real = M_reals / (idx + 1), M_fake = M_fakes/(idx+1))
            
            
# Main file that runs training and creates datasets/dataloader
def main():
    
    # Initalises Discriminator and Generator.
    disc_M = Discriminator(in_channels=3).to(config.DEVICE)
    disc_F = Discriminator(in_channels=3).to(config.DEVICE)
    gen_F = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)
    gen_M = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)

    opt_disc = optim.Adam(
        list(disc_M.parameters()) + list(disc_F.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    opt_gen = optim.Adam(
        list(gen_F.parameters()) + list(gen_M.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    L1 = nn.L1Loss()
    mse = nn.MSELoss()

    # Used to load a models weights. 
    if config.LOAD_MODEL:
        load_checkpoint(config.CHECKPOINT_GEN_M, gen_M, opt_gen, config.LEARNING_RATE)
        load_checkpoint(config.CHECKPOINT_GEN_F, gen_F, opt_gen, config.LEARNING_RATE)
        load_checkpoint(config.CHECKPOINT_CRITIC_M, disc_M, opt_disc, config.LEARNING_RATE)
        load_checkpoint(config.CHECKPOINT_CRITIC_F, disc_F, opt_disc, config.LEARNING_RATE)

    
    # Ensure paths exist before creating dataset
    
    config.validate_path() 
    
    # Creates training Dataset.
    dataset = MaleFemaleDataset(
        root_male=os.path.join(config.TRAIN_DIR, "Male"),
        root_female=os.path.join(config.TRAIN_DIR, "Female"),
        transform=config.transforms,
    )
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
        train_fn(disc_M, disc_F, gen_F, gen_M, loader, opt_disc, opt_gen, L1, mse, d_scaler, g_scaler,)
        
        
        # Saves the model's weight every time epoch is fully ran
        if config.SAVE_MODEL:

            save_checkpoint(gen_M, opt_gen, filename=os.path.join(config.CHECKPOINT_DIR,config.CHECKPOINT_GEN_M))
            save_checkpoint(gen_F, opt_gen, filename=os.path.join(config.CHECKPOINT_DIR,config.CHECKPOINT_GEN_F))
            save_checkpoint(disc_M, opt_disc, filename=os.path.join(config.CHECKPOINT_DIR,config.CHECKPOINT_CRITIC_M))
            save_checkpoint(disc_F, opt_disc, filename=os.path.join(config.CHECKPOINT_DIR,config.CHECKPOINT_CRITIC_F))

if __name__ =="__main__":

    main()