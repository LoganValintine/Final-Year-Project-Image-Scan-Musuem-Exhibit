"""

Dataset class for holding the datasets. This one is used for old -> young


class oldyoungDataset - Gathers the images for the dataset, finds the largest dataset for training
and then turns images from said dataset into tensor images via augmentations and transforms

Version 1.0

Programmed by Lov2


"""
import numpy as np

import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image


# Creates the dataset, using root_young and root_old from the train class.
class OldYoungDataset(Dataset):
    def __init__(self, root_young, root_old, transform=None):
        self.root_young = root_young
        self.root_old = root_old
        self.transform = transform
        print(f"Checking {root_young}")
        print(f"Checking {root_old}")
        
        # In case of dataset crash from large datasets.
        try:
          
          self.young_images = os.listdir(root_young)
          self.old_images = os.listdir(root_old)
          
        # Error handling, if it crashes out will save whatever images were found at that moment.  
        except Exception as e:
          print("Image Request timed out: {e}")
          self.young_images = []
          self.old_images = []
          # Debug statement to find length of both datasets
        #print(len(self.young_images), " and " ,len(self.old_images))
        
        # Biggest dataset is used for training length per epoch.
        self.length_dataset = max(len(self.young_images), len(self.old_images)) 
        self.young_len = len(self.young_images)
        self.old_len = len(self.old_images)
        
    # Creates dataset    
    def __len__(self):
        return self.length_dataset
    
    # Gets a img from the datasets and converts them to an rbg image in case.
    def __getitem__(self, index):
        young_img = self.young_images[index % self.young_len]
        old_img = self.old_images[index % self.old_len]
        
        young_path = os.path.join(self.root_young, young_img)
        old_path = os.path.join(self.root_old, old_img)
        
        young_img = np.array(Image.open(young_path).convert("RGB"))
        old_img = np.array(Image.open(old_path).convert("RGB"))
        
        # Transforms the image to be suitable for cylceGan
        if self.transform:
            augmentations = self.transform(image=young_img, image0=old_img)
            young_img = augmentations["image"]
            old_img = augmentations["image0"]
            
        return young_img, old_img
    
    
