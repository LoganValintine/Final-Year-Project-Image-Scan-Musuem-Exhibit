"""

Dataset class for holding the datasets. This one is used for Male -> Female


class MaleFemaleDataset - Gathers the images for the dataset, finds the largest dataset for training
and then turns images from said dataset into tensor images via augmentations and transforms



Version 1.0

Programmed by Lov2



"""
from PIL import Image
import os
from torch.utils.data import Dataset, DataLoader
import numpy as np


# Creates the dataset, using root_female and root_male from the train class.
class MaleFemaleDataset(Dataset):
    def __init__(self, root_female, root_male, transform=None):
        self.root_female = root_female
        self.root_male = root_male
        self.transform = transform
        print(f"Checking {root_female}")
        print(f"Checking {root_male}")
        
        # In case of dataset crash from large datasets.
        try:
          
          self.female_images = os.listdir(root_female)
          self.male_images = os.listdir(root_male)
          
        # Error handling, if it crashes out will save whatever images were found at that moment.  
        except Exception as e:
          print("Image Request timed out: {e}")
          self.female_images = []
          self.male_images = []
          # Debug statement to find length of both datasets
        #print(len(self.female_images), " and " ,len(self.male_images))
        
        # Biggest dataset is used for training length per epoch.
        self.length_dataset = max(len(self.female_images), len(self.male_images)) 
        self.female_len = len(self.female_images)
        self.male_len = len(self.male_images)
        
    # Creates dataset    
    def __len__(self):
        return self.length_dataset
    
    # Gets a img from the datasets and converts them to an rbg image in case.
    def __getitem__(self, index):
        female_img = self.female_images[index % self.female_len]
        male_img = self.male_images[index % self.male_len]
        
        female_path = os.path.join(self.root_female, female_img)
        male_path = os.path.join(self.root_male, male_img)
        
        female_img = np.array(Image.open(female_path).convert("RGB"))
        male_img = np.array(Image.open(male_path).convert("RGB"))
        
        # Transforms the image to be suitable for cylceGan
        if self.transform:
            augmentations = self.transform(image=female_img, image0=male_img)
            female_img = augmentations["image"]
            male_img = augmentations["image0"]
            
        return female_img, male_img