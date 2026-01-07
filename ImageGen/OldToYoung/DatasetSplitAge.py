"""
Used to split 'CelebA' dataset into young/old

Version 1.0

Programmed by lov2

"""
import pandas as pd
import os
import shutil

# File paths

AttributeList = "RequiredFiles/list_attr_celeba.csv"
Dataset = "img_align_celeba/img_align_celeba"


# Change to youngOld
YoungDir = "ImageGen/archive\YoungOld\Young"
OldDir = "ImageGen/archive\YoungOld\Old"


df = pd.read_csv(AttributeList, delimiter=',')
print(df.columns) 


# Finds the image name and their corresponding value
image_names = df['image_id'].values
age_labels = df['Young'].values

# Adds the image name and age value together for image path
for image_name, age in zip(image_names, age_labels):
    image_path = os.path.join(Dataset, image_name)
    

    # Adds image to Directory based on age. 1 for young -1 for old
    if age == 1:
        target_folder = YoungDir

    elif age == -1:
        target_folder = OldDir
    else:
        continue
        
    
    # Moves the image from the orignal dataset instead of copying
    shutil.move(image_path, os.path.join(target_folder, image_name))
    
    
print("Loop completed.")
