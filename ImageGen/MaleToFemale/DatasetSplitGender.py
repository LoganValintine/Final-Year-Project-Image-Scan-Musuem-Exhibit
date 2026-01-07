"""
Used to split 'CelebA' dataset into male/female

Version 1.0

Programmed by lov2

"""
import pandas as pd
import os
import shutil

# File paths

AttributeList = "RequiredFiles/list_attr_celeba.csv"
Dataset = "img_align_celeba/img_align_celeba"



MaleDir = "archive/MaleFemale/Male"
FemDir = "archive/MaleFemale/Female"



# Used for making small datasets.

male_count = 0
female_count = 0



df = pd.read_csv(AttributeList, delimiter=',')
print(df.columns) 


# Finds the image name and their corresponding value
image_names = df['image_id'].values
gender_labels = df['Young'].values

# Adds the image name and gender value together for image path
for image_name, gender in zip(image_names, gender_labels):
    image_path = os.path.join(Dataset, image_name)
    

    # Adds image to Directory based on gender. 1 for male -1 for female
    if gender == 1:
        target_folder = MaleDir
    #    male_count+=1
    elif gender == -1:
        target_folder = FemDir
   #     female_count+=1
    else:
        continue
        
    
    # Moves the image from the orignal dataset instead of copying
    shutil.move(image_path, os.path.join(target_folder, image_name))
    
    
print("Loop completed.")
