# Lov2 Major project : Face transformation museum exhibit 

## Description

This is my CS39440 Major project: This project is based on the 'Face transformation' musuem exhibit, it will be developed in Python and utilise CycleGan Model's for development as well as PyQt6 for the interface. The end result will be software capable of taking in static images of faces and transform them based on differing age and gender. 

## Introduction
This is a project based on a face-transformation exhibit in London Science Musuem, the exhibit took place in 2004 and was based on transforming the users face based on their selections. 

## CycleGan Explained
CycleGan is an unpaired image to image translation. It uses two sets of images and a pair of generator and discriminators to create fake images via downsampling and upsampling the fake image. The discriminator is then used predict whether these images are fake. In the end fake images are generated. This is explained more in a link below.
https://junyanz.github.io/CycleGAN/


## Author and acknowledgement

Lov2 - Developer
Bernie Tiddeman - Supervisor. 



## Dataset credit 

https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html - Celeb A Dataset. Featuring 200,000 images of celebrities. 

Author : Ziwei Liu, Ping Luo, Xiaogang Wang, Xiaoou Tang




## Installation

Please follow the following instructions for install:


### Requirements:
- Python Version: 3.12.12
- Recommended IDE: Visual Studio Code (Not mandatory, but recommended)

### Installation:
To install all required dependencies, run the following command in your terminal
Please note this must be run in this folder for this to work!:

```
pip install -r Requirements.txt
```
Note: If the above command fails, manually copy and paste the contents of Pipinstall.txt into your terminal. It includes both installs in terminal and in Jupyter notebook

Open your ide and the project files inside the ide. Make sure there is a python/a virtual environment set to python 3.12.12 or equivalent


### WARNING:

VSC is HIGHLY recommended for this, however if using PyCharm, please do the following -

run -> edit configuration -> select TrainingModel as the working directory and then

select MaleToFemaleTrain.py or OldToYoungTrain.py for script configuration

IF using VSC you do not have to do this!

### Training the Models:

IF YOU NEED TO TRAIN THE MODELS PLEASE DOWNLOAD THE DATASET LINKED ABOVE AND READ BELOW:

1) After downloading the dataset, place the 'CelebA' into the top level of the project

2) Depending on model, choose the corresponding dataset splitting file

3) If you want to do the same for the other, duplicate 'CelebA'

4) The splitted datasets will be in archive/OldYoung or archive/MaleFemale

Training the Gender Model:
Run the following script:

python MaleToFemaleTrain.py

Training the Age Model:
Run the following script:

python OldToYoungTrain.py

Important Notes:
- The models are set to run for 100 epochs (training cycles).
- Checkpoint weights are saved in the Checkpoints folder:
  - Gender Model Weights: GenM/GenF
  - Age Model Weights: GenO/GenY
- Checkpoints are saved every 50,000 batches (e.g., 0, 50,000, 100,000, etc.).
- Checkpoints can be saved via using Config.py and setting SAVE_MODEL to True, the same can be done for LOAD_MODEL.
- Training will take a long time! You can stop it if needed, as long as checkpoints are saved.


### Running the file

Please run the following file:

ImageTransformation.py

This will run the UI and by extension the generation model used. 




## License
Licensed under Aberystwyth University. 

## Project status
In Full release

Version 4.0
