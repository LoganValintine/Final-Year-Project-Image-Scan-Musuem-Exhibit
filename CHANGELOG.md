All NOTABLE changes to the document will be in this file. 

## [Unreleased] - 27/01/25

### Added
- Gitlab page, started writing of project outline 


## 0.1 - 03/02/25 - In the Beginning

### Changed
- Project outline now completed.


## 0.2 - 04/02/25 - User interfacing

### Added
- Added 'UserInterface' directory. This will hold all UI info. 

- Added 'Webcam.py'. This is a test file to run a working webcam on python. 



## 0.3 - 07/02/25 - Face capturing

### Added

- Added 'Capture.py'. This is a python file that can open a webcam and save a photo, it will be used as a reference for making my own.

- Added User story issues, for tracking issues throughout major project


### Updated

Updated ReadMe and Changelog. 


## 0.4 - 15/02/25 - Fixing

### Fixed

- Fixed the 'webcam.py' file, it now properly takes an image, saves it to filepath and runs a lot smoother

### Removed

- Removed 'Capture.py' file, no longer needed. 


## 0.5 - 20/02/25 - Google docing 

### Added

- Added TrainModel.py, this will be used to access the google drive for training the model on.
07
- Added Credientals.json and token.pickle, these are secure files for accessing the google drive.
08

### Updated

- Updated ReadMe and Changelog

## 1.0 - 26/02/25 DEMO UPDATE

### Changed

- Changed Webcam.py and FaceCapture.py, now images are temporary and will be deleted after the file is closed.
- Both files now are intergrated with eachother, running seeminglessly together to detect faces from a saved photo.

### Added 

- Changed to 1.0 as it is the first demo, still not fully finished but a stable version.

- RequiredFiles directory, including the requirements.txt and haarcascade file for face detection.

## 1.1 - 02/03/25 Finally starting some image transformation


### Added 

- Added ImageGen.py and config.py, used for the training of the model later on.

- Added Blog_5 

- Added 'CelebA' database

### Removed

- UTK Database

- Credientials.json/token.pickle, as they wont (hopefully) be needed anymore. 


### 1.2 - 09/03/25 Movement to Google Colab

### Added

- Finished basic model for ImageGen.py, doesnt fully work for male -> female. 

- Added Blog_6

- Added DataSplit.py to split celebA dataset, splitting it between male to female files

- list_attr_celebA for getting CelebA's attributes. 

### Changed

- Moved files to google Colab for running cycleGan model.

- Extracted the new Male/Female split datasets of CelebA. 

### Removed

- Config.py


## 2.0  - 20/03/25 The Indenting issues. 


### Added 

- Added the following files, config.py, MFDataset.py, generator_model.py, discriminator_model.py and MaleToFemaleTrain.py. These build a cycleGAN model that can transform male to female faces and vice versa


- Added SaveUtilis.py for saving and loading model data. 


### Updated

- Updated FaceCapute.py to crop images. 

### Fixed

- Multiple issues with said files, after properly fixing indenting issues. 



## 3.0 - 24/04/25, UI rewritten 


### Added

- Added the following files, age_dialog.ui, gender_dialog.ui, loading.qrc, ui.py, webcam.ui, Welcome_screen.ui. These are the fundamentals for the new UI built in Pyqt6. This does all the features in the previous GUI but now has a welcome screen. Dialogs for model options, and a proper webcam implementation, this is now done via a live face capture and a button that can only appear once a face is detected.

- Added a old to young dataset and a male to female dataset for training the model.

- Added a video demo for the model working.

- Added more demo images showcasing the models outputs

- Added new blogs.

### Updated

- Updated previous blogs and ReadMe 

- Updated requirements.txt to match current used libaries 

### Removed

- Removed webcam.py and faceCapture.py. These were no longer needed once the new GUI was written. 



## 3.1 - 28/03/25 Age Model completed

### Added

- Added the following files, DatasetSplitAge.py, OldToYoungTrain.py, OYDataset.py. This creates and runs/trains the old to young model. Similair to how the previous MaleToFemale model works

- Added GenderModel.py, this uses the 'semi' trained model to change the way a person looks like from male to female (and vice versa!)

- Added new blogs

### Updated

- Updated UI.py to use gender model 

### Removed 

- Removed some documentation images and files, werent needed anymore. 


## 4.0 - 01/05/25 Final release

### Added 

- Added new methods in cropping the image down, changed the way face detection worked and image taking

- Added the report.

- Added the full model checkpoints for every model.

- Added ImageTransformation.py this will run the whole system

### Updated

- Updated GenderModel.py this is now GenerationModel and handles every model.

- Ensured all blogs/documentation was updated.


