"""
The UI. Handles the entire ui via usage of .ui files

WARNING : THIS FILE WILL NOT WORK CORRECTLY ON PYCHARM/Other IDE SYSTEMS, TO RUN GO TO 'edit configuration' AND MAKE 'lov2majorproject' THE CURRENT WORKING DIRECTORY!

class MainWindow - Handles the different ui pages via a stacked widget, adds them all to that and is the function behind the 'go home' button

class selectionDialogAge - A dialog pop up for users to select which model they want to use

class selectionDialogGender - A dialog pop for users to select which model they want to use

class Welcome_screen - A Widget class for the welcome screen page, contains the dialog buttons for selecting a model, and then loads the webcam

class Webcam_screen -  A Widget class that loads the webcam live feed. Contains a button to take a photo but uses OpenCV to detect when a face is on screen so the button can appear

class Loading_screen - A Widget class that runs the chosen model's image transformation. Lets users press a button that makes the image appear for them to showcase what the image transformation is.



Version 3.0

Programmed by Lov2

"""
import time
import os
import cv2 as cv
import sys
import numpy as np


from PyQt6 import uic
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QStackedWidget, QDialog, QSizePolicy
from PyQt6.QtCore import pyqtSlot, QTimer
from PyQt6.QtGui import QImage , QPixmap, QIcon
from PIL import Image, ImageQt

sys.path.append("ImageGen")
import config
import GenerationModel

# For choosing the model.
MODEL = ""

# File name path

IMGPATH =  f"Documentation/Temp.jpg"

# Pop up dialog to choose Age model
class SelectionDialogAge(QDialog):
    def __init__(self):
        super().__init__()
        uic.loadUi("Ui/age_dialog.ui", self)
        self.youngbt.clicked.connect(self.on_youngbt_click)
        self.oldbt.clicked.connect(self.on_oldbt_click)
        
        
    # Chooses young model    
    def on_youngbt_click(self):
        
        self.selected = "young"
        self.accept()
        
    # Chooses old model   
    def on_oldbt_click(self):
        
        self.selected = "old"
        
        self.accept()


# Pop up dialog to choose gender model
class SelectionDialogGender(QDialog):
    def __init__(self):
        super().__init__()
        uic.loadUi("Ui/gender_dialog.ui", self)
        self.fembt.clicked.connect(self.on_fembt_click)
        self.mascbt.clicked.connect(self.on_mascbt_click)

    # Chooses fem model
    def on_fembt_click(self):
        
        self.selected = "fem"
        
        self.accept()
        
    # Chooses masc model
    def on_mascbt_click(self):
        self.selected = "masc"
        
        self.accept()

""" Welcome screen, first screen users see """
class WelcomeScreen(QWidget):
    def __init__(self, stacked_widget):
        super().__init__()
        uic.loadUi("Ui/Welcome_screen.ui", self)
        self.setFixedSize(1100, 800)
        self.stacked_widget = stacked_widget
        
        
        
        
        self.Age_button.clicked.connect(self.on_Age_click)
        self.Gender_button.clicked.connect(self.on_Gender_click)
    
    
    # Will handle click. Checks which age model to use   
    def on_Age_click(self):
        global MODEL
        dialog = SelectionDialogAge()
        if dialog.exec():
            MODEL = dialog.selected
            
            # Changes screen to webcam
            self.stacked_widget.setCurrentIndex(1)

    # Will handle click. Checks which gender model to use
    def on_Gender_click(self):
        global MODEL
        dialog = SelectionDialogGender()
        if dialog.exec():
            MODEL = dialog.selected
            self.stacked_widget.setCurrentIndex(1)



""" Webcam screen. Handles taking a photo via the webcam""" 
class WebcamScreen(QWidget):
    def __init__(self, stacked_widget):
        super().__init__()
        self.WebcamUi = uic.loadUi("Ui/webcam.ui", self)
        self.cascadePath = "RequiredFiles/haarcascade_frontalface_default(1).xml"
        
        
       
        
        # Takes a photo/connects to button
        self.TakePhoto.clicked.connect(self.capture_photo)
        # Hides it
        self.TakePhoto.hide()
        
        self.stacked_widget = stacked_widget
        
        # Opens cap for video
        self.cap = cv.VideoCapture(0)
        self.cascade = cv.CascadeClassifier(self.cascadePath)
        
        # Starts a timer for the video. 
        self.timer = QTimer()
        self.timer.timeout.connect(self.camera_frame)
        self.timer.start(30)
        
    
        self.current_frame = None
    
    
   
        
    # Runs the camera, uses a face detection system to detect faces on live feed.
    def camera_frame(self):
        ret, frame = self.cap.read()
        
        
        
        if ret:
            self.current_frame = frame.copy()
            # Detects faces.abs
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            faces = self.cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=4, minSize=(30, 30), flags=cv.CASCADE_SCALE_IMAGE)
            
            # If theres a face detected. Shows button
            if len(faces) > 0:
                self.face_box = max (faces, key = lambda b: b[2] * b[3]) # Finds the biggest face
                self.TakePhoto.show()

            else:
                self.TakePhoto.hide()
                self.face_box = None
                
            # Draws boxes over detected faces  
            # Debug line  
           # for (x, y, w, h) in faces:
          #      cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Setting up the camera. 
            frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            height, width, channel = frame.shape
            bytes_per_line = 3 * width
            qimg = QImage(frame.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
            
            # Creating an image map for the camera
            pixmap = QPixmap.fromImage(qimg)
            self.WebcamUi.Webcam.setPixmap(pixmap)
            
            self.WebcamUi.Webcam.setScaledContents(True)
            
            self.WebcamUi.Webcam.setFixedSize(800, 600)
        
    # Captures a photo once the button is pressed.    
    def capture_photo(self):
        
        if self.current_frame is not None:
            frame_rgb = cv.cvtColor(self.current_frame, cv.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)
            
            
            if self.face_box is not None:
                cropped_img = config.crop_image(pil_img, self.face_box)
            
        
            cropped_img.save(IMGPATH, quality = 95)
            
           
            self.WebcamUi.Text.setText("Photo saved!")
            QApplication.processEvents()
            self.stacked_widget.setCurrentIndex(2)
                
            loading_screen = self.stacked_widget.currentWidget()
            loading_screen.SetDisplayPhoto(pil_img)

                
            
"""Is displayed to the user whilst the image is being translated. """           
class LoadingScreen(QWidget):
    def __init__(self, stacked_widget):
        super().__init__()
        self.loading_screenUI = uic.loadUi("Ui/Loading_screen.ui", self)
        self.transformation.clicked.connect(self.GeneratePhoto)
        
     # Sets the inputted image as the display image   
    def SetDisplayPhoto(self, pil_image):
        
        displayImg = np.array(pil_image)
        height, width, channel = displayImg.shape
        bytes_per_line = channel * width
        qimg = QImage(displayImg.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        
        self.loading_screenUI.ImgLabel.setPixmap(pixmap)
        self.loading_screenUI.ImgLabel.setScaledContents(True)
        
    # Turns the image into a QImage for pixmap.
    def GeneratePhoto(self):
        
        img = GenerationModel.main(MODEL,IMGPATH)
        qimg = ImageQt.ImageQt(img)
        pixmap2 = QPixmap.fromImage(qimg)
        
        
        pixmap2 = QPixmap.fromImage(qimg)
        
        self.loading_screenUI.ImgLabel.setPixmap(pixmap2)
        self.loading_screenUI.ImgLabel.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        
        
        
   
        
"""main window class, creates stacked_widget and adds all screens to it. """       
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        title = "Face Transformation Lov2"
        icon = "Documentation\images\Icon.png"
        
        # Creates a stacked widget to add all screens onto.abs
        self.stacked_widget = QStackedWidget()
        self.setCentralWidget(self.stacked_widget)
        
        
        
        # Creates the welcome, webcam and loading screen.
        self.welcome_screen = WelcomeScreen(self.stacked_widget)
        self.webcam_screen = WebcamScreen(self.stacked_widget)
        self.loading_screen = LoadingScreen(self.stacked_widget)
        
        
        # Set buttons
        
        self.webcam_screen.home.clicked.connect(self.goBack)
        self.loading_screen.home.clicked.connect(self.goBack)
        
        # Adds them to the stacked widget.abs
        self.stacked_widget.addWidget(self.welcome_screen)
        self.stacked_widget.addWidget(self.webcam_screen)
        self.stacked_widget.addWidget(self.loading_screen)

        self.setWindowIcon(QIcon(icon))
        self.setWindowTitle(title)
        self.stacked_widget.setCurrentIndex(0)

    
    def goBack(self):
        
        self.stacked_widget.setCurrentIndex(0)
        self.webcam_screen.Text.setText("Please hold still whilst displaying your face to take a photo")
    
    
    # Once closed removes the img file.abs
    def closeEvent(self, event):
        os.remove(IMGPATH)   
        
def main():
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec()