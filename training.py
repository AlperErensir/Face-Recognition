# Import libraries.

import sys 
sys.path.append('/usr/local/lib/python3.9/site-packages')

import cv2
import os
import numpy as np
from PIL import Image
import pathlib


#Check the existence of directory path.
def assure_path_exists(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)

# Use the Local Binary Patterns Histograms for face recognization 
recognizer = cv2.face.LBPHFaceRecognizer_create()

# For detecting the faces in each frame we will use Haarcascade Frontal Face default classifier of OpenCV
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml");

# Get the images and label data

def getImagesAndLabels(path):

    # Getting all file paths
    imagePaths = [os.path.join(path,f) for f in os.listdir(path)] 
    
    # Empty face sample initialised.
    faceSamples=[]
    
    # Set the ID 
    ids = []

    # Looping through all the file path.
    for imagePath in imagePaths:

        # Convert image to grayscale.
        PIL_img = Image.open(imagePath).convert('L')

        # Convert PIL image to numpy array using array() method of numpy.
        img_numpy = np.array(PIL_img,'uint8')

        # Get the image ID.
        id = int(os.path.split(imagePath)[-1].split(".")[1])

        # Get the face from the training images.
        faces = detector.detectMultiScale(img_numpy)

        # Loop for each face and appending it to their respective IDs
        for (x,y,w,h) in faces:

            # Add the image to face samples.
            faceSamples.append(img_numpy[y:y+h,x:x+w])

            # Add the ID to IDs.
            ids.append(id)

    # Pass the face array and IDs array.
    return faceSamples,ids

# Get the faces and IDs.
faces,ids = getImagesAndLabels('training_data')

# Train the model using the faces and IDs.
recognizer.train(faces, np.array(ids))

# Save the model into s_model.yml.
assure_path_exists('saved_model/')
recognizer.write('saved_model/s_model.yml')