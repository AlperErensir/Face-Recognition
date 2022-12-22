# Import libraries.

import cv2 
import numpy as np
import os 

# Check the existence of directory path.
def assure_path_exists(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)

# Create Local Binary Patterns Histograms for face recognization.
recognizer = cv2.face.LBPHFaceRecognizer_create()

assure_path_exists("saved_model/")

# Load the saved previously trained mode.
recognizer.read('saved_model/s_model.yml')

# Load prebuilt classifier for Frontal Face detection by using haar method.
cascadePath = "haarcascade_frontalface_default.xml"

# Create classifier from prebuilt model.
faceCascade = cv2.CascadeClassifier(cascadePath);

# Font style.
font = cv2.FONT_HERSHEY_SIMPLEX

# Initialize and start the video frame capture from the webcam.
cam = cv2.VideoCapture(0)

while True:
    # Read the video frame
    ret, im =cam.read()

    # Convert the captured frame into grayscale
    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)

    # Getting all faces from the video frame
    faces = faceCascade.detectMultiScale(gray, 1.2,5) #default

    # Detect the faces in the frame, start predicting by using pre trained data.
    for(x,y,w,h) in faces:

        # Create a rectangle around the face
        cv2.rectangle(im, (x-20,y-20), (x+w+20,y+h+20), (0,255,0), 4)

        # Recognize the face belongs to which ID
        Id, confidence = recognizer.predict(gray[y:y+h,x:x+w])  #Our trained model is working here

        # Set the name according to ID.
        if Id == 1:
            if confidence < 80:
                Id = "Alper {0:.2f}%".format(round(100 - confidence, 2))
                cv2.rectangle(im, (x-22,y-90), (x+w+22, y-22), (0,255,0), -1)
                cv2.putText(im, str(Id), (x,y-40), font, 1, (255,255,255), 3)
            else:
                Id = "Not Alper {0:.2f}%".format(round(confidence, 2))
                cv2.rectangle(im, (x-22,y-90), (x+w+22, y-22), (0,0,255), -1)
                cv2.putText(im, str(Id), (x,y-40), font, 1, (255,255,255), 3)
                

    # Display the video frame with the bounded rectangle.
    cv2.imshow('im',im) 

    # Press q to close the program.
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Terminate video.
cam.release()

# Close all windows.
cv2.destroyAllWindows()