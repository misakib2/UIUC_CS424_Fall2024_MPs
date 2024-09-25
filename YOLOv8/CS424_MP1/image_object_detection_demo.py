# UIUC CS 424 Machine Problems 
# Designed and developed by Md Iftekharul Islam Sakib, Ph.D. Student, CS, UIUC (miisakib@gmail.com) under the supervision of Prof. Tarek Abdelzaher, CS, UIUC (zaher@illinois.edu)

import cv2
from ultralytics import YOLO
from PIL import Image

imageReadRelativeDir = "../../dataset/"
imageName = "frame_camera_0.png"
imageSaveRelativeDir = "../../dataset/object_detection_history/"

inferenceModel = YOLO("yolov8n.pt")

# Load the image
currentImage = cv2.imread(imageReadRelativeDir+imageName)

# Predict the bounding boxes
predictionResults = inferenceModel.predict(currentImage)
currentImageWithPredictionResultsArray=predictionResults[0].plot(conf=True, img=currentImage)
currentImageWithPredictionResults = Image.fromarray(currentImageWithPredictionResultsArray[..., ::-1])  # RGB PIL image
currentImageWithPredictionResults.show()  # show image
currentImageWithPredictionResults.save(imageSaveRelativeDir+imageName)
