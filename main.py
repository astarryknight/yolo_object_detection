#imports
import cv2
import urllib.request
import numpy as np
import os
import matplotlib.pyplot as plt
import ultralytics
from ultralytics import YOLO

#after training, pick the best weights from training to use for inference
model = YOLO("./runs/detect/train6/weights/best.pt")

#setting video capture to the webcam
cap = cv2.VideoCapture(0)

while True:
    #get the current frame from the webcam
    ret, frame = cap.read()

    # detect objects (model inference)
    results = model(frame)

    #loop through all of the detected objects
    for result in results:
        for box in result.boxes: #access the boxes object for each detection
            coordinates = (box.xyxy).tolist()[0] #get coordinates
            left, top, right, bottom = coordinates[0], coordinates[1], coordinates[2], coordinates[3] 
            cv2.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), (36,255,12), 3) #draw the bounding box around each object
            cv2.putText(frame, f'Object {int(box.conf*100)}%', (int(left), int(top)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2) #put text above each object

    #show the annotated frame on opencv
    cv2.imshow("Frame", frame)

    #add esc key to stop the program loop
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()