import cv2
import urllib.request
import numpy as np
import os



import ultralytics
#ultralytics.checks()

from ultralytics import YOLO
#model = YOLO("yolo11s.pt")

#model.train(data='./datasets/pen/data.yaml', epochs=3)  # train the model

model = YOLO("./runs/detect/train6/weights/best.pt")

import numpy as np
import matplotlib.pyplot as plt

# Perform inference
#results = model('./pen_test.jpg')

# Access the plotted image
#plotted_image = results[0].plot()

# Display the image using matplotlib
# plt.clf()
# plt.imshow(plotted_image)
# plt.axis('off')  # Hide axes
# plt.show()

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    # Detect Objects
    #implement name, x1, y2, x2, y2
    results = model(frame)
    #print(results)

    for result in results:
        #print(result.boxes)
        for box in result.boxes:
            coordinates = (box.xyxy).tolist()[0] #get coordinates
            print(coordinates)
            left, top, right, bottom = coordinates[0], coordinates[1], coordinates[2], coordinates[3]
            print(left, top)
            print(right, bottom)
            cv2.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), (36,255,12), 3) #draw the bounding box around each object
            cv2.rectangle(frame, (0,0), (200,200), (36,255,12), 3)
            cv2.putText(frame, f'Object {int(box.conf*100)}%', (int(left), int(top)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2) #put text above each object


    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
