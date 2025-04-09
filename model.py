import ultralytics
from ultralytics import YOLO

#check ultralytics install
ultralytics.checks()

#Model training
model = YOLO("yolo11s.pt")
dataset_name='pen'
model.train(data=f'./datasets/{dataset_name}/data.yaml', epochs=3)  # train the model