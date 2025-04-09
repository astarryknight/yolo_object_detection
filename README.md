# yolo_object_detection
Live Object detection using the YOLO system + Webcam

## To use this repository:
Install the conda environment using `conda env create -f conda_env.yml`, then activate using `conda activate yolo-1`.

Run `model.py` to train the yolo model.
Run `main.py` to perform live inference using the webcam.

## Custom datasets:
Place your custom dataset in the `datasets` directory. Make sure the file structure matches the sample dataset "pen".

In `model.py`, change the `dataset_name` variable to match the name of the folder you placed in the `datasets` directory.
Then, run `model.py` to train the model and `main.py` to perform live inference using the webcam.