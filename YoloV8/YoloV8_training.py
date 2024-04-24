#YoloV8 training

# This file was adapted from Computer vision engineer's Image Segmentation YoloV8 Collection
# Source: https://github.com/computervisioneng/image-segmentation-yolov8
# The original source is licensed under the AGPL-3.0 license.

from ultralytics import YOLO

#Change paths as needed

DATA_DIR = './data/'

model = YOLO('./yolov8x-seg.pt')  

model.train(data='./config.yaml', epochs=100, imgsz=640)