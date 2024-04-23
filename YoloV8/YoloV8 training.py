#YoloV8 training
from ultralytics import YOLO

DATA_DIR = './data/'

model = YOLO('./yolov8x-seg.pt')  # load a pretrained model

model.train(data='./config.yaml', epochs=100, imgsz=640)