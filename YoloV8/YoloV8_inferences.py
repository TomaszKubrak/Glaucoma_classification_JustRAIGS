# Creating more accurate masks for further training of YoloV8

# This file was adapted from Computer vision engineer's Image Segmentation YoloV8 Collection
# Source: https://github.com/computervisioneng/image-segmentation-yolov8
# The original source is licensed under the AGPL-3.0 license.

import os
from ultralytics import YOLO
import cv2

##### Adjust this section as needed #####
model_path = './YoloV8.py'
image_path = './images'
destination_path = './masks'
#########################################

# Check if destination path exists, if not, create it
if not os.path.exists(destination_path):
    os.makedirs(destination_path)

missing_masks= []

# Load the model
model = YOLO(model_path)

# Loop through all files in the image_path
for filename in os.listdir(image_path):
      file_path = os.path.join(image_path, filename)

      # Read the image
      img = cv2.imread(file_path)
      H, W, _ = img.shape

      # Make predictions
      results = model(img)

      for result in results:
        if result.masks is not None:
            for j, mask in enumerate(result.masks.data):
              mask = mask.cpu().numpy() * 255
              mask = cv2.resize(mask, (W, H))

              # Save the mask, using the original filename in the destination_path
              output_path = os.path.join(destination_path, filename)
              cv2.imwrite(output_path, mask)
        else:
          missing_masks.append(filename)
          print(f"No detections for {filename}")

print("Processing complete.")
print(len(missing_masks))