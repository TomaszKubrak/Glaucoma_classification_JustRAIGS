# This code was adapted from: https://github.com/Paresh-shahare/Optic-Disc-Segmentation
# It was used to obtaine first masks since JustRAIGS dataset doesn't provide masks of optic discs

import os
import cv2
from keras.models import load_model
import numpy as np

##### Adjust this section as needed #####
model_path = './UNetW_final.h5'
Training_data = r'./images/'
save_dir = r'./masks/'
#########################################

# Create save directory if it doesn't exist
if not os.path.exists(save_dir):
    os.makedirs(save_dir)  

model = load_model(model_path)
width = 512
height = 512

# Prepearing training data
training_data = [x for x in sorted(os.listdir(Training_data))]
x_train_data = np.empty((len(training_data),width,height),dtype = 'float32')
x_train_data = x_train_data.reshape(x_train_data.shape[0],width,height,1)

# Running inferences 
result = model.predict(x_train_data,verbose=1)
result = (result > 0.5).astype(np.uint8)

for i, (mask, image_filename) in enumerate(zip(result, training_data)):
    # Extraction of Image Basename:
    image_base = os.path.basename(image_filename)

    # Convert mask to 0-255 scale and reshape if needed
    mask_scaled = (mask * 255).astype(np.uint8)
    mask_2d = np.reshape(mask_scaled, (width, height))

    # Construct save path with matching image name
    save_path = os.path.join(save_dir, image_base)

    # Save the mask image
    cv2.imwrite(save_path, mask_2d)

print(f"Saved {len(result)} masks to {save_dir}")