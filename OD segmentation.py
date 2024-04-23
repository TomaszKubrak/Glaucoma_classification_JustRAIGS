import cv2
import pandas as pd
import torch
import numpy as np
from ultralytics import YOLO
import os

def find_encompassing_bbox(bboxes):
    """
    Find the encompassing bounding box from multiple bounding boxes.

    Parameters:
    - bboxes: A list of bounding boxes, each in the format [x1, y1, x2, y2].

    Returns:
    - The encompassing bounding box as [min_x1, min_y1, max_x2, max_y2].
    """
    # Initialize min and max coordinates with the first bounding box
    min_x1, min_y1, max_x2, max_y2 = bboxes[0]

    for bbox in bboxes[1:]:
        x1, y1, x2, y2 = bbox
        min_x1 = min(min_x1, x1)
        min_y1 = min(min_y1, y1)
        max_x2 = max(max_x2, x2)
        max_y2 = max(max_y2, y2)

    return [min_x1, min_y1, max_x2, max_y2]

# Function to crop and resize an image
def crop_and_resize_image(img, bbox, target_size=(518, 518)):
    """
    Attempts to crop a 518x518 square from the center of a given bounding box. If the square
    exceeds image dimensions, it crops the largest possible square and resizes it to 518x518.
    If a given bounding box exceeds 518x518, segmentation is considered as invalid.

    Parameters:
    - img: The preprocessed image read by CV2.
    - bbox: The bounding box coordinates as [x1, y1, x2, y2].
    - target_size: The target size for cropping and resizing as (width, height).

    Returns:
    - Resized image if cropping is possible, otherwise None.
    """
    x1, y1, x2, y2 = map(int, bbox)
    
    # Calculate the width and height of the bounding box
    bbox_width = x2 - x1
    bbox_height = y2 - y1

    # Check if the bounding box doesn't exceed 518x518
    if bbox_width > target_size[0] or bbox_height > target_size[1]:
        return None 

    # Calculate the center of the bounding box
    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2

    # Calculate half the target size
    half_target = target_size[0] // 2

    # Define initial maximum square crop that can fit within the image boundaries
    start_x = max(0, center_x - half_target)
    start_y = max(0, center_y - half_target)
    end_x = min(img.shape[1], center_x + half_target)
    end_y = min(img.shape[0], center_y + half_target)

    # Validate crop dimensions
    crop_width = end_x - start_x
    crop_height = end_y - start_y

    # Adjust crop dimensions to the largest possible square within the boundary
    if crop_width < target_size[0] or crop_height < target_size[1]:
        # Calculate the largest possible dimension that can be squared within the limits
        max_possible_square = min(crop_width, crop_height)
        start_x = center_x - max_possible_square // 2
        start_y = center_y - max_possible_square // 2
        end_x = start_x + max_possible_square
        end_y = start_y + max_possible_square
        # Re-validate boundaries (important in cases where center is near the image edge)
        start_x = max(0, start_x)
        start_y = max(0, start_y)
        end_x = min(img.shape[1], end_x)
        end_y = min(img.shape[0], end_y)

    # Crop the image
    cropped_img = img[start_y:end_y, start_x:end_x]
    if cropped_img.size == 0:
        return None  # Return None if the cropped image is empty

    # Resize to the desired target size
    resized_img = cv2.resize(cropped_img, target_size, interpolation=cv2.INTER_CUBIC)

    return resized_img

# Function to determine path of an image
def determine_path(base_path, eye_id):
  base_path = os.path.join(base_path, eye_id)
  for ext in ['.JPG', '.PNG','.JPEG']:
      full_path = base_path + ext
      if os.path.exists(full_path):
          return full_path

# Function to apply ROI on all datasets
def process_images(dataset, model, base_path_img, save_path, size=(518, 518)):
    """
    Processes and saves images with successful ROI segmentation .

    Iterates through the dataset, checks if the image already exists in the save path, and if so, classifies as OD detected.
    If not, detects OD using YOLO, and if found, validate bbox size, crops, resizes, and saves the image.
    Updates the dataset with ROI detection results.

    Parameters:
        dataset (DataFrame): Dataset with 'Eye ID' for image processing.
        model (YOLO): Pretrained YOLO model for OD detection.
        base_path_img (str): Path to the directory containing images.
        save_path (str): Path to save processed images.

    Returns:
        Updated dataset.
    """
   
    od_detected = []
    for index, row in dataset.iterrows():
        eye_id = row['Eye ID']
        output_filename = f"{eye_id}.jpg"
        output_path = os.path.join(save_path, output_filename)

        # Check if the output file already exists
        if os.path.exists(output_path):
            print(f"Image for {eye_id} already processed.")
            od_detected.append(1)
            continue

        # Determine image path
        img_path = determine_path(base_path_img, eye_id)
        if not img_path:
            print(f"Path for {eye_id} not found.")
            od_detected.append(0)
            continue

        # Load the preprocessed image
        img_array = cv2.imread(img_path)

        # Make predictions using YOLO model
        results = model(img_array)
        boxes = results[0].boxes

        # Checking if optic disc was detected
        if len(boxes) == 0:
            od_detected.append(0)
        else:
            # Find the encompassing bounding box, crop, resize, and save
            bboxes = [box.numpy().tolist() for box in boxes.xyxy]
            encompassing_bbox = find_encompassing_bbox(bboxes)
            cropped_resized_img = crop_and_resize_image(img_array, encompassing_bbox, target_size=size)
            if cropped_resized_img is None:
                print("BBox was too large")
                od_detected.append(0)
            else:
                cv2.imwrite(output_path, cropped_resized_img)
                od_detected.append(1)

    # Update the DataFrame with the OD detection results
    dataset['OD'] = od_detected
    return dataset

def main():
    # Initialize YOLO model
    model_path = './YoloV8.pt'
    model = YOLO(model_path)

    file_path_img = './preprocessed_img/'
    save_path = './ROI_images/'

    # Base path for datasets
    dataset_base_path = './Datasets/'

    # List of CSV files to read from
    input_files = [
        '10_features_no_mask_test.csv',
        '10_features_no_mask_train.csv',
        'glaucoma_no_mask_test.csv',
        'glaucoma_no_mask_train.csv'
    ]

    # Corresponding output file names
    output_files = [
        '10_features_masks_test.csv',
        '10_features_masks_train.csv',
        'glaucoma_masks_test.csv',
        'glaucoma_masks_train.csv'
    ]

    # Loop over the file lists, combining with the base path
    for input_file, output_file in zip(input_files, output_files):
        input_csv = f"{dataset_base_path}{input_file}"
        output_csv = f"{dataset_base_path}{output_file}"
        
        df_temp = pd.read_csv(input_csv)  # Read the DataFrame
        processed_dataset = process_images(df_temp, model, file_path_img, save_path)  # Process images
        segmentation_dataset = processed_dataset[processed_dataset["OD"] == 1]  # Filter rows where OD was detected
        segmentation_dataset.drop("OD", axis=1, inplace=True)  # Drop the 'OD' column
        segmentation_dataset.reset_index(drop=True, inplace=True)  # Reset the DataFrame index
        segmentation_dataset.to_csv(output_csv, index=False)  # Save the processed DataFrame

if __name__ == '__main__':
    main()