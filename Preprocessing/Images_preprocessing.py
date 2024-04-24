# The function 'trim_and_resize' was adapted from Aladdin Persson's Machine Learning Collection
# Source: https://github.com/aladdinpersson/Machine-Learning-Collection/tree/master/ML/Kaggles/DiabeticRetinopathy/preprocess_images.py
# The original source is licensed under the MIT License.

import os
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

def apply_clahe(img, clip_limit=3.0, tile_grid_size=(8, 8)):
    """Applying CLAHE contrast enhancement on each color channel separately."""
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    r, g, b = cv2.split(img)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    r_clahe, g_clahe, b_clahe = clahe.apply(r), clahe.apply(g), clahe.apply(b)
    clahe_img = cv2.merge([r_clahe, g_clahe, b_clahe])
    return cv2.cvtColor(clahe_img, cv2.COLOR_BGR2RGB)

def trim_and_resize(im, output_size):
    """
    Trimming the margins of an image where the pixel intensity is near zero,
    padding black pixels to maintaine square retio of an image and resizing to output_size.
    """
    percentage = 0.02
    img = np.array(im)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    im_binary = img_gray > 0.1 * np.mean(img_gray[img_gray != 0])
    row_sums = np.sum(im_binary, axis=1)
    col_sums = np.sum(im_binary, axis=0)
    rows = np.where(row_sums > img.shape[1] * percentage)[0]
    cols = np.where(col_sums > img.shape[0] * percentage)[0]
    if rows.size and cols.size:
        min_row, min_col = np.min(rows), np.min(cols)
        max_row, max_col = np.max(rows), np.max(cols)
        img = img[min_row:max_row+1, min_col:max_col+1]
    im_pil = Image.fromarray(img)
    old_size = im_pil.size
    ratio = float(output_size) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])
    im_resized = im_pil.resize(new_size, Image.LANCZOS)
    new_im = Image.new("RGB", (output_size, output_size))
    new_im.paste(im_resized, ((output_size - new_size[0]) // 2, (output_size - new_size[1]) // 2))
    return new_im

def process_and_save_images(input_path_folder, output_path_folder, output_size):
    """
    Processes images by applying trimming, resizing, and CLAHE enhancement, then saves them to a specified output folder.
    Skips images that are already processed.
    """
    if not os.path.exists(output_path_folder):
        os.makedirs(output_path_folder)

    files = [f for f in os.listdir(input_path_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    for img_file in tqdm(files):
        image_path = os.path.join(input_path_folder, img_file)
        output_image_path = os.path.join(output_path_folder, img_file)
        if not os.path.exists(output_image_path):
            try:
                image_original = cv2.imread(image_path)
                if image_original is not None:
                    image_trimmed_resized = trim_and_resize(image_original, output_size)
                    image_clahe = apply_clahe(image_trimmed_resized)
                    cv2.imwrite(output_image_path, image_clahe)
                    print(f"Processed and saved: {output_image_path}")
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
        else:
            print(f"Skipping {output_image_path}, already exists.")

if __name__ == "__main__":
    # Change the input/output directory
    input_dir = "./images"
    output_dir = "./preprocessed_img"
    output_size = 2000  # Update the desired output size as needed
    process_and_save_images(input_dir, output_dir, output_size)
