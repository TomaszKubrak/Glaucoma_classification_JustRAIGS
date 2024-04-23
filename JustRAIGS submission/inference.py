from ultralytics import YOLO
import cv2
from PIL import Image
from helper import DEFAULT_GLAUCOMATOUS_FEATURES, inference_tasks
import torch
import torch.nn as nn
import torch.optim
from torchvision import transforms
import numpy as np

def apply_clahe(img, clip_limit=3.0, tile_grid_size=(8, 8)):
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    r, g, b = cv2.split(img)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    r_clahe, g_clahe, b_clahe = clahe.apply(r), clahe.apply(g), clahe.apply(b)
    clahe_img = cv2.merge([r_clahe, g_clahe, b_clahe])
    return cv2.cvtColor(clahe_img, cv2.COLOR_BGR2RGB)

def trim_and_resize(im, output_size):
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
    im_resized = im_pil.resize(new_size, Image.Resampling.LANCZOS)
    new_im = Image.new("RGB", (output_size, output_size))
    new_im.paste(im_resized, ((output_size - new_size[0]) // 2, (output_size - new_size[1]) // 2))
    return new_im

def find_encompassing_bbox(bboxes):
    min_x1, min_y1, max_x2, max_y2 = bboxes[0]
    for bbox in bboxes[1:]:
        x1, y1, x2, y2 = bbox
        min_x1 = min(min_x1, x1)
        min_y1 = min(min_y1, y1)
        max_x2 = max(max_x2, x2)
        max_y2 = max(max_y2, y2)
    return [min_x1, min_y1, max_x2, max_y2]

def crop_and_resize_image(img, bbox, target_size=(518, 518)):
    x1, y1, x2, y2 = map(int, bbox)
    bbox_width = x2 - x1
    bbox_height = y2 - y1
    if bbox_width > target_size[0] or bbox_height > target_size[1]:
        return None  
    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2
    half_target = target_size[0] // 2
    start_x = max(0, center_x - half_target)
    start_y = max(0, center_y - half_target)
    end_x = min(img.shape[1], center_x + half_target)
    end_y = min(img.shape[0], center_y + half_target)
    crop_width = end_x - start_x
    crop_height = end_y - start_y
    if crop_width < target_size[0] or crop_height < target_size[1]:
        max_possible_square = min(crop_width, crop_height)
        start_x = center_x - max_possible_square // 2
        start_y = center_y - max_possible_square // 2
        end_x = start_x + max_possible_square
        end_y = start_y + max_possible_square
        start_x = max(0, start_x)
        start_y = max(0, start_y)
        end_x = min(img.shape[1], end_x)
        end_y = min(img.shape[0], end_y)
    cropped_img = img[start_y:end_y, start_x:end_x]
    if cropped_img.size == 0:
        return None
    resized_img = cv2.resize(cropped_img, target_size, interpolation=cv2.INTER_CUBIC)
    return resized_img

def run():
    _show_torch_cuda_info()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    #YoloV8 for segmentation
    segmentation_model_path = './model/YoloV8.pt'
    segmentation_model = YOLO(segmentation_model_path)

    
    # ViT for glaucoma classification without ROI preprocessing
    glaucoma_model = torch.load('./model/ViT_pretrained.pth', map_location=device)
    glaucoma_model.heads.head = torch.nn.Linear(1024, 1)
    glaucoma_model_path = './model/ViT_RG_NO_ROI.pth'
    state_dict = torch.load(glaucoma_model_path, map_location=device)
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    glaucoma_model.load_state_dict(new_state_dict)
    glaucoma_model.to(device).eval()

    # ViT for glaucoma classification with ROI preprocessing
    glaucoma_model_ROI = torch.load('./model/ViT_pretrained.pth', map_location=device)
    glaucoma_model_ROI.heads.head = torch.nn.Linear(1024, 1)
    glaucoma_model_ROI_path = './model/ViT_RG_ROI.pth'
    state_dict_ROI = torch.load(glaucoma_model_ROI_path, map_location=device)
    new_state_dict_ROI = {k.replace('module.', ''): v for k, v in state_dict_ROI.items()}
    glaucoma_model_ROI.load_state_dict(new_state_dict_ROI)
    glaucoma_model_ROI.to(device).eval()

    # ViT for extra features classification without ROI preprocessing
    features_model = torch.load('./model/ViT_pretrained.pth', map_location=device)
    features_model.heads.head = torch.nn.Linear(1024, 10) 
    features_model_path = './model/ViT_10_NO_ROI.pth'
    state_dict_features = torch.load(features_model_path, map_location=device)
    new_state_dict_features = {k.replace('module.', ''): v for k, v in state_dict_features.items()}
    features_model.load_state_dict(new_state_dict_features)
    features_model.to(device).eval()

    # ViT for extra features classification with ROI preprocessing
    features_model_ROI = torch.load('./model/ViT_pretrained.pth', map_location=device)
    features_model_ROI.heads.head = torch.nn.Linear(1024, 10) 
    features_model_ROI_path = './model/ViT_10_ROI.pth'
    state_dict_features_ROI = torch.load(features_model_ROI_path, map_location=device)
    new_state_dict_features_ROI = {k.replace('module.', ''): v for k, v in state_dict_features_ROI.items()}
    features_model_ROI.load_state_dict(new_state_dict_features_ROI)
    features_model_ROI.to(device).eval()

    for jpg_image_file_name, save_prediction in inference_tasks():
        print(f"Running inference on {jpg_image_file_name}")
        
        img_path_str = str(jpg_image_file_name)
        img_array = cv2.imread(img_path_str)
        trimmed_img = trim_and_resize(img_array, 2000)
        clahe_img_array = apply_clahe(trimmed_img)

        # Make predictions using YOLO model on the CLAHE processed image
        results = segmentation_model(clahe_img_array)
        boxes = results[0].boxes

        # Checking if optic disc was detected if not CLAHE image is passed for inferences
        if len(boxes) == 0:
            cropped_resized_img = None
        else:
            # Find the encompassing bounding box, crop and resize
            bboxes = [box.cpu().numpy().tolist() for box in boxes.xyxy]
            encompassing_bbox = find_encompassing_bbox(bboxes)
            cropped_resized_img = crop_and_resize_image(clahe_img_array, encompassing_bbox)
            
        transform = transforms.Compose([
            transforms.Resize((512, 512), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        if cropped_resized_img is not None:
            preprocessed_img = transform(Image.fromarray(cv2.cvtColor(cropped_resized_img, cv2.COLOR_BGR2RGB)))
            ROI_detected = True
        else:
            preprocessed_img = transform(Image.fromarray(cv2.cvtColor(clahe_img_array, cv2.COLOR_BGR2RGB)))
            ROI_detected = False
       
        preprocessed_img = preprocessed_img.to(device)
        preprocessed_img = preprocessed_img.unsqueeze(0)

        with torch.no_grad():
            if ROI_detected:
                glaucoma_output_ROI = glaucoma_model_ROI(preprocessed_img)
                is_referable_glaucoma_likelihood = torch.sigmoid(glaucoma_output_ROI).item()
                features_output = features_model_ROI(preprocessed_img)
                features_probs = torch.sigmoid(features_output).squeeze().tolist()
                features = {k: v > 0.5 for k, v in zip(DEFAULT_GLAUCOMATOUS_FEATURES.keys(), features_probs)}
            else:
                glaucoma_output = glaucoma_model(preprocessed_img)
                is_referable_glaucoma_likelihood = torch.sigmoid(glaucoma_output).item()
                features_output = features_model(preprocessed_img)
                features_probs = torch.sigmoid(features_output).squeeze().tolist()
                features = {k: v > 0.5 for k, v in zip(DEFAULT_GLAUCOMATOUS_FEATURES.keys(), features_probs)}

        is_referable_glaucoma = is_referable_glaucoma_likelihood > 0.5

        # Finally, save the answer
        save_prediction(
            is_referable_glaucoma,
            is_referable_glaucoma_likelihood,
            features,
        )
    return 0

def _show_torch_cuda_info():
    import torch

    print("=+=" * 10)
    print(f"Torch CUDA is available: {(available := torch.cuda.is_available())}")
    if available:
        print(f"\tnumber of devices: {torch.cuda.device_count()}")
        print(f"\tcurrent device: { (current_device := torch.cuda.current_device())}")
        print(f"\tproperties: {torch.cuda.get_device_properties(current_device)}")
    print("=+=" * 10)


if __name__ == "__main__":
    raise SystemExit(run())