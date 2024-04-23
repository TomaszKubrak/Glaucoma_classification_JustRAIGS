import os
import pandas as pd
import torch
import torch.nn as nn
from torchvision.models import vit_l_16, ViT_L_16_Weights
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
from tqdm import tqdm

from data_utils import GlaucomaDataset, test_transform

####### Adjust this section as needed ############################
model_save_directory = './model/ViT_glaucoma_ROI'    
img_folder = './ROI_images' 
test_df = pd.read_csv('./Datasets/glaucoma_masks_test.csv')

# Unmute to validate ViT without ROI
# model_save_directory = './model/ViT_glaucoma_NO_ROI'  
# img_folder = './preprocessed_img' 
# test_df = pd.read_csv('./Datasets/glaucoma_no_mask_test.csv')
#################################################################

# Load test data
test_dataset = GlaucomaDataset(dataframe=test_df, img_folder=img_folder, transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=11, shuffle=True, num_workers=2)

# Model setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = vit_l_16(weights=ViT_L_16_Weights.IMAGENET1K_SWAG_E2E_V1)
model.heads.head = nn.Linear(model.heads.head.in_features, 1)
model.to(device)

def load_model(model, model_path, device):
    state_dict = torch.load(model_path, map_location=device)
    if any(k.startswith('module.') for k in state_dict.keys()):
        new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    else:
        new_state_dict = state_dict
    model.load_state_dict(new_state_dict)

def compute_metrics(actuals, probabilities):
    fpr, tpr, thresholds = roc_curve(actuals, probabilities)
    target_specificity = 0.95
    target_fpr = 1 - target_specificity

    # Find the first threshold where FPR is <= target FPR
    index = np.where(fpr <= target_fpr)[0][0]
    optimal_threshold = thresholds[index]
    predictions = (probabilities >= optimal_threshold).astype(int)

    TP = np.sum((actuals == 1) & (predictions == 1))
    TN = np.sum((actuals == 0) & (predictions == 0))
    FP = np.sum((actuals == 0) & (predictions == 1))
    FN = np.sum((actuals == 1) & (predictions == 0))

    sensitivity = TP / (TP + FN) if TP + FN > 0 else 0
    specificity = TN / (TN + FP) if TN + FP > 0 else 0
    accuracy = (TP + TN) / (TP + TN + FP + FN) if TP + TN + FP + FN > 0 else 0
    auc = roc_auc_score(actuals, probabilities) if len(np.unique(actuals)) > 1 else 0
    return sensitivity, specificity, accuracy, auc, optimal_threshold

# Evaluate all models in the directory
for filename in os.listdir(model_save_directory):
    if filename.endswith(".pth"):
        model_path = os.path.join(model_save_directory, filename)
        load_model(model, model_path, device)
        model.eval()

        all_labels = []
        all_probabilities = []
        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc=f"Evaluating {filename}", leave=True):
                images = images.to(device)
                outputs = model(images)
                probabilities = torch.sigmoid(outputs).squeeze()
                all_labels.extend(labels.numpy())
                all_probabilities.extend(probabilities.cpu().numpy())

        sensitivity, specificity, accuracy, auc_score, optimal_threshold = compute_metrics(np.array(all_labels), np.array(all_probabilities))
        print(f"Model: {filename}")
        print(f"Sensitivity: {sensitivity:.4f}")
        print(f"Specificity: {specificity:.4f}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"AUC Score: {auc_score:.4f}")
        print(f"Optimal Threshold: {optimal_threshold:.4f}\n")
