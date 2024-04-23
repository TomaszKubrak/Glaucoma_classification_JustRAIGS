import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim
from torchvision.models import vit_l_16, ViT_L_16_Weights
from torch.utils.data import DataLoader
from tqdm import tqdm
import re

from data_utils import GlaucomaDataset, train_transform, test_transform

################ Adjust this section as needed ################
model_name = "ViT_10_ROI"
model_save_directory = f'./model/{model_name}'
img_folder = './ROI_images'
train_df = pd.read_csv('./Datasets/10_features_masks_train.csv')
test_df = pd.read_csv('./Datasets/10_features_masks_test.csv')

# Unmute to train ViT without ROI
# model_name = "ViT_10_NO_ROI"
# model_save_directory = f'./model/{model_name}'
# img_folder = './preprocessed_img' 
# train_df = pd.read_csv('./Datasets/10_features_no_mask_train.csv')
# test_df = pd.read_csv('./Datasets/10_features_no_mask_test.csv')
###############################################################

if not os.path.exists(model_save_directory):
    os.makedirs(model_save_directory)

extra_features = ['ANRS', 'ANRI', 'RNFLDS', 'RNFLDI', 'BCLVS', 'BCLVI', 'NVT', 'DH', 'LD', 'LC']

train_dataset = GlaucomaDataset(dataframe=train_df, img_folder=img_folder, transform=train_transform, extra_features=extra_features)
test_dataset = GlaucomaDataset(dataframe=test_df, img_folder=img_folder, transform=test_transform, extra_features=extra_features)

train_loader = DataLoader(train_dataset, batch_size=20, shuffle=True, num_workers=8)
test_loader = DataLoader(test_dataset, batch_size=20, shuffle=True, num_workers=8)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

weights = ViT_L_16_Weights.IMAGENET1K_SWAG_E2E_V1
model = vit_l_16(weights=weights)
num_features = model.heads.head.in_features  
model.heads.head = nn.Linear(num_features, 10)

if torch.cuda.device_count() > 1:
    print(f"Let's use {torch.cuda.device_count()} GPUs!")
    model = nn.DataParallel(model)
model.to(device)

# Define weights and weighted BCELoss
pos_weights = [len(train_df[train_df[feature] == 0]) / len(train_df[train_df[feature] == 1]) for feature in extra_features]
pos_weights_tensor = torch.tensor(pos_weights, dtype=torch.float, device=device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights_tensor)

# Accessing parameters after applying DataParallel
if torch.cuda.device_count() > 1:
    base_params = [p for n, p in model.module.named_parameters() if 'heads.head' not in n]
    classifier_params = model.module.heads.head.parameters()
else:
    base_params = [p for n, p in model.named_parameters() if 'heads.head' not in n]
    classifier_params = model.heads.head.parameters()

# Define an AdamW optimizer with differential learning rate and weight decay
optimizer = torch.optim.AdamW([
    {'params': base_params, 'lr': 1e-5, 'weight_decay': 1e-4},
    {'params': classifier_params, 'lr': 1e-4, 'weight_decay': 1e-4}
])

# Check if model was trained before, if so continue training from last epoch
# Load the last saved model state
latest_model_path = os.path.join(model_save_directory, f"{model_name}_last.pth")

# Check if the model file exists and load it
if os.path.isfile(latest_model_path):
    model.load_state_dict(torch.load(latest_model_path))
    print(f"Loaded model from {latest_model_path}, continuing training from last saved state.")
else:
    print("No saved model found, starting training from scratch")

# Training and validation loop
num_epochs = 100
best_hamming_loss = float('inf')
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}")
    for batch_idx, (images, labels) in progress_bar:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        
        outputs = model(images)  
        loss = criterion(outputs, labels.float())
        loss.backward()  
        optimizer.step()  

        train_loss += loss.item() * images.size(0)
        progress_bar.set_postfix({'train_loss': loss.item()})
    train_loss /= len(train_loader.dataset)

    # Validation phase
    model.eval()
    validation_loss = 0.0
    hamming_loss_sum = 0.0
    total_labels = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)  # Normal forward pass
            loss = criterion(outputs, labels.float())
            validation_loss += loss.item() * images.size(0)

            predictions = (torch.sigmoid(outputs) > 0.5)
            hamming_loss_sum += (predictions != labels).sum().item()
            total_labels += labels.numel()

    validation_loss /= len(test_loader.dataset)
    hamming_loss = hamming_loss_sum / total_labels

    print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Validation Loss: {validation_loss:.4f}, Hamming Loss: {hamming_loss:.4f}")

    epoch_save_path = os.path.join(model_save_directory, f"{model_name}_last.pth")
    torch.save(model.state_dict(), epoch_save_path)
    print(f"Model saved to {epoch_save_path} after epoch {epoch+1}")

    if hamming_loss < best_hamming_loss:
        best_hamming_loss = hamming_loss
        best_model_save_path = os.path.join(model_save_directory, f"{model_name}_best.pth")
        torch.save(model.state_dict(), best_model_save_path)
        print(f"New best model saved to {best_model_save_path} with Hamming Loss: {hamming_loss:.4f}")