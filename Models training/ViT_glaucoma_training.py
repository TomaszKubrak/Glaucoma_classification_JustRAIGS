import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim
from torchvision.models import vit_l_16, ViT_L_16_Weights
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import re

from data_utils import GlaucomaDataset, train_transform

################ Adjust this section as needed ################
model_name = "ViT_RG_ROI"
model_save_directory = f'./model/{model_name}'
img_folder = './ROI_images'
train_df = pd.read_csv('./Datasets/glaucoma_masks_train.csv')

# Unmute to train ViT without ROI
# model_name = "ViT_RG_NO_ROI"
# model_save_directory = f'./model/{model_name}'
# img_folder = './preprocessed_img' 
# train_df = pd.read_csv('./Datasets/glaucoma_no_mask_train.csv')
###############################################################

if not os.path.exists(model_save_directory):
    os.makedirs(model_save_directory)

train_dataset = GlaucomaDataset(dataframe=train_df, img_folder=img_folder, transform=train_transform, extra_features=None)

train_loader = DataLoader(train_dataset, batch_size=20, shuffle=True, num_workers=8) 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

weights = ViT_L_16_Weights.IMAGENET1K_SWAG_E2E_V1
model = vit_l_16(weights=weights)
num_features = model.heads.head.in_features  
model.heads.head = nn.Linear(num_features, 1)

if torch.cuda.device_count() > 1:
    print(f"Let's use {torch.cuda.device_count()} GPUs!")
    model = nn.DataParallel(model)

model.to(device)

# Define weights and weighted BCELoss
negative_class = (len(train_df[train_df['Final Label']==0]))
positive_class = (len(train_df[train_df['Final Label']==1]))
pos_weight_value = negative_class / positive_class
pos_weight_tensor = torch.tensor([pos_weight_value],  dtype=torch.float, device=device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

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

# Check if model has existing epoch, if so continue training from last epoch
latest_model_path = None
start_epoch = 0
for file in os.listdir(model_save_directory):
    if file.startswith(f"{model_name}_epoch_") and file.endswith(".pth"):
        epoch_num = int(re.findall(r"\d+", file)[0])
        if epoch_num > start_epoch:
            start_epoch = epoch_num
            latest_model_path = os.path.join(model_save_directory, file)

if latest_model_path:
    model.load_state_dict(torch.load(latest_model_path))
    print(f"Loaded model from {latest_model_path}, continuing training from epoch {start_epoch+1}")
else:
    print("No saved model found, starting training from scratch")

scaler = GradScaler()
num_epochs = 100

# Training and validation loop
for epoch in range(start_epoch, num_epochs):
    model.train() 
    train_loss = 0.0

    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}")
    for batch_idx, (images, labels) in progress_bar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        # Using autocast for the forward pass
        with autocast():
            outputs = model(images)
            loss = criterion(outputs.squeeze(), labels.float())

        # Scales loss and calls backward() to create scaled gradients
        scaler.scale(loss).backward()

        # Updates the weights
        scaler.step(optimizer)
        scaler.update()

        train_loss += loss.item() * images.size(0)
        progress_bar.set_postfix({'train_loss': loss.item()})

    train_loss /= len(train_loader.dataset)

    # Save the model state after each epoch with epoch number
    epoch_save_path = os.path.join(model_save_directory, f"{model_name}_epoch_{epoch+1}.pth")
    torch.save(model.state_dict(), epoch_save_path)
    print(f"Model saved to {epoch_save_path} after epoch {epoch+1}")