import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import autoaugment

def train_transform():
    return transforms.Compose([
        transforms.Resize((512, 512), interpolation=transforms.InterpolationMode.BICUBIC),
        autoaugment.AutoAugment(autoaugment.AutoAugmentPolicy.IMAGENET),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def test_transform():
    return transforms.Compose([
        transforms.Resize((512, 512), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

class GlaucomaDataset(Dataset):
    """
    Args:
        dataframe (DataFrame): DataFrame containing the dataset information.
        img_folder (string): Directory with all the images.
        transform (callable, optional): Optional transform to be applied on a sample.
        extra_features (list of str, optional): Column names for the extra features.
    """
    def __init__(self, dataframe, img_folder, transform=None, extra_features=None):
        self.dataframe = dataframe
        self.img_folder = img_folder
        self.transform = transform
        self.extra_features = extra_features
        
    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_id = self.dataframe.iloc[idx]['Eye ID']
        for ext in ['.JPG','.JPEG', '.PNG', '.png', '.jpg', '.jpeg']:
            img_path = os.path.join(self.img_folder, f"{img_id}{ext}")
            if os.path.exists(img_path):
                break
        else:
            raise FileNotFoundError(f"No image found for ID {img_id} with any supported extension.")

        image = Image.open(img_path)
        if self.transform:
            image = self.transform(image)

        # Handling extra features 
        if self.extra_features == None :
            img_class = self.dataframe.iloc[idx]['Final Label']
            labels = torch.tensor(img_class, dtype=torch.float32)
        else:           
            extra_labels = self.dataframe.iloc[idx][self.extra_features].values.astype(float)
            labels = torch.tensor(extra_labels, dtype=torch.float32)
        
        return image, labels
