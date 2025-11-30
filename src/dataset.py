import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
# We resize to 256x256 to fit your 6GB VRAM comfortably
IMAGE_SIZE = 256
BATCH_SIZE = 16  # Small batch size for stability
NUM_WORKERS = 0  # Set to 0 for Windows to avoid multiprocessing errors

class BUSIDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Get image path from column 0 ('filepath')
        img_path = self.data_frame.iloc[idx, 0]
        
        # Construct full path (if needed, but your CSV likely has relative paths)
        # We assume the CSV paths are relative to the project root, 
        # so we might not need to join if running from root.
        # Let's be safe and check if it exists as is.
        if not os.path.exists(img_path):
             # Try joining with root_dir if provided
             img_path = os.path.join(self.root_dir, img_path)

        image = Image.open(img_path).convert("RGB")
        
        # Get label from column 1 ('label')
        label = int(self.data_frame.iloc[idx, 1])

        if self.transform:
            image = self.transform(image)

        return image, label

def get_dataloaders(data_dir="data/processed", batch_size=BATCH_SIZE):
    """
    Creates and returns DataLoaders for train, val, and test sets.
    """
    # Define standard transforms
    # 1. Resize to fixed size
    # 2. Convert to Tensor (0-1 float)
    # 3. Normalize (Standard ImageNet means/stds are usually safe to start)
    data_transforms = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Create Datasets
    train_dataset = BUSIDataset(csv_file=os.path.join(data_dir, "train.csv"), 
                                root_dir=".", 
                                transform=data_transforms)
    
    val_dataset = BUSIDataset(csv_file=os.path.join(data_dir, "val.csv"), 
                              root_dir=".", 
                              transform=data_transforms)
    
    test_dataset = BUSIDataset(csv_file=os.path.join(data_dir, "test.csv"), 
                               root_dir=".", 
                               transform=data_transforms)

    # Create DataLoaders
    # num_workers=0 is safest for Windows
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS)

    return train_loader, val_loader, test_loader

# --- TEST BLOCK ---
# Run this script directly to check if it works!
if __name__ == "__main__":
    print("Testing Data Loader...")
    
    try:
        train_loader, val_loader, test_loader = get_dataloaders()
        
        # Grab one batch to see if it works
        images, labels = next(iter(train_loader))
        
        print(f"SUCCESS! Loaded a batch of data.")
        print(f"Image Batch Shape: {images.shape}") # Should be [16, 3, 256, 256]
        print(f"Label Batch Shape: {labels.shape}") # Should be [16]
        print("Labels:", labels)
        
    except Exception as e:
        print(f"ERROR: {e}")
        print("Check your paths and CSV files.")