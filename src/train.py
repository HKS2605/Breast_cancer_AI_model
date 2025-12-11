import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import time
import copy
import os

# Import our custom tools from the previous step
from dataset import get_dataloaders

# --- CONFIGURATION (Tuned for RTX 3050 6GB) ---
BATCH_SIZE = 16        # Keep small for VRAM
LEARNING_RATE = 0.0001  # chamged the learning rate to 0.0001
NUM_EPOCHS = 50        # increased number of epochs for better training
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def build_model(num_classes=3):
    """
    Loads a pre-trained EfficientNet-B0 and adapts it for our 3 classes.
    """
    # Load the modern EfficientNet-B0 model
    # Weights="DEFAULT" downloads the best available pre-trained weights
    from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
    
    weights = EfficientNet_B0_Weights.DEFAULT
    model = efficientnet_b0(weights=weights)
    
    # FREEZE the early layers (feature extractor)
    # This prevents us from destroying the pre-trained knowledge
    for param in model.features.parameters():
        param.requires_grad = True # Set to True to fine-tune the feature extractor
        
    # REPLACE the final "head" (classifier)
    # EfficientNet's classifier is a Sequential block. We replace the last Linear layer.
    # The original last layer has 1280 inputs.
    num_features = model.classifier[1].in_features
    
    model.classifier[1] = nn.Linear(num_features, num_classes)
    
    return model.to(DEVICE)

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10):
    since = time.time()
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    # Initialize the GradScaler for Mixed Precision (Crucial for 6GB VRAM)
    scaler = torch.cuda.amp.GradScaler()
    
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
                dataloader = train_loader
            else:
                model.eval()   # Set model to evaluate mode
                dataloader = val_loader

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data
            # tqdm creates the nice progress bar
            for inputs, labels in tqdm(dataloader, desc=f"{phase} Phase"):
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward Pass
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    
                    # Mixed Precision Context
                    # This runs the heavy math in 16-bit to save VRAM
                    with torch.cuda.amp.autocast(enabled=(phase=='train')):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                    # Backward + Optimize only if in training phase
                    if phase == 'train':
                        # Scale the loss and call backward (Magic of AMP)
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_corrects.double() / len(dataloader.dataset)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            # Save history
            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc.item())
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc.item())

            # Deep copy the model if it's the best one so far
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                
                # Save checkpoint immediately
                torch.save(model.state_dict(), 'models/best_model.pth')
                print(f"--> New Best Model Saved! (Acc: {best_acc:.4f})")

    time_elapsed = time.time() - since
    print(f'\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model, history

if __name__ == "__main__":
    # 1. Setup Directories
    os.makedirs("models", exist_ok=True)
    
    # 2. Get Data
    print("Initializing Data Loaders...")
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=BATCH_SIZE)
    
    # 3. Build Model
    print(f"Building EfficientNet-B0 on {DEVICE}...")
    model = build_model(num_classes=3)
    
    # 4. Setup Training Tools
    # CrossEntropyLoss is standard for multi-class classification
    criterion = nn.CrossEntropyLoss()
    
    # Adam is a standard, efficient optimizer
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # 5. Run Training
    print("Starting Training...")
    model, history = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=NUM_EPOCHS)
    
    print("DONE! Your AI brain is ready.")