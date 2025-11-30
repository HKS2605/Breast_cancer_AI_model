import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from dataset import get_dataloaders
from train import build_model # Re-use our model builder

# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "models/best_model.pth"
BATCH_SIZE = 16

def evaluate_model():
    # 1. Get Test Data
    _, _, test_loader = get_dataloaders(batch_size=BATCH_SIZE)
    class_names = ['normal', 'benign', 'malignant'] # 0, 1, 2

    # 2. Load Model
    print(f"Loading model from {MODEL_PATH}...")
    model = build_model(num_classes=3)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.to(DEVICE)
    model.eval() # Set to evaluation mode (turns off dropout, etc.)

    # 3. Run Predictions
    all_preds = []
    all_labels = []

    print("Running predictions on test set...")
    with torch.no_grad(): # No need to track gradients for eval
        for inputs, labels in test_loader:
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    # 4. Generate Metrics
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    # 5. Plot Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    print("Confusion Matrix saved as 'confusion_matrix.png'")
    plt.show()

if __name__ == "__main__":
    evaluate_model()