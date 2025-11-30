import os
import pandas as pd
from sklearn.model_selection import train_test_split

# 1. Define Paths
RAW_DATA_PATH = "data/raw"
PROCESSED_DATA_PATH = "data/processed"

# 2. Define Categories (Labels)
categories = {
    'normal': 0,
    'benign': 1,
    'malignant': 2
}

def create_dataset():
    data = []
    
    # Scan through each folder
    for category, label in categories.items():
        folder_path = os.path.join(RAW_DATA_PATH, category)
        
        # Check if folder exists
        if not os.path.exists(folder_path):
            print(f"WARNING: Folder not found: {folder_path}")
            continue
            
        # Loop through images
        for filename in os.listdir(folder_path):
            # We only want the images, not the masks (which end in _mask.png)
            if filename.endswith(".png") and "_mask" not in filename:
                img_path = os.path.join(folder_path, filename)
                data.append([img_path, label, category])
                
    # Create a DataFrame
    df = pd.DataFrame(data, columns=['filepath', 'label', 'label_name'])
    
    # Shuffle the data
    df = df.sample(frac=1).reset_index(drop=True)
    
    print(f"Total Images Found: {len(df)}")
    print(df['label_name'].value_counts())
    
    return df

if __name__ == "__main__":
    # Create directory if it doesn't exist
    os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)
    
    # 1. Create Master List
    print("Scanning data...")
    df = create_dataset()
    
    # 2. Split Data (80% Train, 10% Validation, 10% Test)
    # First split: Train vs Temp (Test + Val)
    train_df, temp_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)
    
    # Second split: Temp into Val and Test
    val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['label'], random_state=42)
    
    # 3. Save to CSV
    train_df.to_csv(f"{PROCESSED_DATA_PATH}/train.csv", index=False)
    val_df.to_csv(f"{PROCESSED_DATA_PATH}/val.csv", index=False)
    test_df.to_csv(f"{PROCESSED_DATA_PATH}/test.csv", index=False)
    
    print("\nSUCCESS: Data processed and split!")
    print(f"Train: {len(train_df)}")
    print(f"Val:   {len(val_df)}")
    print(f"Test:  {len(test_df)}")