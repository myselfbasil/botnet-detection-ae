import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

def split_and_save_dataset(file_path):
    # Create directory for splits if it doesn't exist
    split_dir = '/Users/basilshaji/Desktop/Projects/data-science/dataset/splits'
    os.makedirs(split_dir, exist_ok=True)
    
    # Load the dataset
    print("Loading dataset...")
    df = pd.read_csv(file_path)
    
    # First split: 60% train, 40% temp (for val and test)
    train_data, temp_data = train_test_split(
        df, 
        test_size=0.4,
        random_state=42,
        shuffle=True
    )
    
    # Second split: Split temp into validation and test (50% each, resulting in 20% of original data each)
    val_data, test_data = train_test_split(
        temp_data,
        test_size=0.5,
        random_state=42,
        shuffle=True
    )
    
    # Save the splits
    train_path = os.path.join(split_dir, 'train.csv')
    val_path = os.path.join(split_dir, 'validation.csv')
    test_path = os.path.join(split_dir, 'test.csv')
    
    train_data.to_csv(train_path, index=False)
    val_data.to_csv(val_path, index=False)
    test_data.to_csv(test_path, index=False)
    
    # Print information about the splits
    print("\nDataset split complete:")
    print(f"Original dataset size: {len(df)}")
    print(f"Training set size: {len(train_data)} ({len(train_data)/len(df)*100:.1f}%)")
    print(f"Validation set size: {len(val_data)} ({len(val_data)/len(df)*100:.1f}%)")
    print(f"Test set size: {len(test_data)} ({len(test_data)/len(df)*100:.1f}%)")
    print(f"\nFiles saved in: {split_dir}")

if __name__ == "__main__":
    file_path = "/Users/basilshaji/Desktop/Projects/data-science/dataset/cleaned_dataset.csv"
    split_and_save_dataset(file_path)