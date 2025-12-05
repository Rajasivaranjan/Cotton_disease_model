import os
import shutil
import random
from pathlib import Path

def prepare_dataset(source_dir, target_dir, split_ratio=(0.7, 0.2, 0.1)):
    """
    Splits the dataset into train, val, and test sets.
    
    Args:
        source_dir (str): Path to the original dataset.
        target_dir (str): Path where the new dataset will be created.
        split_ratio (tuple): Ratios for (train, val, test). Must sum to 1.
    """
    source_path = Path(source_dir)
    target_path = Path(target_dir)
    
    if target_path.exists():
        print(f"Removing existing target directory: {target_path}")
        shutil.rmtree(target_path)
    
    # Define subsets
    subsets = ['train', 'val', 'test']
    
    # Get all classes (subdirectories in source)
    classes = [d.name for d in source_path.iterdir() if d.is_dir()]
    print(f"Found classes: {classes}")
    
    for class_name in classes:
        class_dir = source_path / class_name
        images = [f for f in class_dir.iterdir() if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']]
        
        # Shuffle images
        random.shuffle(images)
        
        # Calculate split indices
        n_total = len(images)
        n_train = int(n_total * split_ratio[0])
        n_val = int(n_total * split_ratio[1])
        # n_test is the remainder
        
        splits = {
            'train': images[:n_train],
            'val': images[n_train:n_train + n_val],
            'test': images[n_train + n_val:]
        }
        
        print(f"Processing class '{class_name}': {n_total} images -> Train: {len(splits['train'])}, Val: {len(splits['val'])}, Test: {len(splits['test'])}")
        
        for subset in subsets:
            subset_dir = target_path / subset / class_name
            subset_dir.mkdir(parents=True, exist_ok=True)
            
            for img in splits[subset]:
                shutil.copy2(img, subset_dir / img.name)
                
    print(f"\nDataset preparation complete! Saved to {target_path}")

if __name__ == "__main__":
    SOURCE_DIR = "/Users/rajasivaranjan/cotton_disease_model/Dataset/Cotton_Original_Dataset"
    TARGET_DIR = "/Users/rajasivaranjan/cotton_disease_model/yolo_dataset"
    
    prepare_dataset(SOURCE_DIR, TARGET_DIR)
