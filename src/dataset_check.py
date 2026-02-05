"""
Dataset Validation Script
--------------------------
Validates YOLO-format dataset and creates preview images with bounding boxes.

Usage:
    python src/dataset_check.py --data data/roboflow_dataset/data.yaml

This script:
1. Reads the data.yaml file
2. Counts images and labels in train/val/test splits
3. Validates label format
4. Creates preview images with bounding boxes
5. Shows class distribution
"""

import yaml
import argparse
from pathlib import Path
import cv2
import numpy as np
from collections import Counter
import random

# Color palette for bounding boxes (BGR format for OpenCV)
COLORS = [
    (255, 0, 0),      # Blue
    (0, 255, 0),      # Green
    (0, 0, 255),      # Red
    (255, 255, 0),    # Cyan
    (255, 0, 255),    # Magenta
    (0, 255, 255),    # Yellow
    (128, 0, 128),    # Purple
    (255, 165, 0),    # Orange
]


def load_yaml(yaml_path):
    """Load and parse data.yaml file."""
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    return data


def convert_yolo_to_bbox(yolo_coords, img_width, img_height):
    """
    Convert YOLO format (normalized center x, center y, width, height)
    to pixel coordinates (x1, y1, x2, y2).
    """
    x_center, y_center, width, height = yolo_coords
    
    # Convert to pixel coordinates
    x_center *= img_width
    y_center *= img_height
    width *= img_width
    height *= img_height
    
    # Convert to corner coordinates
    x1 = int(x_center - width / 2)
    y1 = int(y_center - height / 2)
    x2 = int(x_center + width / 2)
    y2 = int(y_center + height / 2)
    
    return x1, y1, x2, y2


def draw_boxes_on_image(image_path, label_path, class_names):
    """Draw bounding boxes on image."""
    # Read image
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Warning: Could not read image {image_path}")
        return None
    
    img_height, img_width = img.shape[:2]
    
    # Read labels
    if not label_path.exists():
        print(f"Warning: No label file for {image_path.name}")
        return img
    
    with open(label_path, 'r') as f:
        labels = f.readlines()
    
    # Draw each bounding box
    for label in labels:
        parts = label.strip().split()
        if len(parts) < 5:
            continue
        
        class_id = int(parts[0])
        yolo_coords = [float(x) for x in parts[1:5]]
        
        # Convert to pixel coordinates
        x1, y1, x2, y2 = convert_yolo_to_bbox(yolo_coords, img_width, img_height)
        
        # Get class name and color
        class_name = class_names[class_id] if class_id < len(class_names) else f"Class_{class_id}"
        color = COLORS[class_id % len(COLORS)]
        
        # Draw rectangle
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        
        # Draw label background
        label_text = class_name
        (text_width, text_height), baseline = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        cv2.rectangle(
            img,
            (x1, y1 - text_height - baseline - 5),
            (x1 + text_width, y1),
            color,
            -1
        )
        
        # Draw label text
        cv2.putText(
            img,
            label_text,
            (x1, y1 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )
    
    return img


def validate_dataset(data_yaml_path, num_previews=10):
    """Validate dataset and create preview images."""
    
    print("=" * 70)
    print("DATASET VALIDATION")
    print("=" * 70)
    
    # Load data.yaml
    print(f"\n[1/5] Loading configuration from: {data_yaml_path}")
    data_yaml_path = Path(data_yaml_path)
    dataset_root = data_yaml_path.parent
    
    config = load_yaml(data_yaml_path)
    
    # Extract information
    class_names = config.get('names', [])
    num_classes = config.get('nc', len(class_names))
    
    print(f"✅ Dataset root: {dataset_root}")
    print(f"✅ Number of classes: {num_classes}")
    print(f"✅ Class names: {class_names}")
    
    # Check each split
    print(f"\n[2/5] Checking dataset splits...")
    splits = ['train', 'valid', 'test']
    split_stats = {}
    all_class_counts = Counter()
    
    for split in splits:
        images_dir = dataset_root / split / 'images'
        labels_dir = dataset_root / split / 'labels'
        
        if not images_dir.exists():
            print(f"⚠️  Warning: {split}/images/ not found")
            continue
        
        if not labels_dir.exists():
            print(f"⚠️  Warning: {split}/labels/ not found")
            continue
        
        # Count images and labels
        image_files = list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.png'))
        label_files = list(labels_dir.glob('*.txt'))
        
        # Count classes in this split
        class_counts = Counter()
        for label_file in label_files:
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        class_counts[class_id] += 1
                        all_class_counts[class_id] += 1
        
        split_stats[split] = {
            'images': len(image_files),
            'labels': len(label_files),
            'class_counts': class_counts
        }
        
        print(f"\n{split.upper()}:")
        print(f"  Images: {len(image_files)}")
        print(f"  Labels: {len(label_files)}")
        
        if len(image_files) != len(label_files):
            print(f"  ⚠️  Warning: Mismatch between images and labels!")
    
    # Show class distribution
    print(f"\n[3/5] Overall class distribution:")
    print("-" * 50)
    for class_id in sorted(all_class_counts.keys()):
        class_name = class_names[class_id] if class_id < len(class_names) else f"Class_{class_id}"
        count = all_class_counts[class_id]
        print(f"  {class_name:30s}: {count:5d} instances")
    print("-" * 50)
    print(f"  {'TOTAL':30s}: {sum(all_class_counts.values()):5d} instances")
    
    # Create preview images
    print(f"\n[4/5] Creating {num_previews} preview images...")
    output_dir = Path("artifacts/sample_outputs/dataset_preview")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get random images from train split
    train_images_dir = dataset_root / 'train' / 'images'
    train_labels_dir = dataset_root / 'train' / 'labels'
    
    if train_images_dir.exists():
        all_images = list(train_images_dir.glob('*.jpg')) + list(train_images_dir.glob('*.png'))
        
        if len(all_images) > 0:
            # Select random images
            preview_images = random.sample(all_images, min(num_previews, len(all_images)))
            
            for idx, img_path in enumerate(preview_images):
                # Get corresponding label file
                label_path = train_labels_dir / (img_path.stem + '.txt')
                
                # Draw boxes
                img_with_boxes = draw_boxes_on_image(img_path, label_path, class_names)
                
                if img_with_boxes is not None:
                    output_path = output_dir / f"preview_{idx+1:02d}.jpg"
                    cv2.imwrite(str(output_path), img_with_boxes)
                    print(f"  ✅ Saved: {output_path.name}")
        else:
            print("  ⚠️  No images found in train/images/")
    
    # Summary
    print(f"\n[5/5] Validation complete!")
    print("=" * 70)
    print("SUMMARY:")
    print("-" * 70)
    
    total_images = sum(stats['images'] for stats in split_stats.values())
    total_labels = sum(stats['labels'] for stats in split_stats.values())
    
    print(f"Total images: {total_images}")
    print(f"Total labels: {total_labels}")
    print(f"Classes: {num_classes}")
    print(f"Preview images saved to: {output_dir}")
    
    # Check for potential issues
    print("\n" + "=" * 70)
    print("DATASET HEALTH CHECK:")
    print("-" * 70)
    
    issues = []
    
    # Check class balance
    if len(all_class_counts) > 0:
        max_count = max(all_class_counts.values())
        min_count = min(all_class_counts.values())
        imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
        
        if imbalance_ratio > 10:
            issues.append(f"⚠️  Class imbalance detected (ratio: {imbalance_ratio:.1f}:1)")
        else:
            print("✅ Class distribution is reasonably balanced")
    
    # Check dataset size
    if total_images < 100:
        issues.append("⚠️  Small dataset (<100 images) - consider augmentation")
    elif total_images < 500:
        print("✅ Dataset size is adequate (consider more data if possible)")
    else:
        print("✅ Dataset size is good")
    
    # Check split sizes
    if 'train' in split_stats and 'valid' in split_stats:
        train_size = split_stats['train']['images']
        valid_size = split_stats['valid']['images']
        if train_size > 0:
            val_ratio = valid_size / train_size
            if val_ratio < 0.1 or val_ratio > 0.3:
                issues.append(f"⚠️  Unusual train/val split ratio ({val_ratio:.2f})")
            else:
                print("✅ Train/validation split is appropriate")
    
    if issues:
        print("\nPotential Issues:")
        for issue in issues:
            print(f"  {issue}")
    else:
        print("\n✅ No major issues detected")
    
    print("=" * 70)
    print("\n✅ You're ready to train! Run:")
    print(f"   python src/train.py --data {data_yaml_path}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description='Validate YOLO dataset')
    parser.add_argument(
        '--data',
        type=str,
        required=True,
        help='Path to data.yaml file'
    )
    parser.add_argument(
        '--previews',
        type=int,
        default=10,
        help='Number of preview images to generate (default: 10)'
    )
    
    args = parser.parse_args()
    
    validate_dataset(args.data, args.previews)


if __name__ == "__main__":
    main()