"""
Improved Training Script - V2
-----------------------------
Enhanced training with better augmentation for rust, occlusion, and edge cases.

Usage:
    python src/train_v2.py --data data/roboflow_dataset/data.yaml --epochs 100
"""

import argparse
from ultralytics import YOLO
from pathlib import Path
import shutil
import yaml


def train_model_v2(
    data_yaml,
    model_name='yolo11s.pt',  # Using 'small' instead of 'nano' for better accuracy
    epochs=100,
    imgsz=640,
    batch=8,  # Reduced batch for larger model
    patience=20,
    project_name='railway_defect'
):
    """
    Train YOLO11 model V2 with improved augmentation for real-world robustness.
    
    Key improvements:
    - Stronger augmentation for rust/corrosion
    - Random erasing for occlusion handling
    - Better color jitter for lighting variation
    - Longer training (100 epochs vs 50)
    - Larger model (YOLO11s vs YOLO11n)
    """
    
    print("=" * 70)
    print("YOLO11 TRAINING V2 - Enhanced for Real-World Robustness")
    print("=" * 70)
    
    # Validate data.yaml exists
    data_path = Path(data_yaml)
    if not data_path.exists():
        raise FileNotFoundError(f"Data configuration not found: {data_yaml}")
    
    # Load dataset info
    with open(data_path, 'r') as f:
        data_config = yaml.safe_load(f)
    
    class_names = data_config.get('names', [])
    num_classes = data_config.get('nc', len(class_names))
    
    print(f"\n[1/5] Dataset Configuration:")
    print(f"  Data config: {data_yaml}")
    print(f"  Classes: {class_names}")
    print(f"  Number of classes: {num_classes}")
    
    # Load model
    print(f"\n[2/5] Loading pre-trained model: {model_name}")
    print("  Using YOLO11s (small) for better accuracy than nano")
    print("  Trade-off: Slower inference but better detection")
    
    model = YOLO(model_name)
    print("  ✅ Model loaded successfully")
    
    # Training configuration
    print(f"\n[3/5] Enhanced Training Configuration:")
    print(f"  Epochs: {epochs} (increased from 50)")
    print(f"  Image size: {imgsz}x{imgsz}")
    print(f"  Batch size: {batch}")
    print(f"  Patience: {patience} epochs")
    print(f"  Enhanced augmentation: YES")
    print("    • Stronger HSV shifts (for rust/corrosion)")
    print("    • Random erasing (for debris/occlusion)")
    print("    • Copy-paste (for small objects)")
    
    # Start training
    print(f"\n[4/5] Starting enhanced training...")
    print("-" * 70)
    
    try:
        results = model.train(
            data=str(data_yaml),
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            patience=patience,
            name=f'{project_name}_v2',
            pretrained=True,
            optimizer='AdamW',  # Better optimizer for small datasets
            lr0=0.001,  # Lower initial learning rate
            lrf=0.01,  # Final learning rate factor
            momentum=0.937,
            weight_decay=0.0005,
            warmup_epochs=3,
            verbose=True,
            seed=42,
            deterministic=True,
            save=True,
            plots=True,
            
            # ENHANCED AUGMENTATION FOR REAL-WORLD ROBUSTNESS
            # ================================================
            
            # HSV augmentation (handle rust, lighting, corrosion)
            hsv_h=0.03,      # Hue: Increased from 0.015 (handle color shifts from rust)
            hsv_s=0.9,       # Saturation: Increased from 0.7 (handle faded/corroded surfaces)
            hsv_v=0.6,       # Value: Increased from 0.4 (handle shadows, overcast)
            
            # Geometric augmentation
            degrees=5.0,     # Slight rotation (was 0, now 5° for angle variation)
            translate=0.15,  # Translation: Increased from 0.1
            scale=0.7,       # Scale: Increased from 0.5 (zoom in/out more)
            shear=2.0,       # Shear: Added (was 0) for perspective variation
            perspective=0.0001,  # Slight perspective (was 0)
            
            # Flip augmentation
            flipud=0.0,      # No vertical flip (tracks don't flip vertically)
            fliplr=0.5,      # Horizontal flip
            
            # Advanced augmentation
            mosaic=1.0,      # Mosaic (combines 4 images)
            mixup=0.1,       # MixUp: Increased from 0 (blend images)
            copy_paste=0.3,  # NEW: Copy-paste small objects to handle occlusion
            erasing=0.3,     # NEW: Random erasing (simulates debris/occlusion)
            
            # Class-specific settings
            cls=0.5,         # Class loss weight (increased for better classification)
            box=7.5,         # Box loss weight
            dfl=1.5,         # DFL loss weight
            
            # Other settings
            auto_augment='randaugment',  # Automatic augmentation policy
            close_mosaic=10,  # Disable mosaic in last 10 epochs for fine-tuning
        )
        
        print("\n" + "-" * 70)
        print("✅ Training completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        return None
    
    # Save results summary
    print(f"\n[5/5] Saving results...")
    
    run_dir = Path(f'runs/detect/{project_name}_v2')
    
    # Copy important files to reports directory
    reports_dir = Path('reports')
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    if (run_dir / 'results.csv').exists():
        shutil.copy(run_dir / 'results.csv', reports_dir / 'training_results_v2.csv')
        print(f"  ✅ Saved: reports/training_results_v2.csv")
    
    if (run_dir / 'confusion_matrix.png').exists():
        shutil.copy(run_dir / 'confusion_matrix.png', reports_dir / 'confusion_matrix_v2.png')
        print(f"  ✅ Saved: reports/confusion_matrix_v2.png")
    
    if (run_dir / 'results.png').exists():
        shutil.copy(run_dir / 'results.png', reports_dir / 'training_curves_v2.png')
        print(f"  ✅ Saved: reports/training_curves_v2.png")
    
    # Print summary
    print("\n" + "=" * 70)
    print("TRAINING V2 COMPLETE")
    print("=" * 70)
    print("\n🎯 Key Improvements in V2:")
    print("  • Stronger color augmentation (rust/corrosion handling)")
    print("  • Random erasing (debris/occlusion simulation)")
    print("  • Copy-paste augmentation (small object detection)")
    print("  • Longer training (100 epochs vs 50)")
    print("  • Better optimizer (AdamW)")
    
    print("\n📊 Expected Improvements:")
    print("  • Better recall on corroded fasteners")
    print("  • More robust to debris/vegetation")
    print("  • Improved detection of small/occluded objects")
    
    print("\n📁 Model saved to:")
    print(f"  {run_dir / 'weights' / 'best.pt'}")
    
    print("\n🚀 Test with:")
    print(f"  python src/infer.py --model {run_dir / 'weights' / 'best.pt'} --image <test_image>")
    print("=" * 70)
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Train YOLO11 V2 with enhanced augmentation')
    
    parser.add_argument('--data', type=str, required=True, help='Path to data.yaml')
    parser.add_argument('--model', type=str, default='yolo11s.pt', help='Model size (yolo11s.pt recommended)')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs (default: 100)')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size (default: 640)')
    parser.add_argument('--batch', type=int, default=8, help='Batch size (default: 8)')
    parser.add_argument('--patience', type=int, default=20, help='Early stopping patience (default: 20)')
    parser.add_argument('--name', type=str, default='railway_defect', help='Project name')
    
    args = parser.parse_args()
    
    train_model_v2(
        data_yaml=args.data,
        model_name=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        project_name=args.name
    )


if __name__ == "__main__":
    main()