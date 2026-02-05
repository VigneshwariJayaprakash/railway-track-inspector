"""
YOLO11 Training Script
-----------------------
Trains YOLO11 model on railway track defect dataset.

Usage:
    python src/train.py --data data/roboflow_dataset/data.yaml --epochs 50

This script:
1. Loads pre-trained YOLO11n model (transfer learning)
2. Fine-tunes on railway defect dataset
3. Saves training results and best model weights
4. Generates performance metrics and visualizations
"""

import argparse
from ultralytics import YOLO
from pathlib import Path
import shutil
import yaml


def train_model(
    data_yaml,
    model_name='yolo11n.pt',
    epochs=50,
    imgsz=640,
    batch=16,
    patience=10,
    project_name='railway_defect'
):
    """
    Train YOLO11 model on railway defect dataset.
    
    Args:
        data_yaml: Path to data.yaml file
        model_name: Pre-trained model to use (yolo11n.pt, yolo11s.pt, etc.)
        epochs: Number of training epochs
        imgsz: Input image size
        batch: Batch size
        patience: Early stopping patience
        project_name: Name for this training run
    """
    
    print("=" * 70)
    print("YOLO11 TRAINING - Railway Track Defect Detection")
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
    print("  This uses transfer learning - starting from a model trained on")
    print("  general objects and fine-tuning it for railway defects.")
    
    model = YOLO(model_name)
    print("  ✅ Model loaded successfully")
    
    # Training configuration
    print(f"\n[3/5] Training Configuration:")
    print(f"  Epochs: {epochs}")
    print(f"  Image size: {imgsz}x{imgsz}")
    print(f"  Batch size: {batch}")
    print(f"  Patience: {patience} epochs (early stopping)")
    print(f"  Device: GPU if available, else CPU")
    
    # Start training
    print(f"\n[4/5] Starting training...")
    print("  This will take some time. Progress will be shown below.")
    print("  Training metrics will be saved to: runs/detect/{project_name}_v1/")
    print("-" * 70)
    
    try:
        results = model.train(
            data=str(data_yaml),
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            patience=patience,
            name=f'{project_name}_v1',
            pretrained=True,
            optimizer='auto',
            verbose=True,
            seed=42,
            deterministic=True,
            save=True,
            save_period=-1,  # Save only best and last
            plots=True,
            # Data augmentation (built into YOLO)
            hsv_h=0.015,  # Hue augmentation
            hsv_s=0.7,    # Saturation augmentation
            hsv_v=0.4,    # Value augmentation
            degrees=0.0,  # Rotation (0 for railway tracks - they're usually horizontal)
            translate=0.1, # Translation augmentation
            scale=0.5,    # Scale augmentation
            shear=0.0,    # Shear augmentation (0 for straight tracks)
            perspective=0.0, # Perspective augmentation
            flipud=0.0,   # Vertical flip (0 - tracks don't flip vertically)
            fliplr=0.5,   # Horizontal flip (0.5 probability)
            mosaic=1.0,   # Mosaic augmentation
            mixup=0.0,    # MixUp augmentation
        )
        
        print("\n" + "-" * 70)
        print("✅ Training completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        return None
    
    # Save results summary
    print(f"\n[5/5] Saving results...")
    
    run_dir = Path(f'runs/detect/{project_name}_v1')
    
    # Copy important files to reports directory
    reports_dir = Path('reports')
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy results.csv
    if (run_dir / 'results.csv').exists():
        shutil.copy(run_dir / 'results.csv', reports_dir / 'training_results.csv')
        print(f"  ✅ Saved: reports/training_results.csv")
    
    # Copy confusion matrix
    if (run_dir / 'confusion_matrix.png').exists():
        shutil.copy(run_dir / 'confusion_matrix.png', reports_dir / 'confusion_matrix.png')
        print(f"  ✅ Saved: reports/confusion_matrix.png")
    
    # Copy training curves
    if (run_dir / 'results.png').exists():
        shutil.copy(run_dir / 'results.png', reports_dir / 'training_curves.png')
        print(f"  ✅ Saved: reports/training_curves.png")
    
    # Copy a few validation predictions
    val_batch_dir = run_dir
    val_images = list(val_batch_dir.glob('val_batch*_pred.jpg'))
    if val_images:
        sample_outputs_dir = Path('artifacts/sample_outputs')
        sample_outputs_dir.mkdir(parents=True, exist_ok=True)
        
        for idx, img_path in enumerate(val_images[:3]):  # Copy first 3
            dest = sample_outputs_dir / f'validation_pred_{idx+1}.jpg'
            shutil.copy(img_path, dest)
            print(f"  ✅ Saved: {dest}")
    
    # Print summary
    print("\n" + "=" * 70)
    print("TRAINING SUMMARY")
    print("=" * 70)
    
    # Read final metrics from results.csv
    try:
        import pandas as pd
        results_df = pd.read_csv(run_dir / 'results.csv')
        results_df.columns = results_df.columns.str.strip()  # Remove whitespace
        
        final_metrics = results_df.iloc[-1]
        
        print("\nFinal Metrics:")
        print(f"  mAP@50:       {final_metrics.get('metrics/mAP50(B)', 'N/A'):.4f}")
        print(f"  mAP@50-95:    {final_metrics.get('metrics/mAP50-95(B)', 'N/A'):.4f}")
        print(f"  Precision:    {final_metrics.get('metrics/precision(B)', 'N/A'):.4f}")
        print(f"  Recall:       {final_metrics.get('metrics/recall(B)', 'N/A'):.4f}")
        
        print("\n📊 Metrics Interpretation:")
        print("  • mAP@50:     Box accuracy (target: >0.6 for good performance)")
        print("  • Recall:     Defect detection rate (target: >0.8 for safety)")
        print("  • Precision:  False alarm rate (target: >0.7)")
        
        # Provide guidance based on results
        mAP50 = final_metrics.get('metrics/mAP50(B)', 0)
        recall = final_metrics.get('metrics/recall(B)', 0)
        
        print("\n💡 Recommendations:")
        if mAP50 < 0.4:
            print("  ⚠️  Low mAP - Consider:")
            print("     - More training epochs")
            print("     - Better quality training data")
            print("     - Check if labels are correct")
        elif mAP50 < 0.6:
            print("  ⚠️  Moderate mAP - Consider:")
            print("     - More training data")
            print("     - Longer training (more epochs)")
        else:
            print("  ✅ Good mAP! Model is learning well.")
        
        if recall < 0.6:
            print("  ⚠️  Low recall - Model is missing defects!")
            print("     - This is critical for safety applications")
            print("     - Consider adjusting confidence threshold lower")
        elif recall < 0.8:
            print("  ⚠️  Moderate recall - Room for improvement")
        else:
            print("  ✅ Good recall! Model detects most defects.")
        
    except Exception as e:
        print(f"  Could not read metrics: {e}")
    
    print("\n" + "=" * 70)
    print("📁 OUTPUT LOCATIONS:")
    print("-" * 70)
    print(f"  Model weights:      {run_dir / 'weights' / 'best.pt'}")
    print(f"  Training logs:      {run_dir / 'results.csv'}")
    print(f"  Visualizations:     {run_dir}")
    print(f"  Reports:            reports/")
    print("=" * 70)
    
    print("\n🚀 NEXT STEPS:")
    print("  1. Review training curves in: reports/training_curves.png")
    print("  2. Check confusion matrix: reports/confusion_matrix.png")
    print("  3. Test inference:")
    print(f"     python src/infer.py --model {run_dir / 'weights' / 'best.pt'} --image <test_image>")
    print("  4. If results are good, proceed to building the Streamlit app!")
    print("=" * 70)
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Train YOLO11 on railway defect dataset')
    
    parser.add_argument(
        '--data',
        type=str,
        required=True,
        help='Path to data.yaml file'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='yolo11n.pt',
        help='Pre-trained model (yolo11n.pt, yolo11s.pt, yolo11m.pt, etc.)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='Number of training epochs (default: 50)'
    )
    parser.add_argument(
        '--imgsz',
        type=int,
        default=640,
        help='Input image size (default: 640)'
    )
    parser.add_argument(
        '--batch',
        type=int,
        default=16,
        help='Batch size (reduce if out of memory, default: 16)'
    )
    parser.add_argument(
        '--patience',
        type=int,
        default=10,
        help='Early stopping patience (default: 10)'
    )
    parser.add_argument(
        '--name',
        type=str,
        default='railway_defect',
        help='Project name (default: railway_defect)'
    )
    
    args = parser.parse_args()
    
    # Train model
    train_model(
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