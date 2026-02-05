"""
Inference Script - Run YOLO11 Model on New Images
--------------------------------------------------
Run trained model on new railway track images.

Usage:
    # Single image
    python src/infer.py --model runs/detect/railway_defect_v2/weights/best.pt --image test.jpg
    
    # With custom confidence threshold
    python src/infer.py --model runs/detect/railway_defect_v2/weights/best.pt --image test.jpg --conf 0.25
    
    # Save output
    python src/infer.py --model runs/detect/railway_defect_v2/weights/best.pt --image test.jpg --output result.jpg
"""

import argparse
from pathlib import Path
from ultralytics import YOLO
import cv2
import json


def run_inference(model_path, image_path, conf_threshold=0.25, output_path=None):
    """
    Run inference on a single image.
    
    Args:
        model_path: Path to trained model weights (.pt file)
        image_path: Path to input image
        conf_threshold: Confidence threshold (0-1)
        output_path: Optional path to save annotated image
        
    Returns:
        dict: Detection results with boxes, classes, and confidences
    """
    
    print("=" * 70)
    print("RAILWAY DEFECT DETECTION - INFERENCE")
    print("=" * 70)
    
    # Load model
    print(f"\n[1/4] Loading model: {model_path}")
    model = YOLO(model_path)
    print("  ✅ Model loaded successfully")
    
    # Load image
    print(f"\n[2/4] Loading image: {image_path}")
    img_path = Path(image_path)
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    img = cv2.imread(str(img_path))
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    img_height, img_width = img.shape[:2]
    print(f"  ✅ Image loaded: {img_width}x{img_height}")
    
    # Run inference
    print(f"\n[3/4] Running inference (conf threshold: {conf_threshold})...")
    results = model.predict(
        source=str(img_path),
        conf=conf_threshold,
        verbose=False
    )
    
    result = results[0]
    
    # Extract detections
    detections = []
    boxes = result.boxes
    
    for box in boxes:
        detection = {
            'class_id': int(box.cls[0]),
            'class_name': result.names[int(box.cls[0])],
            'confidence': float(box.conf[0]),
            'bbox': {
                'x1': float(box.xyxy[0][0]),
                'y1': float(box.xyxy[0][1]),
                'x2': float(box.xyxy[0][2]),
                'y2': float(box.xyxy[0][3])
            }
        }
        detections.append(detection)
    
    num_detections = len(detections)
    print(f"  ✅ Found {num_detections} defect(s)")
    
    # Print detections
    if num_detections > 0:
        print("\n  Detections:")
        print("  " + "-" * 66)
        print(f"  {'Class':<30} {'Confidence':<15} {'Location'}")
        print("  " + "-" * 66)
        
        for det in detections:
            bbox = det['bbox']
            location = f"({bbox['x1']:.0f}, {bbox['y1']:.0f})"
            print(f"  {det['class_name']:<30} {det['confidence']:.2%}          {location}")
        
        print("  " + "-" * 66)
    
    # Save output
    print(f"\n[4/4] Saving results...")
    
    # Get annotated image
    annotated_img = result.plot()
    
    # Determine output path
    if output_path is None:
        output_dir = Path("artifacts/sample_outputs")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"inference_{img_path.stem}_result.jpg"
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save annotated image
    cv2.imwrite(str(output_path), annotated_img)
    print(f"  ✅ Annotated image saved to: {output_path}")
    
    # Save JSON summary
    json_path = output_path.parent / f"{output_path.stem}.json"
    summary = {
        'image': str(img_path),
        'model': str(model_path),
        'confidence_threshold': conf_threshold,
        'num_detections': num_detections,
        'detections': detections
    }
    
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"  ✅ JSON summary saved to: {json_path}")
    
    # Summary
    print("\n" + "=" * 70)
    print("INFERENCE COMPLETE")
    print("=" * 70)
    print(f"Image: {img_path.name}")
    print(f"Detections: {num_detections}")
    
    if num_detections > 0:
        print(f"\nDetected classes:")
        class_counts = {}
        for det in detections:
            class_name = det['class_name']
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        for class_name, count in class_counts.items():
            print(f"  • {class_name}: {count}")
    
    print("=" * 70)
    
    return summary


def main():
    parser = argparse.ArgumentParser(description='Run inference on railway track images')
    
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to trained model weights (.pt file)'
    )
    parser.add_argument(
        '--image',
        type=str,
        required=True,
        help='Path to input image'
    )
    parser.add_argument(
        '--conf',
        type=float,
        default=0.25,
        help='Confidence threshold (0-1, default: 0.25)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Path to save annotated output image (optional)'
    )
    
    args = parser.parse_args()
    
    # Run inference
    run_inference(
        model_path=args.model,
        image_path=args.image,
        conf_threshold=args.conf,
        output_path=args.output
    )


if __name__ == "__main__":
    main()