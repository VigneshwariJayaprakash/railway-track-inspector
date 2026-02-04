"""
Quick Test Script - Verify YOLO11 Installation
-----------------------------------------------
This script:
1. Downloads a pre-trained YOLO11n model
2. Runs inference on a sample image
3. Saves annotated output to verify everything works

Run this FIRST before proceeding with the project.
"""

from ultralytics import YOLO
import cv2
import os
from pathlib import Path

def main():
    print("=" * 60)
    print("YOLO11 Quick Test - Verifying Installation")
    print("=" * 60)
    
    # Create output directory
    output_dir = Path("artifacts/sample_outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n[1/4] Loading YOLO11 nano model...")
    try:
        model = YOLO('yolo11n.pt')  # This will auto-download if not present
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    print("\n[2/4] Running inference on sample image...")
    try:
        # Use Ultralytics sample image
        results = model('https://ultralytics.com/images/bus.jpg')
        print("✅ Inference completed")
    except Exception as e:
        print(f"❌ Error during inference: {e}")
        return
    
    print("\n[3/4] Processing results...")
    try:
        # Get the annotated image
        annotated_img = results[0].plot()
        
        # Count detections
        num_detections = len(results[0].boxes)
        print(f"✅ Found {num_detections} objects")
        
        # Print detected classes
        if num_detections > 0:
            class_names = results[0].names
            detected_classes = [class_names[int(box.cls)] for box in results[0].boxes]
            print(f"   Detected: {', '.join(set(detected_classes))}")
        
    except Exception as e:
        print(f"❌ Error processing results: {e}")
        return
    
    print("\n[4/4] Saving output image...")
    try:
        output_path = output_dir / "quick_test_output.jpg"
        cv2.imwrite(str(output_path), annotated_img)
        print(f"✅ Saved to: {output_path}")
    except Exception as e:
        print(f"❌ Error saving image: {e}")
        return
    
    print("\n" + "=" * 60)
    print("✅ SUCCESS! YOLO11 is working correctly")
    print("=" * 60)
    print(f"\nCheck your output at: {output_path.absolute()}")
    print("\nYou're ready to proceed with the railway detection project!")
    print("=" * 60)

if __name__ == "__main__":
    main()
