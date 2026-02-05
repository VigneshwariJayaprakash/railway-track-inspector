# Dataset Setup Guide

## Quick Setup

1. Download railway track defect dataset from [Roboflow Universe](https://universe.roboflow.com/)
2. Search for "railway defect detection" or "railway track"
3. Export in **YOLOv11** or **Ultralytics YOLO** format
4. Extract to this folder: `data/roboflow_dataset/`

---

## Expected Structure

After setup, this folder should contain:
```
data/
├── README.md                    # This file
└── roboflow_dataset/
    ├── data.yaml                # Dataset configuration
    ├── train/
    │   ├── images/              # Training images (.jpg)
    │   └── labels/              # Training labels (.txt)
    ├── valid/
    │   ├── images/              # Validation images
    │   └── labels/              # Validation labels
    └── test/
        ├── images/              # Test images
        └── labels/              # Test labels
```

---

## Dataset Used in This Project

**Source:** Roboflow Universe

**Statistics:**
- Total: 321 images
- Train: 102 images (70%)
- Valid: 31 images (23%)
- Test: 188 images (not used in training)

**Classes (4):**
1. `missing fastener` - Critical defect
2. `defective fishplate` - Serious defect
3. `fastener` - Normal component (flagged)
4. `non defective fishplate` - Normal component

---

## Verify Dataset

After downloading, verify with:
```bash
python src/dataset_check.py
```

This will check:
- ✅ Image and label counts
- ✅ YOLO format validity
- ✅ Class distribution
- ✅ Generate preview images

---

## YOLO Label Format

Each `.txt` label file contains:
```
<class_id> <x_center> <y_center> <width> <height>
```

All values are normalized (0-1).

Example:
```
0 0.5 0.5 0.1 0.1
```
- Class 0 (missing fastener)
- Center at (50%, 50%)
- Size: 10% x 10%

---

## Why Dataset is Not in GitHub

The dataset is **excluded** from version control because:
- 📦 Large size (50-100 MB)
- 📜 Licensing restrictions
- 🔒 May contain sensitive imagery

**You must download it separately.**

---

## Troubleshooting

### Error: "data.yaml not found"

**Fix:** Make sure dataset is extracted to `data/roboflow_dataset/` with `data.yaml` in the root.

### Error: "No images found"

**Fix:** Check that images are in `train/images/`, `valid/images/`, `test/images/`

### Error: "Labels missing"

**Fix:** Ensure you downloaded in **YOLO format** (not COCO or Pascal VOC)

---

## Custom Dataset

To use your own images:

1. **Label images** using [Roboflow](https://roboflow.com) or [LabelImg](https://github.com/heartexlabs/labelImg)
2. **Export in YOLO format**
3. **Create data.yaml** with paths and class names
4. **Validate** with `src/dataset_check.py`

---

## Data Augmentation

Training applies automatic augmentation:
- HSV color shifts (rust simulation)
- Random erasing (occlusion)
- Horizontal flip
- Scale variation
- Copy-paste (small objects)

See `src/train_v2.py` for details.

---

**Need help?** See main [README.md](../README.md) or open an issue.

**Last Updated:** February 2026