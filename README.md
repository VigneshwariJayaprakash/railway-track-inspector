# Dataset Setup Instructions

This folder contains the railway defect detection dataset. **It is NOT committed to GitHub** due to size constraints.

---

## 📥 Download Dataset

### Option 1: Roboflow Universe (Recommended)

1. Visit [Roboflow Universe](https://universe.roboflow.com/)
2. Search for one of these datasets:
   - "railway defect detection"
   - "rail fastener detection"
   - "railroad track inspection"

3. **Recommended Dataset Characteristics:**
   - At least 500+ images
   - Classes: `missing_fastener`, `crack`, `defective_fishplate`, or similar
   - Already annotated with bounding boxes
   - Available in YOLO format

4. Click "Fork Dataset" to add it to your account

5. Go to **Versions** → **Create New Version**

6. Add augmentations (recommended):
   - Horizontal flip
   - Brightness: -15% to +15%
   - Rotation: ±10°
   - Blur: up to 1.5px
   - Noise: up to 2%

7. Click **Export** → Select **YOLO (Ultralytics)** format

8. Download the ZIP file

---

## 📂 Expected Folder Structure

After downloading and unzipping, your `data/` folder should look like this:

```
data/
├── README.md                 # This file
└── roboflow_dataset/         # Unzip your download here
    ├── train/
    │   ├── images/           # Training images (.jpg)
    │   └── labels/           # YOLO format labels (.txt)
    ├── valid/
    │   ├── images/           # Validation images
    │   └── labels/           # Validation labels
    ├── test/
    │   ├── images/           # Test images
    │   └── labels/           # Test labels
    └── data.yaml             # Dataset configuration
```

---

## ✅ Validate Your Dataset

After setting up, run the validation script:

```bash
python src/dataset_check.py --data data/roboflow_dataset/data.yaml
```

This will:
- ✅ Check file structure
- ✅ Count images and labels
- ✅ Verify class distributions
- ✅ Generate preview images with bounding boxes

---

## 📊 Understanding `data.yaml`

The `data.yaml` file tells YOLO where to find your data:

```yaml
train: ../train/images
val: ../valid/images
test: ../test/images

nc: 3  # Number of classes
names: ['missing_fastener', 'crack', 'defective_fishplate']
```

**Important:** Make sure the paths in `data.yaml` are correct relative to where you run your training script.

---

## 🔍 Example Datasets on Roboflow

Here are some publicly available options (as of Feb 2026):

1. **Rail Defect Detection** - ~1,000 images
   - Classes: missing_fastener, crack, defective_rail
   
2. **Railway Component Detection** - ~800 images
   - Classes: bolt, fastener, fishplate, crack

3. **Track Inspection Dataset** - ~1,200 images
   - Classes: missing_component, surface_defect, geometric_defect

*(Search current Roboflow Universe for the latest/best options)*

---

## 🚨 Troubleshooting

### "Data path not found"
- Make sure you've unzipped the dataset into `data/roboflow_dataset/`
- Check that `data.yaml` exists in the root of the dataset folder

### "No labels found"
- Verify the `labels/` folders contain `.txt` files
- Each image should have a corresponding label file with the same name

### "Class mismatch"
- Open `data.yaml` and verify the `nc` (number of classes) matches your dataset
- Check that class names are spelled correctly

---

## 📝 Notes

- **Dataset is local only** - Never commit the dataset to GitHub
- The `.gitignore` file automatically excludes this folder
- If sharing your project, direct people to this README for dataset setup
- Consider data licensing when using public datasets

---

## 🎯 Next Steps

Once your dataset is set up and validated:

1. Run `python src/dataset_check.py` to verify
2. Proceed to training with `python src/train.py`
3. Check `reports/` for evaluation metrics

---

**Need help?** Open an issue in the GitHub repository.
