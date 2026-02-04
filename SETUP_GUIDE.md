# 🚀 IMMEDIATE SETUP GUIDE
## Railway Track Inspector - Week 1 Execution Plan

---

## ✅ STEP 1: Set Up Your Local Repository

### 1.1 Navigate to your projects folder

```bash
# Windows
cd C:\Users\YourName\Documents

# Mac/Linux
cd ~/Documents
```

### 1.2 Clone your repository

```bash
git clone https://github.com/VigneshwariJayaprakash/railway-track-inspector.git
cd railway-track-inspector
```

---

## ✅ STEP 2: Create Folder Structure

### Windows (Command Prompt or Anaconda Prompt):

```bash
mkdir app
mkdir src
mkdir scripts
mkdir artifacts\screenshots
mkdir artifacts\sample_outputs
mkdir reports
mkdir data
```

### Mac/Linux:

```bash
mkdir -p app src scripts artifacts/screenshots artifacts/sample_outputs reports data
```

### Verify it worked:

```bash
dir       # Windows
ls -la    # Mac/Linux
```

You should see these folders:
- app/
- src/
- scripts/
- artifacts/
- reports/
- data/
- README.md

---

## ✅ STEP 3: Add Project Files

Copy these files from the downloads I provided into your local repository:

1. **scripts/quick_test.py** → Put in `scripts/` folder
2. **README.md** → Replace the existing one
3. **.gitignore** → Put in root directory
4. **requirements.txt** → Put in root directory
5. **data/README.md** → Put in `data/` folder
6. **artifacts/screenshots/.gitkeep** → Put in `artifacts/screenshots/`
7. **artifacts/sample_outputs/.gitkeep** → Put in `artifacts/sample_outputs/`

---

## ✅ STEP 4: Set Up Python Environment

### 4.1 Create Conda Environment

```bash
conda create -n railway-vision python=3.10 -y
```

### 4.2 Activate Environment

```bash
conda activate railway-vision
```

You should see `(railway-vision)` at the start of your command line.

### 4.3 Install Dependencies

```bash
pip install ultralytics streamlit opencv-python pillow pandas numpy matplotlib
```

This will take 2-3 minutes. Wait for it to complete.

### 4.4 Verify Installation

```bash
python -c "from ultralytics import YOLO; print('✅ Ultralytics installed successfully')"
```

---

## ✅ STEP 5: Run Your First Test

### 5.1 Run the quick test

```bash
python scripts/quick_test.py
```

### 5.2 What you should see:

```
============================================================
YOLO11 Quick Test - Verifying Installation
============================================================

[1/4] Loading YOLO11 nano model...
✅ Model loaded successfully

[2/4] Running inference on sample image...
✅ Inference completed

[3/4] Processing results...
✅ Found 4 objects
   Detected: person, bus

[4/4] Saving output image...
✅ Saved to: artifacts/sample_outputs/quick_test_output.jpg

============================================================
✅ SUCCESS! YOLO11 is working correctly
============================================================
```

### 5.3 Check the output

Open the file:
```
artifacts/sample_outputs/quick_test_output.jpg
```

You should see a bus with bounding boxes around detected objects!

---

## ✅ STEP 6: Commit Your Initial Setup

```bash
git add .
git commit -m "Initial project setup - Week 1"
git push origin main
```

### Verify on GitHub:

Go to: https://github.com/VigneshwariJayaprakash/railway-track-inspector

You should see:
- ✅ All your folders
- ✅ README.md with project description
- ✅ requirements.txt
- ✅ .gitignore

**Note:** The `data/` folder and `artifacts/sample_outputs/quick_test_output.jpg` will NOT appear on GitHub (they're in .gitignore). This is correct!

---

## ✅ STEP 7: Download Your Dataset

### 7.1 Go to Roboflow Universe

Visit: https://universe.roboflow.com/

### 7.2 Search for dataset

In the search bar, type:
- "railway defect detection"
- OR "rail fastener detection"

### 7.3 Choose a dataset

Look for one with:
- ✅ At least 500+ images
- ✅ Classes like: `missing_fastener`, `crack`, `defective_fishplate`
- ✅ Clear bounding box annotations

### 7.4 Fork the dataset

Click the "Fork Dataset" button (you'll need to create a free Roboflow account)

### 7.5 Export and Download

1. Click "Generate" (use default augmentations for now)
2. Click "Export"
3. Choose format: **YOLO (Ultralytics)**
4. Click "Download ZIP"

### 7.6 Unzip into your project

Unzip the downloaded file into:
```
railway-track-inspector/data/roboflow_dataset/
```

Your structure should be:
```
data/
└── roboflow_dataset/
    ├── train/
    ├── valid/
    ├── test/
    └── data.yaml
```

---

## ✅ CHECKPOINT: Week 1 Complete!

At this point, you should have:

- [x] GitHub repository created
- [x] Local folder structure set up
- [x] Python environment configured
- [x] `quick_test.py` ran successfully
- [x] Initial commit pushed to GitHub
- [x] Dataset downloaded from Roboflow

---

## 🎯 NEXT STEPS (Week 2)

In the next session, I'll help you create:

1. **src/dataset_check.py** - Validate your dataset
2. **src/train.py** - Train your first model
3. **src/utils.py** - Helper functions

But for now, **STOP HERE** and verify everything above works.

---

## 🚨 Troubleshooting

### "conda: command not found"
- Make sure Anaconda/Miniconda is installed
- Windows: Use "Anaconda Prompt" not regular Command Prompt

### "ModuleNotFoundError: No module named 'ultralytics'"
- Make sure you activated the environment: `conda activate railway-vision`
- Re-run: `pip install ultralytics`

### "Git push rejected"
- Run: `git pull origin main` first
- Then: `git push origin main`

### "quick_test.py fails with GPU error"
- This is OK - YOLO will automatically use CPU
- It will be slower but will still work

---

## 📧 Need Help?

If you get stuck:

1. Check the exact error message
2. Make sure you're in the correct directory
3. Verify your conda environment is activated
4. Try the command again

---

**Reply with:** "Week 1 setup complete ✅" when you've finished all steps above!
