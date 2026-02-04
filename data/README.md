# 🚂 Railway Track Inspector

**AI-Powered Defect Detection System using YOLO11**

A computer vision system that detects railway track defects (missing fasteners, cracks, damaged components) using deep learning object detection.

---

## 📋 Project Overview

This project implements an end-to-end machine vision pipeline for railway track inspection:

- **Input:** Railway track images or video
- **Processing:** YOLO11-based object detection
- **Output:** Annotated images + safety decision ("Safe" / "Needs Inspection")

### Key Features

- ✅ Real-time defect detection
- ✅ Confidence-based filtering
- ✅ Video persistence logic (N-of-M rule)
- ✅ Interactive Streamlit dashboard
- ✅ Exportable inspection reports

---

## 🏗️ Architecture

```
┌─────────────┐
│   Image     │
│  Upload     │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  YOLO11     │  ← Transfer learning on railway defects
│  Detection  │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Decision   │  ← Safe / Needs Inspection logic
│  Logic      │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Streamlit  │  ← Interactive dashboard
│  Dashboard  │
└─────────────┘
```

---

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/VigneshwariJayaprakash/railway-track-inspector.git
cd railway-track-inspector
```

### 2. Set Up Environment

```bash
# Create conda environment
conda create -n railway-vision python=3.10 -y
conda activate railway-vision

# Install dependencies
pip install -r requirements.txt
```

### 3. Verify Installation

```bash
python scripts/quick_test.py
```

✅ If this runs successfully, you're ready to proceed!

---

## 📁 Project Structure

```
railway-track-inspector/
│
├── app/                       # Streamlit web application
│   ├── app.py                 # Main dashboard
│   └── ui_helpers.py          # UI utility functions
│
├── src/                       # Core logic modules
│   ├── train.py               # Model training script
│   ├── infer.py               # Inference engine
│   ├── decision.py            # Safety decision logic
│   ├── dataset_check.py       # Dataset validation
│   └── utils.py               # Helper functions
│
├── scripts/                   # Executable utilities
│   ├── quick_test.py          # Installation verification
│   └── export_preds.py        # Export predictions
│
├── artifacts/                 # Generated outputs (for portfolio)
│   ├── screenshots/           # UI screenshots
│   └── sample_outputs/        # Sample predictions
│
├── reports/                   # Analysis documentation
│   ├── eval_baseline.md       # Model evaluation
│   ├── error_analysis.md      # Failure case analysis
│   └── model_card.md          # Model documentation
│
├── data/                      # Dataset (local only, not in git)
│   └── README.md              # Dataset setup instructions
│
├── requirements.txt           # Python dependencies
├── README.md                  # This file
└── .gitignore                 # Git ignore rules
```

---

## 📊 Dataset Setup

This project uses a railway defect detection dataset from Roboflow.

### Steps:

1. Go to [Roboflow Universe](https://universe.roboflow.com/)
2. Search for "railway defect detection" or "rail fastener detection"
3. Fork the dataset to your account
4. Export in **YOLO (Ultralytics)** format
5. Download and unzip into `data/roboflow_dataset/`

Your `data/` folder should look like:

```
data/
└── roboflow_dataset/
    ├── train/
    │   ├── images/
    │   └── labels/
    ├── valid/
    │   ├── images/
    │   └── labels/
    ├── test/
    │   ├── images/
    │   └── labels/
    └── data.yaml
```

### Validate Your Dataset:

```bash
python src/dataset_check.py --data data/roboflow_dataset/data.yaml
```

---

## 🎓 Training the Model

### Basic Training:

```bash
python src/train.py \
  --data data/roboflow_dataset/data.yaml \
  --epochs 50 \
  --imgsz 640 \
  --batch 16
```

### Training Parameters:

| Parameter | Description | Recommended Value |
|-----------|-------------|-------------------|
| `epochs` | Training iterations | 50-100 |
| `imgsz` | Input image size | 640 |
| `batch` | Batch size | 16 (reduce if OOM error) |
| `patience` | Early stopping patience | 10 |

### Output:

Training results will be saved to `runs/detect/railway_defect_v1/`:
- `weights/best.pt` - Best model checkpoint
- `results.csv` - Training metrics
- `confusion_matrix.png` - Performance visualization

---

## 🔍 Running Inference

### On a Single Image:

```bash
python src/infer.py \
  --model runs/detect/railway_defect_v1/weights/best.pt \
  --image path/to/test_image.jpg \
  --conf 0.25 \
  --out artifacts/sample_outputs/prediction.jpg
```

### Key Metrics:

- **mAP@50:** Measures box accuracy (target: >0.6)
- **Recall:** Detects all defects (target: >0.8 for safety)
- **Precision:** Avoids false alarms (target: >0.7)

---

## 🖥️ Running the Dashboard

```bash
streamlit run app/app.py
```

This opens an interactive web interface where you can:

1. Upload railway track images
2. Adjust confidence threshold
3. View annotated detections
4. Get safety recommendations
5. Export inspection reports

---

## 📈 Project Milestones

### Week 1: Setup & Validation
- [x] Environment setup
- [x] YOLO installation verification
- [ ] Dataset download and validation

### Week 2: Model Training
- [ ] Baseline model training
- [ ] Metrics evaluation
- [ ] Error analysis

### Week 3: Inference & Dashboard
- [ ] Inference module
- [ ] Streamlit app development
- [ ] Decision logic implementation

### Week 4: Polish & Deployment
- [ ] Video processing (persistence logic)
- [ ] Documentation completion
- [ ] Portfolio optimization

---

## 🎯 Key Concepts (Andrew Ng's Deep Learning)

This project demonstrates:

- **Transfer Learning:** Starting from pre-trained YOLO11n
- **Data Augmentation:** Expanding limited datasets
- **Precision/Recall Trade-off:** Prioritizing recall for safety
- **Train/Val/Test Split:** Proper evaluation methodology
- **mAP (Mean Average Precision):** Object detection metrics

---

## 📝 Model Card

### Intended Use
- Support tool for railway track inspection
- Pre-filtering system for human inspectors
- NOT for autonomous safety-critical decisions

### Limitations
- Performance depends on training data quality
- May struggle with low-light or occluded defects
- Requires human verification for critical decisions

### Ethical Considerations
- This is a decision-support tool, not a replacement for human expertise
- False negatives could pose safety risks
- Model should be regularly retrained with new data

---

## 🤝 Contributing

Contributions welcome! Areas for improvement:

- [ ] Add more defect classes
- [ ] Implement video processing
- [ ] Add model quantization for edge deployment
- [ ] Expand test coverage

---

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details

---

## 🙏 Acknowledgments

- **YOLO11:** Ultralytics team
- **Dataset:** Roboflow community
- **Inspiration:** Railway safety initiatives (BNSF, Norfolk Southern)

---

## 📧 Contact

**Vigneshwari Jayaprakash**
- GitHub: [@VigneshwariJayaprakash](https://github.com/VigneshwariJayaprakash)

---

**Status:** 🚧 In Development (Week 1/4)
