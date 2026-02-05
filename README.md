# 🚂 Railway Track Inspector

![Status](https://img.shields.io/badge/Status-Complete-success)
![Model](https://img.shields.io/badge/Model-YOLO11s-blue)
![mAP](https://img.shields.io/badge/mAP@50-48.2%25-orange)

**AI-Powered Railway Track Defect Detection System using YOLO11**

An end-to-end machine learning system that detects railway track defects (missing fasteners, cracks, damaged components) using deep learning object detection with a three-tier confidence system for safety-critical applications.

---

## 📊 Project Overview

This project implements a complete ML pipeline for railway track inspection:

- **Input:** Railway track images or video
- **Processing:** YOLO11s-based object detection with transfer learning
- **Output:** Annotated images + safety decision ("Safe" / "Review" / "Needs Inspection" / "Critical")

### Key Features

- ✅ Real-time defect detection
- ✅ Three-tier confidence system (High/Medium/Low)
- ✅ Interactive Streamlit dashboard
- ✅ Human-in-the-loop verification workflow
- ✅ Handles occlusion and rust detection
- ✅ Exportable inspection reports

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10
- Conda (recommended) or pip
- Git

### Installation
```bash
# Clone the repository
git clone https://github.com/VigneshwariJayaprakash/railway-track-inspector.git
cd railway-track-inspector

# Create environment
conda create -n railway-vision python=3.10 -y
conda activate railway-vision

# Install dependencies
pip install -r requirements.txt
```

### Dataset Setup

1. Download railway defect dataset from [Roboflow Universe](https://universe.roboflow.com/)
2. Export in **YOLO (Ultralytics)** format
3. Unzip into `data/roboflow_dataset/`

See [data/README.md](data/README.md) for detailed instructions.

### Run the Dashboard
```bash
streamlit run app/app.py
```

The dashboard will open in your browser at `http://localhost:8501`

### Run Inference on Test Image
```bash
python src/infer.py \
  --model runs/detect/railway_defect_v2/weights/best.pt \
  --image data/roboflow_dataset/test/images/<image_name>.jpg \
  --conf 0.15
```

---

## 📈 Performance

### Model V2 (Final)

| Metric | V1 (Baseline) | V2 (Enhanced) | Improvement |
|--------|---------------|---------------|-------------|
| **mAP@50** | 42.6% | **48.2%** | **+5.6 pts** ✅ |
| **Recall** | 41.3% | **51.1%** | **+9.8 pts** ✅ |
| **Precision** | 60.1% | 39.9% | -20.2 pts* |

*Trade-off accepted: Prioritize recall (don't miss defects) over precision (false positives verified by humans)

### Per-Class Performance

| Class | mAP@50 | Status |
|-------|--------|--------|
| Missing Fastener | **65.5%** | ✅ Best (safety-critical) |
| Non Defective Fishplate | 47.9% | ✅ Good |
| Defective Fishplate | 29.1% | ⚠️ Challenging |
| Fastener | 27.1% | ⚠️ Needs improvement |

---

## 🧪 Real-World Test Results

Tested on 6 challenging images with various conditions:

| Image | Condition | V1 Result | V2 Result | Status |
|-------|-----------|-----------|-----------|--------|
| 1 | Clean track | Correct | ✅ Correct | Maintained |
| 2 | Normal components | Correct | ✅ Correct | Maintained |
| 3 | **Debris occlusion** | ❌ Missed | ✅ **2 detections** | **Major improvement** |
| 4 | **Moderate rust** | ⚠️ Struggled | ✅ **1 detection** | **Improved** |
| 5 | **Severe corrosion** | ❌ Missed | ⚠️ Low confidence (16%) | Edge case |
| 6 | Hook fasteners | ⚠️ Confused | ✅ Correct | **Improved** |

**Key Insight:** Three-tier confidence system (conf=0.15) enables detection of extreme cases while maintaining quality.

See [detailed analysis](reports/v1_vs_v2_comparison.md) for complete test results.

---

## 🏗️ Project Structure
```
railway-track-inspector/
├── app/                       # Streamlit dashboard
│   └── app.py                 # Main web interface
├── src/                       # Core ML modules
│   ├── train.py               # Model training (V1)
│   ├── train_v2.py            # Enhanced training (V2)
│   ├── infer.py               # Inference engine
│   ├── decision.py            # Three-tier decision logic
│   └── dataset_check.py       # Dataset validation
├── reports/                   # Analysis & documentation
│   ├── error_analysis.md      # Failure mode analysis
│   ├── v1_vs_v2_comparison.md # Model comparison
│   └── training_results_v2.csv # Training metrics
├── artifacts/                 # Sample outputs
│   ├── screenshots/           # Dashboard screenshots
│   └── sample_outputs/        # Detection examples
├── data/                      # Dataset (local only)
│   └── README.md              # Dataset setup guide
├── requirements.txt           # Dependencies
└── README.md                  # This file
```

---

## 🎯 Three-Tier Confidence System

Our safety-critical decision system categorizes detections by confidence:
```python
HIGH (≥40%):    Immediate inspection required
MEDIUM (25-40%): Schedule within 48 hours  
LOW (15-25%):   Human review recommended
```

**Why this matters:**
- **High confidence** → Immediate action (train could be at risk)
- **Medium confidence** → Standard inspection workflow
- **Low confidence** → Expert review (may be extreme condition like heavy corrosion)

This approach prioritizes **safety** (don't miss defects) over **efficiency** (some false positives).

---

## 🔧 Technical Implementation

### Architecture
```
Dataset (321 images, 4 classes)
  ↓
YOLO11s Transfer Learning (5M parameters)
  ↓
Enhanced Augmentation (rust, occlusion, color jitter)
  ↓
Three-Tier Decision Logic
  ↓
Streamlit Dashboard (Human verification)
```

### Key Technologies

- **Model:** YOLO11s (Ultralytics)
- **Framework:** PyTorch
- **Frontend:** Streamlit
- **Augmentation:** HSV shifts, random erasing, copy-paste
- **Deployment:** Human-in-the-loop workflow

### Training Improvements (V1 → V2)

| Aspect | V1 | V2 |
|--------|----|----|
| Model size | YOLO11n (2.6M) | YOLO11s (5M) |
| Epochs | 50 (stopped at 25) | 100 (stopped at 63) |
| HSV augmentation | Moderate (0.015, 0.7, 0.4) | Strong (0.03, 0.9, 0.6) |
| Occlusion handling | None | Random erasing (0.3) |
| Small objects | None | Copy-paste (0.3) |

---

## 💡 Key Learnings

### What Worked Well

1. ✅ **Transfer learning** - Pre-trained YOLO11 provided strong baseline
2. ✅ **Enhanced augmentation** - Rust simulation and occlusion handling improved robustness
3. ✅ **Three-tier confidence** - Captures edge cases without overwhelming false positives
4. ✅ **Iterative improvement** - V1 → V2 showed measurable gains (+5.6 mAP, +9.8 recall)

### Challenges Encountered

1. ⚠️ **Small dataset** (321 images) - Limited generalization
2. ⚠️ **Class confusion** - Fastener vs defective fishplate visually similar
3. ⚠️ **Extreme corrosion** - Heavy rust masks component appearance
4. ⚠️ **CPU training time** - 5+ hours (GPU would be 30-45 min)

### Solutions Implemented

1. ✅ Aggressive data augmentation
2. ✅ Larger model (YOLO11s)
3. ✅ Confidence threshold tuning (0.15 for safety)
4. ✅ Comprehensive documentation of limitations

---

## 🚀 Production Readiness

### Current Status

**✅ Ready for Human-Assisted Deployment**

### Suitable For:

- ✅ Pre-screening tool (filters safe tracks)
- ✅ Alert system with human verification
- ✅ Training tool for inspectors
- ✅ Data collection for model improvement

### NOT Suitable For:

- ❌ Fully autonomous safety decisions
- ❌ Regulatory compliance without oversight
- ❌ Unsupervised operations

### Deployment Recommendations

**Workflow:**
1. AI scans images at conf=0.15
2. System categorizes by confidence tier
3. High/Medium → Inspector reviews
4. Low confidence → Expert review
5. Feedback loop → Model retraining

---

## 📚 Future Improvements

### Short-Term (1-2 months)

- [ ] Expand dataset to 500+ images per class
- [ ] Implement ensemble methods
- [ ] Add class-specific confidence thresholds
- [ ] Collect field deployment feedback

### Long-Term (3-6 months)

- [ ] Two-stage pipeline (detect → classify severity)
- [ ] Multi-modal fusion (RGB + IR/thermal)
- [ ] Active learning with hard examples
- [ ] Edge deployment (real-time on vehicles)

---

## 🎓 Academic Alignment

### Connects to Andrew Ng's Deep Learning Concepts:

- **Transfer Learning:** Fine-tuning pre-trained YOLO
- **Data Augmentation:** Expanding limited dataset
- **Bias-Variance Trade-off:** Model capacity vs performance
- **Precision/Recall Trade-off:** Safety-critical tuning
- **Iterative Development:** Baseline → Enhanced model

---

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **YOLO11:** Ultralytics team
- **Dataset:** Roboflow community
- **Inspiration:** Railway safety initiatives (BNSF, Norfolk Southern)

---

## 📧 Contact

**Vigneshwari Jayaprakash**

- Email: vjayapr1@asu.edu
- GitHub: [@VigneshwariJayaprakash](https://github.com/VigneshwariJayaprakash)
- LinkedIn: [Add your LinkedIn URL]

---

## 📊 Project Milestones

- [x] Week 1: Setup & dataset validation ✅
- [x] Week 2: Model V1 training (baseline) ✅
- [x] Week 3: Model V2 training (enhanced) ✅
- [x] Week 4: Dashboard & documentation ✅
- [ ] Deployment on Streamlit Cloud (optional)

---

**Last Updated:** February 2026  
**Status:** ✅ Complete  
**Version:** 2.0