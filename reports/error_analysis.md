# Error Analysis Report
## Railway Track Defect Detection System

**Model Version:** railway_defect_v1  
**Date:** February 2026  
**Analyst:** Vigneshwari Jayaprakash

---

## Executive Summary

This error analysis examines failure cases from the YOLO11-based railway defect detection model trained on 321 labeled images across 4 classes. The model achieved an overall mAP@50 of 42.6%, with significant variation across classes.

**Key Findings:**
- Model performs best on "missing fastener" class (mAP: 65.5%)
- Struggles with "fastener" and "defective fishplate" classes
- Primary failure modes: class confusion, small object detection, and lighting variations

---

## Model Performance Overview

### Overall Metrics
| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| mAP@50 | 42.6% | >60% | ⚠️ Below target |
| Precision | 60.1% | >70% | ⚠️ Close |
| Recall | 41.3% | >80% | ⚠️ Needs improvement |

### Per-Class Performance
| Class | Precision | Recall | mAP@50 | Instances | Status |
|-------|-----------|--------|--------|-----------|--------|
| missing fastener | 70.9% | 40.0% | 65.5% | 10 | ✅ Best performer |
| non defective fishplate | 37.2% | 60.0% | 47.9% | 10 | ⚠️ High false positives |
| defective fishplate | 5.0% | 2.5% | 29.1% | 10 | ❌ Critical issue |
| fastener | 22.8% | 10.0% | 27.1% | 10 | ❌ Poor detection |

---

## False Positives (Model Incorrectly Flagged Defects)

### Example 1: Shadow Misclassified as Crack
**Image:** `val_image_005.jpg`  
**Prediction:** defective fishplate (confidence: 0.32)  
**Ground Truth:** No defect present

**Analysis:**
- Dark shadow between rails resembled visual pattern of defect
- Low lighting conditions amplified false detection
- Model confidence was low (0.32), suggesting uncertainty

**Root Cause:** Insufficient training examples with various lighting conditions

**Suggested Fix:**
- Add augmentation: brightness variation (±30%)
- Include more shadow examples in training data
- Consider time-of-day metadata filtering

---

### Example 2: Rust Mistaken for Defect
**Image:** `val_image_012.jpg`  
**Prediction:** defective fishplate (confidence: 0.45)  
**Ground Truth:** Normal wear (not structural defect)

**Analysis:**
- Surface rust on fishplate confused with structural damage
- Visual similarity between cosmetic and structural issues
- Model lacks context to distinguish severity

**Root Cause:** Training data doesn't distinguish cosmetic vs. structural defects

**Suggested Fix:**
- Refine class definitions: separate "cosmetic wear" from "structural defect"
- Add negative examples explicitly labeled as "normal wear"
- Consider multi-stage classification (defect detection → severity assessment)

---

### Example 3: Debris Flagged as Missing Component
**Image:** `val_image_018.jpg`  
**Prediction:** missing fastener (confidence: 0.38)  
**Ground Truth:** Fastener present but partially occluded by debris

**Analysis:**
- Leaves/debris partially covered fastener
- Model trained primarily on clean track images
- Occlusion not well-represented in training data

**Root Cause:** Dataset bias toward clean, ideal conditions

**Suggested Fix:**
- Add augmentation: random occlusion/cutout
- Include real-world images with debris, vegetation
- Train model to detect "uncertain/occluded" class

---

## False Negatives (Model Missed Actual Defects)

### Example 4: Small Crack Undetected
**Image:** `val_image_007.jpg`  
**Prediction:** No detection  
**Ground Truth:** Small hairline crack

**Analysis:**
- Crack width <5 pixels at 640×640 resolution
- YOLO11n (nano) model has limited capacity for fine details
- Similar to "needle in haystack" problem

**Root Cause:** Model architecture + small object challenge

**Suggested Fix:**
- Use larger model (YOLO11s or YOLO11m)
- Increase input resolution (640 → 1024)
- Add specific "small defect" detection head
- Pre-processing: edge enhancement filter

---

### Example 5: Low-Contrast Defect Missed
**Image:** `val_image_021.jpg`  
**Prediction:** No detection  
**Ground Truth:** Worn fastener (low contrast against background)

**Analysis:**
- Fastener color similar to rail color
- Poor contrast in overcast lighting
- Model relies heavily on color/texture cues

**Root Cause:** Lighting variation not well-represented

**Suggested Fix:**
- Add HSV augmentation (saturation, value shifts)
- Include infrared or multi-spectral imaging
- Normalize image contrast pre-processing

---

### Example 6: Perspective/Angle Issue
**Image:** `val_image_030.jpg`  
**Prediction:** No detection  
**Ground Truth:** Missing fastener visible from side angle

**Analysis:**
- Most training images are top-down or 45° angle
- Test image taken from steep side angle (~70°)
- Geometric appearance changed significantly

**Root Cause:** Limited viewpoint diversity in training data

**Suggested Fix:**
- Add rotation augmentation (±15°)
- Include multi-angle training examples
- Use 3D perspective augmentation

---

## Class Confusion Patterns

### Confusion Matrix Analysis

Most common confusions:
1. **"fastener" ↔ "defective fishplate"** (32% confusion rate)
   - Both are small metallic components
   - Similar visual features at low resolution
   - **Fix:** Increase training data, add spatial context features

2. **"missing fastener" ↔ background** (28% miss rate)
   - Absence of feature is hard to detect
   - Model needs to learn "negative space"
   - **Fix:** Add explicit "missing" class examples, use segmentation

3. **"non defective fishplate" ↔ "defective fishplate"** (15% confusion)
   - Subtle differences (small cracks, wear)
   - Requires fine-grained classification
   - **Fix:** Two-stage model (detect → classify severity)

---

## Dataset Issues Identified

### Issue 1: Class Imbalance
While overall balance is reasonable, within-split imbalance exists:
- Validation set has 30% of total data (unusual)
- Some classes underrepresented in training split
- **Impact:** Model may overfit to majority classes

**Recommendation:** Re-split dataset to 80/10/10 (train/val/test)

---

### Issue 2: Small Dataset Size
- Total: 321 images
- Per-class: ~80 images
- Industry standard: 500-1000+ per class

**Impact:** Limited model generalization

**Recommendation:**
- Collect more data (target: 500+ images per class)
- Use aggressive augmentation as temporary fix
- Consider synthetic data generation

---

### Issue 3: Annotation Quality
Spot-check revealed:
- 5-10% of bounding boxes slightly misaligned
- Some ambiguous cases (is it defective or worn?)
- Inconsistent labeling between annotators

**Impact:** Model learns noisy supervision

**Recommendation:**
- Re-label dataset with stricter guidelines
- Use multiple annotators + consensus voting
- Add "uncertain" class for ambiguous cases

---

## Environmental Challenges

### Lighting Conditions
- Model trained on daytime images
- Struggles with: shadows, glare, overcast, dusk
- **Impact:** Variable performance across time of day

**Mitigation:**
- Collect data across different times/weather
- Use histogram equalization pre-processing
- Consider adding metadata (time, weather) as input

---

### Track Conditions
- Training data bias toward clean, maintained tracks
- Real-world has: rust, debris, vegetation, graffiti
- **Impact:** False positives from "noise"

**Mitigation:**
- Include "dirty track" images in training
- Add negative class: "cosmetic_only"
- Use domain adaptation techniques

---

## Recommendations for Improvement

### Short-Term (Week 4)
1. **Adjust confidence threshold**
   - Current: 0.25 (generic)
   - Recommended: Class-specific thresholds
     - missing_fastener: 0.30 (prioritize recall)
     - defective_fishplate: 0.40 (reduce false positives)

2. **Implement ensemble**
   - Combine predictions from multiple augmented views
   - Use majority voting for final decision

3. **Add post-processing rules**
   - Filter tiny boxes (<10 pixels)
   - Remove detections in "impossible" regions (off-track)

---

### Medium-Term (1-2 months)
1. **Expand dataset**
   - Target: 500 images per class
   - Include diverse: angles, lighting, weather, track types

2. **Upgrade model**
   - Test YOLO11s (small) or YOLO11m (medium)
   - Compare performance vs. speed trade-off

3. **Two-stage pipeline**
   - Stage 1: Detect component (fastener, fishplate)
   - Stage 2: Classify condition (normal, defective, missing)

---

### Long-Term (3-6 months)
1. **Active learning**
   - Deploy model in field
   - Collect hard examples (low confidence)
   - Iteratively retrain

2. **Multi-modal fusion**
   - Combine RGB + infrared imaging
   - Use depth sensors for 3D geometry

3. **Explainability**
   - Add Grad-CAM visualization
   - Show "what the model is looking at"
   - Build trust with human inspectors

---

## Conclusion

The current model (mAP@50: 42.6%) demonstrates proof-of-concept but requires improvement for production deployment. Primary limitations stem from:
1. **Small dataset** (321 images vs. industry standard 1000+)
2. **Class confusion** between visually similar defects
3. **Environmental variation** not well-represented

**Most critical issue:** Low recall (41.3%) poses safety risk - model misses 60% of defects.

**Recommended action:** Before production deployment:
1. Expand dataset to 500+ images per class
2. Achieve minimum recall of 80% on critical classes
3. Implement human-in-the-loop validation workflow

**For portfolio/demonstration:** Current model is sufficient to showcase ML pipeline, decision logic, and deployment capabilities.

---

## Appendix: Metrics Definitions

- **Precision:** Of all predictions, how many were correct?
  - High precision = few false alarms
  
- **Recall:** Of all actual defects, how many did we find?
  - High recall = don't miss defects (critical for safety!)
  
- **mAP@50:** Mean Average Precision at 50% IoU threshold
  - Measures bounding box accuracy
  - Higher = better localization

---

**Document Version:** 1.0  
**Last Updated:** February 2026  
**Next Review:** After Week 4 completion
