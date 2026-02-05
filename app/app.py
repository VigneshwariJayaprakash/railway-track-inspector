"""
Railway Track Inspector - Streamlit Dashboard (V2)
---------------------------------------------------
Interactive web application for railway defect detection with
three-tier confidence system.

Usage:
    streamlit run app/app.py
"""

import streamlit as st
from pathlib import Path
import sys
import cv2
import numpy as np
from PIL import Image
import pandas as pd
import json

# Add parent directory to path to import from src/
sys.path.append(str(Path(__file__).parent.parent))

# Check if model exists (for deployment)
DEMO_MODE = not Path("runs/detect/railway_defect_v2/weights/best.pt").exists()

from ultralytics import YOLO

# Import decision logic
try:
    from src.decision import make_decision_single_image, get_recommendation, DEFECT_SEVERITY
except ImportError:
    # Fallback if running from different directory
    st.warning("Decision logic module not found. Using simplified logic.")
    DEFECT_SEVERITY = {
        'missing fastener': 3,
        'defective fishplate': 2,
        'fastener': 1,
        'non defective fishplate': 0
    }


# Page configuration
st.set_page_config(
    page_title="Railway Track Inspector V2",
    page_icon="🚂",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .safe-status {
        color: #28a745;
        font-weight: bold;
        font-size: 1.5rem;
    }
    .warning-status {
        color: #ffc107;
        font-weight: bold;
        font-size: 1.5rem;
    }
    .review-status {
        color: #17a2b8;
        font-weight: bold;
        font-size: 1.5rem;
    }
    .critical-status {
        color: #dc3545;
        font-weight: bold;
        font-size: 1.5rem;
    }
    </style>
""", unsafe_allow_html=True)


def load_model(model_path):
    """Load YOLO model with caching."""
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


def run_detection(model, image, conf_threshold):
    """Run detection on uploaded image."""
    try:
        # Run inference
        results = model.predict(
            source=image,
            conf=conf_threshold,
            verbose=False
        )
        
        return results[0]
    except Exception as e:
        st.error(f"Error during detection: {e}")
        return None


def parse_detections(result):
    """Parse YOLO results into structured format."""
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
    
    return detections


def simple_decision(detections, conf_threshold):
    """Simplified decision logic if module import fails."""
    if not detections:
        return "Safe", {'reason': 'No defects detected', 'num_detections': 0}
    
    actual_defects = [d for d in detections if d['class_name'] != 'non defective fishplate']
    
    if not actual_defects:
        return "Safe", {'reason': 'Only non-defective components detected', 'num_detections': len(detections)}
    
    # Check for critical defects
    critical = any(d['confidence'] > 0.40 and DEFECT_SEVERITY.get(d['class_name'], 0) >= 3 for d in actual_defects)
    
    if critical:
        return "Critical", {'reason': 'Critical defects detected', 'num_detections': len(actual_defects)}
    
    high_conf = any(d['confidence'] > 0.25 for d in actual_defects)
    
    if high_conf:
        return "Needs Inspection", {'reason': f'{len(actual_defects)} defect(s) detected', 'num_detections': len(actual_defects)}
    
    return "Review", {'reason': 'Low confidence detection', 'num_detections': len(actual_defects)}


def main():
    # Header
    st.markdown('<div class="main-header">🚂 Railway Track Inspector V2</div>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("**AI-Powered Railway Track Defect Detection System with Three-Tier Confidence**")
    st.markdown("Upload an image to detect missing fasteners, cracks, and other track defects.")
    
    # Demo mode warning for deployment
    if DEMO_MODE:
        st.warning("🔧 **Demo Mode:** This deployment shows the dashboard interface. Model inference requires local setup with trained weights.")
        st.info("📁 **To run with full detection:** Clone the [GitHub repository](https://github.com/VigneshwariJayaprakash/railway-track-inspector) and follow setup instructions in the README.")
        
        st.markdown("---")
        st.subheader("📸 Dashboard Preview")
        st.markdown("""
        This demo shows the user interface and features:
        - ✅ Three-tier confidence system (High/Medium/Low)
        - ✅ Interactive controls and settings
        - ✅ Professional inspection reports
        - ✅ Exportable results (JSON/CSV)
        
        **For full functionality with live detection**, run locally following the setup guide.
        """)
        st.markdown("---")
    
    # Sidebar
    st.sidebar.header("⚙️ Configuration")
    
    # Model selection
    st.sidebar.subheader("Model Settings")
    
    # Default model path - UPDATED TO V2
    default_model = "runs/detect/railway_defect_v2/weights/best.pt"
    
    model_path = st.sidebar.text_input(
        "Model Path",
        value=default_model,
        help="Path to trained YOLO model weights (.pt file)"
    )
    
    # Confidence threshold slider - UPDATED DEFAULT TO 0.15
    conf_threshold = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.10,
        max_value=0.50,
        value=0.15,  # Changed from 0.25
        step=0.05,
        help="Lower = more sensitive (catches extreme cases like heavy corrosion)"
    )
    
    # NEW: Confidence tier guide
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🎯 Confidence Guide")
    st.sidebar.info("""
**HIGH (≥40%):** Immediate action required
- Model is very certain
- Inspect within 24 hours

**MEDIUM (25-40%):** Standard inspection
- Good confidence level
- Schedule within 48 hours

**LOW (15-25%):** Human review needed
- May be extreme condition (heavy corrosion)
- Or possible false positive
- Expert verification recommended

**Below 15%:** Likely noise/artifacts

*Lower threshold catches extreme corrosion but may increase false positives*
    """)
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Detection Settings")
    
    show_labels = st.sidebar.checkbox("Show Labels", value=True)
    show_conf = st.sidebar.checkbox("Show Confidence Scores", value=True)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### About")
    st.sidebar.info("""
**YOLO11s Object Detection**

**Model Version:** V2 (Enhanced)

**Performance:**
- mAP@50: 48.2% (V1: 42.6%)
- Recall: 51.1% (V1: 41.3%)
- Best class: Missing Fastener (65.5%)

**Key Improvements:**
- ✅ Better occlusion handling
- ✅ Improved rust detection  
- ✅ Three-tier confidence system
- ✅ Larger model (YOLO11s vs YOLO11n)

**Confidence threshold: 0.15** for safety-critical detection

**Note:** Lower precision is acceptable - false positives are verified by humans, but missing real defects could be dangerous.
    """)
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📤 Upload Image")
        uploaded_file = st.file_uploader(
            "Choose a railway track image",
            type=['jpg', 'jpeg', 'png'],
            help="Upload an image of railway tracks for defect detection"
        )
        
        if uploaded_file is not None:
            # Display original image
            image = Image.open(uploaded_file)
            st.image(image, caption="Original Image", use_container_width=True)
            
            # Convert to numpy array for OpenCV
            image_np = np.array(image)
            if image_np.shape[2] == 4:  # RGBA
                image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)
    
    with col2:
        st.subheader("🔍 Detection Results")
        
        if uploaded_file is not None:
            # Check if model exists or demo mode
            if DEMO_MODE or not Path(model_path).exists():
                st.error("❌ Model file not available in this deployment.")
                st.info("**This is a UI demonstration.** For full functionality with live detection:")
                st.markdown("""
                ### Run Locally:
                
                1. **Clone the repository:**
```bash
                   git clone https://github.com/VigneshwariJayaprakash/railway-track-inspector.git
                   cd railway-track-inspector
```
                
                2. **Install dependencies:**
```bash
                   conda create -n railway-vision python=3.10 -y
                   conda activate railway-vision
                   pip install -r requirements.txt
```
                
                3. **Download dataset and train model** (or use pre-trained weights)
                
                4. **Run dashboard:**
```bash
                   streamlit run app/app.py
```
                
                **📖 Full setup guide:** [README.md](https://github.com/VigneshwariJayaprakash/railway-track-inspector#readme)
                """)
                
                st.markdown("---")
                st.subheader("📊 Example Results")
                st.markdown("""
                When running locally with the trained model, you would see:
                - Annotated images with bounding boxes
                - Safety assessment (Safe/Review/Needs Inspection/Critical)
                - Confidence scores and detection tables
                - Exportable reports (JSON/CSV)
                
                **See screenshots in the repository** for examples of actual detections.
                """)
            else:
                # Load model (only if not in demo mode)
                with st.spinner("Loading model..."):
                    model = load_model(model_path)
                
                if model is not None:
                    # Run detection
                    with st.spinner("Running detection..."):
                        result = run_detection(model, image_np, conf_threshold)
                    
                    if result is not None:
                        # Get annotated image
                        annotated_img = result.plot()
                        annotated_img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
                        
                        # Display annotated image
                        st.image(annotated_img_rgb, caption="Detected Defects", use_container_width=True)
                        
                        # Parse detections
                        detections = parse_detections(result)
                        
                        # Make decision with three-tier system
                        try:
                            decision_label, decision_details = make_decision_single_image(
                                detections, 
                                conf_threshold=0.15,
                                medium_conf_threshold=0.25,
                                critical_conf_threshold=0.40
                            )
                            recommendation = get_recommendation(decision_label, decision_details)
                        except:
                            decision_label, decision_details = simple_decision(detections, conf_threshold)
                            recommendation = f"Status: {decision_label}"
                        
                        # Display decision with appropriate styling
                        st.markdown("---")
                        st.subheader("🎯 Safety Assessment")
                        
                        # Status badge with color coding
                        if decision_label == "Safe":
                            st.markdown(f'<div class="safe-status">✅ {decision_label}</div>', unsafe_allow_html=True)
                        elif decision_label == "Critical":
                            st.markdown(f'<div class="critical-status">🚨 {decision_label}</div>', unsafe_allow_html=True)
                        elif decision_label == "Review":
                            st.markdown(f'<div class="review-status">🔍 {decision_label}</div>', unsafe_allow_html=True)
                        else:
                            st.markdown(f'<div class="warning-status">⚠️ {decision_label}</div>', unsafe_allow_html=True)
                        
                        # Show confidence tier if available
                        if 'confidence_tier' in decision_details:
                            st.markdown(f"**Confidence Tier:** {decision_details['confidence_tier']}")
                        
                        st.markdown(f"**Recommendation:** {recommendation}")
                        
                        # Show action if available
                        if 'action' in decision_details:
                            st.markdown(f"**Action:** {decision_details['action']}")
                        
                        # Metrics
                        st.markdown("---")
                        st.subheader("📊 Detection Summary")
                        
                        num_detections = len(detections)
                        actual_defects = [d for d in detections if d['class_name'] != 'non defective fishplate']
                        
                        metric_col1, metric_col2, metric_col3 = st.columns(3)
                        
                        with metric_col1:
                            st.metric("Total Detections", num_detections)
                        
                        with metric_col2:
                            st.metric("Defects Found", len(actual_defects))
                        
                        with metric_col3:
                            max_conf = max([d['confidence'] for d in detections]) if detections else 0
                            st.metric("Max Confidence", f"{max_conf:.1%}")
                        
                        # Detailed detections table
                        if detections:
                            st.markdown("---")
                            st.subheader("📋 Detailed Detections")
                            
                            # Create DataFrame
                            df_data = []
                            for i, det in enumerate(detections, 1):
                                # Determine confidence tier
                                conf = det['confidence']
                                if conf >= 0.40:
                                    tier = "🔴 HIGH"
                                elif conf >= 0.25:
                                    tier = "🟡 MEDIUM"
                                elif conf >= 0.15:
                                    tier = "🔵 LOW"
                                else:
                                    tier = "⚪ VERY LOW"
                                
                                df_data.append({
                                    '#': i,
                                    'Class': det['class_name'],
                                    'Confidence': f"{det['confidence']:.1%}",
                                    'Tier': tier,
                                    'Location': f"({det['bbox']['x1']:.0f}, {det['bbox']['y1']:.0f})",
                                    'Severity': DEFECT_SEVERITY.get(det['class_name'], 0)
                                })
                            
                            df = pd.DataFrame(df_data)
                            
                            # Style the dataframe
                            st.dataframe(
                                df,
                                use_container_width=True,
                                hide_index=True
                            )
                            
                            # Show defects by tier if available
                            if 'defects_by_tier' in decision_details:
                                tiers = decision_details['defects_by_tier']
                                
                                if tiers['high']:
                                    st.warning(f"🔴 **{len(tiers['high'])} HIGH confidence defect(s)** - Immediate action required")
                                if tiers['medium']:
                                    st.info(f"🟡 **{len(tiers['medium'])} MEDIUM confidence defect(s)** - Schedule inspection")
                                if tiers['low']:
                                    st.info(f"🔵 **{len(tiers['low'])} LOW confidence defect(s)** - Human review recommended")
                            
                            # Export options
                            st.markdown("---")
                            st.subheader("💾 Export Results")
                            
                            export_col1, export_col2 = st.columns(2)
                            
                            with export_col1:
                                # Export as JSON
                                export_data = {
                                    'image': uploaded_file.name,
                                    'model': model_path,
                                    'confidence_threshold': conf_threshold,
                                    'decision': decision_label,
                                    'recommendation': recommendation,
                                    'num_detections': num_detections,
                                    'detections': detections,
                                    'decision_details': decision_details
                                }
                                
                                json_str = json.dumps(export_data, indent=2)
                                st.download_button(
                                    label="📄 Download JSON",
                                    data=json_str,
                                    file_name=f"inspection_{uploaded_file.name}.json",
                                    mime="application/json"
                                )
                            
                            with export_col2:
                                # Export as CSV
                                csv = df.to_csv(index=False)
                                st.download_button(
                                    label="📊 Download CSV",
                                    data=csv,
                                    file_name=f"detections_{uploaded_file.name}.csv",
                                    mime="text/csv"
                                )
                        else:
                            st.info("No defects detected in this image.")
        else:
            st.info("👆 Upload an image to begin inspection")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666; padding: 1rem;'>
        <small>Railway Track Inspector v2.0 | Built with YOLO11s + Streamlit | 
        Enhanced with Three-Tier Confidence System</small><br>
        <small>mAP@50: 48.2% | Recall: 51.1% | Optimized for Safety-Critical Applications</small><br>
        <small>📁 <a href="https://github.com/VigneshwariJayaprakash/railway-track-inspector" target="_blank">View on GitHub</a></small>
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()