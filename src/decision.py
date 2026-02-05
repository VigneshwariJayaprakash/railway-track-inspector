"""
Decision Logic Module - Enhanced with Confidence Tiers
-------------------------------------------------------
Implements safety decision rules for railway track inspection.

This module converts raw detections into actionable decisions with
a three-tier confidence system for safety-critical applications.

Decision Tiers:
- "Safe" - No defects detected
- "Review" - Low-medium confidence detections (human verification needed)
- "Needs Inspection" - Medium-high confidence defects
- "Critical" - High confidence critical defects
"""

from typing import List, Dict, Tuple


# Severity levels for each defect class
# Higher = more critical
DEFECT_SEVERITY = {
    'missing fastener': 3,      # Critical - could cause derailment
    'defective fishplate': 2,   # Serious - structural integrity issue
    'fastener': 1,              # Minor - fastener present but flagged
    'non defective fishplate': 0  # Not a defect
}


def make_decision_single_image(
    detections: List[Dict],
    conf_threshold: float = 0.15,
    medium_conf_threshold: float = 0.25,
    critical_conf_threshold: float = 0.40
) -> Tuple[str, Dict]:
    """
    Make safety decision based on detections from a single image.
    Uses three-tier confidence system for safety-critical applications.
    
    Args:
        detections: List of detection dictionaries
        conf_threshold: Minimum confidence to consider (0.15 for safety)
        medium_conf_threshold: Threshold for standard inspection (0.25)
        critical_conf_threshold: Threshold for immediate action (0.40)
        
    Returns:
        Tuple of (decision_label, decision_details)
        decision_label: "Safe", "Review", "Needs Inspection", or "Critical"
        decision_details: Dictionary with reasoning
    """
    
    if not detections:
        return "Safe", {
            'status': 'safe',
            'reason': 'No defects detected',
            'num_detections': 0,
            'confidence_tier': 'N/A',
            'max_severity': 0,
            'defects_by_tier': {'high': [], 'medium': [], 'low': []}
        }
    
    # Filter out non-defects
    actual_defects = [
        d for d in detections 
        if d['class_name'] != 'non defective fishplate'
    ]
    
    if not actual_defects:
        return "Safe", {
            'status': 'safe',
            'reason': 'Only non-defective components detected',
            'num_detections': len(detections),
            'confidence_tier': 'N/A',
            'max_severity': 0,
            'defects_by_tier': {'high': [], 'medium': [], 'low': []}
        }
    
    # Categorize defects by confidence tier
    high_conf_defects = []    # >= 0.40
    medium_conf_defects = []  # 0.25 - 0.40
    low_conf_defects = []     # 0.15 - 0.25
    
    max_severity = 0
    severity_counts = {}
    
    for det in actual_defects:
        class_name = det['class_name']
        confidence = det['confidence']
        severity = DEFECT_SEVERITY.get(class_name, 1)
        
        max_severity = max(max_severity, severity)
        severity_counts[class_name] = severity_counts.get(class_name, 0) + 1
        
        defect_info = {
            'class': class_name,
            'confidence': confidence,
            'severity': severity,
            'bbox': det.get('bbox', {})
        }
        
        # Categorize by confidence
        if confidence >= critical_conf_threshold:
            high_conf_defects.append(defect_info)
        elif confidence >= medium_conf_threshold:
            medium_conf_defects.append(defect_info)
        else:
            low_conf_defects.append(defect_info)
    
    num_defects = len(actual_defects)
    
    # Decision logic based on confidence tiers
    
    # TIER 1: Critical - High confidence defects
    if high_conf_defects:
        # Check if any are high severity
        critical_high_severity = [d for d in high_conf_defects if d['severity'] >= 3]
        
        if critical_high_severity:
            return "Critical", {
                'status': 'critical',
                'reason': f'High-confidence critical defects detected',
                'num_detections': num_defects,
                'confidence_tier': 'HIGH (≥40%)',
                'max_severity': max_severity,
                'defects_by_tier': {
                    'high': high_conf_defects,
                    'medium': medium_conf_defects,
                    'low': low_conf_defects
                },
                'severity_counts': severity_counts,
                'action': 'IMMEDIATE inspection required'
            }
        else:
            return "Needs Inspection", {
                'status': 'needs_inspection',
                'reason': f'High-confidence defects detected',
                'num_detections': num_defects,
                'confidence_tier': 'HIGH (≥40%)',
                'max_severity': max_severity,
                'defects_by_tier': {
                    'high': high_conf_defects,
                    'medium': medium_conf_defects,
                    'low': low_conf_defects
                },
                'severity_counts': severity_counts,
                'action': 'Schedule inspection within 24 hours'
            }
    
    # TIER 2: Needs Inspection - Medium confidence defects
    if medium_conf_defects:
        return "Needs Inspection", {
            'status': 'needs_inspection',
            'reason': f'{len(medium_conf_defects)} medium-confidence defect(s) detected',
            'num_detections': num_defects,
            'confidence_tier': 'MEDIUM (25-40%)',
            'max_severity': max_severity,
            'defects_by_tier': {
                'high': high_conf_defects,
                'medium': medium_conf_defects,
                'low': low_conf_defects
            },
            'severity_counts': severity_counts,
            'action': 'Inspect this section within 48 hours'
        }
    
    # TIER 3: Review - Low confidence defects (human verification needed)
    if low_conf_defects:
        return "Review", {
            'status': 'review',
            'reason': f'{len(low_conf_defects)} low-confidence detection(s) - human verification needed',
            'num_detections': num_defects,
            'confidence_tier': 'LOW (15-25%)',
            'max_severity': max_severity,
            'defects_by_tier': {
                'high': high_conf_defects,
                'medium': medium_conf_defects,
                'low': low_conf_defects
            },
            'severity_counts': severity_counts,
            'action': 'Manual review recommended (may be false positive or extreme condition)'
        }
    
    # Fallback
    return "Safe", {
        'status': 'safe',
        'reason': 'No defects above confidence threshold',
        'num_detections': 0,
        'confidence_tier': 'N/A',
        'max_severity': 0,
        'defects_by_tier': {'high': [], 'medium': [], 'low': []}
    }


def make_decision_video(
    detection_history: List[List[Dict]],
    n_threshold: int = 3,
    m_window: int = 5,
    critical_conf: float = 0.40
) -> Tuple[str, Dict]:
    """
    Make decision based on video frames using N-of-M persistence rule.
    
    N-of-M Rule: Trigger alert if defect appears in at least N of the last M frames.
    This reduces false positives from temporary occlusions, shadows, etc.
    
    Args:
        detection_history: List of detection lists (one per frame)
        n_threshold: Minimum frames with detection to trigger (N)
        m_window: Total frames to consider (M)
        critical_conf: Confidence threshold for immediate alert override
        
    Returns:
        Tuple of (decision_label, decision_details)
    """
    
    # Use only the last M frames
    recent_frames = detection_history[-m_window:] if len(detection_history) >= m_window else detection_history
    
    # Count frames with each defect type
    defect_frame_counts = {}
    all_defects = []
    
    for frame_detections in recent_frames:
        frame_defect_types = set()
        
        for det in frame_detections:
            class_name = det['class_name']
            
            # Skip non-defects
            if class_name == 'non defective fishplate':
                continue
            
            frame_defect_types.add(class_name)
            all_defects.append(det)
            
            # Check for critical high-confidence detection (immediate override)
            if det['confidence'] >= critical_conf:
                severity = DEFECT_SEVERITY.get(class_name, 1)
                if severity >= 3:
                    return "Critical", {
                        'status': 'critical',
                        'reason': f'High-confidence critical defect: {class_name} ({det["confidence"]:.2%})',
                        'trigger': 'immediate_override',
                        'defect_class': class_name,
                        'confidence': det['confidence']
                    }
        
        # Count frames for each defect type
        for defect_type in frame_defect_types:
            defect_frame_counts[defect_type] = defect_frame_counts.get(defect_type, 0) + 1
    
    # Check N-of-M rule
    persistent_defects = {
        defect: count 
        for defect, count in defect_frame_counts.items() 
        if count >= n_threshold
    }
    
    if not persistent_defects:
        return "Safe", {
            'status': 'safe',
            'reason': f'No defects persisted in {n_threshold}/{m_window} frames',
            'frames_analyzed': len(recent_frames),
            'transient_detections': len(all_defects)
        }
    
    # Determine severity
    max_severity = max(DEFECT_SEVERITY.get(d, 1) for d in persistent_defects.keys())
    
    if max_severity >= 3:
        return "Critical", {
            'status': 'critical',
            'reason': f'Critical defects persistent across {n_threshold}+ frames',
            'trigger': 'persistence_rule',
            'persistent_defects': persistent_defects,
            'max_severity': max_severity
        }
    
    return "Needs Inspection", {
        'status': 'needs_inspection',
        'reason': f'Defects persistent in {n_threshold}/{m_window} frames',
        'persistent_defects': persistent_defects,
        'max_severity': max_severity,
        'frames_analyzed': len(recent_frames)
    }


def get_recommendation(decision_label: str, decision_details: Dict) -> str:
    """
    Get human-readable recommendation based on decision.
    
    Args:
        decision_label: Decision status
        decision_details: Details dictionary
        
    Returns:
        String with recommendation
    """
    
    if decision_label == "Safe":
        return "✅ Track appears to be in good condition. Continue routine monitoring."
    
    elif decision_label == "Review":
        return "🔍 Low-confidence detection - Manual review recommended. May be extreme condition (heavy corrosion) or false positive."
    
    elif decision_label == "Needs Inspection":
        num_defects = decision_details.get('num_detections', 0)
        tier = decision_details.get('confidence_tier', '')
        
        if 'HIGH' in tier:
            return f"⚠️ {num_defects} defect(s) detected with high confidence. Inspect within 24 hours."
        else:
            return f"⚠️ {num_defects} defect(s) detected. Schedule inspection within 48 hours."
    
    elif decision_label == "Critical":
        return "🚨 CRITICAL: High-confidence severe defects detected. IMMEDIATE inspection required. Consider halting operations until verified."
    
    return "Unknown status"


def get_confidence_explanation(confidence: float) -> str:
    """
    Get explanation for confidence level.
    
    Args:
        confidence: Confidence value (0-1)
        
    Returns:
        String explaining what the confidence means
    """
    
    if confidence >= 0.40:
        return "HIGH confidence - Model is very certain"
    elif confidence >= 0.25:
        return "MEDIUM confidence - Standard detection threshold"
    elif confidence >= 0.15:
        return "LOW confidence - Requires human verification"
    else:
        return "VERY LOW confidence - Likely noise or edge case"


# Example usage
if __name__ == "__main__":
    # Test single image decision with different confidence levels
    
    print("=" * 70)
    print("TEST 1: High Confidence Critical Defect")
    print("=" * 70)
    test_detections_high = [
        {
            'class_name': 'missing fastener',
            'confidence': 0.85,
            'bbox': {'x1': 100, 'y1': 200, 'x2': 150, 'y2': 250}
        }
    ]
    
    decision, details = make_decision_single_image(test_detections_high)
    recommendation = get_recommendation(decision, details)
    
    print(f"Decision: {decision}")
    print(f"Reason: {details['reason']}")
    print(f"Confidence Tier: {details['confidence_tier']}")
    print(f"Recommendation: {recommendation}")
    
    print("\n" + "=" * 70)
    print("TEST 2: Low Confidence Defect (Extreme Corrosion)")
    print("=" * 70)
    test_detections_low = [
        {
            'class_name': 'defective fishplate',
            'confidence': 0.16,
            'bbox': {'x1': 0, 'y1': 144, 'x2': 50, 'y2': 200}
        }
    ]
    
    decision2, details2 = make_decision_single_image(test_detections_low)
    recommendation2 = get_recommendation(decision2, details2)
    
    print(f"Decision: {decision2}")
    print(f"Reason: {details2['reason']}")
    print(f"Confidence Tier: {details2['confidence_tier']}")
    print(f"Recommendation: {recommendation2}")
    
    print("\n" + "=" * 70)
    print("TEST 3: Medium Confidence Defect")
    print("=" * 70)
    test_detections_medium = [
        {
            'class_name': 'missing fastener',
            'confidence': 0.32,
            'bbox': {'x1': 220, 'y1': 232, 'x2': 270, 'y2': 280}
        }
    ]
    
    decision3, details3 = make_decision_single_image(test_detections_medium)
    recommendation3 = get_recommendation(decision3, details3)
    
    print(f"Decision: {decision3}")
    print(f"Reason: {details3['reason']}")
    print(f"Confidence Tier: {details3['confidence_tier']}")
    print(f"Recommendation: {recommendation3}")