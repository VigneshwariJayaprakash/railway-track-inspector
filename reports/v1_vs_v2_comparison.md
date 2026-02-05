## Confidence Threshold Impact

### Test Image 5 (Severe Corrosion)

**Challenge:** Heavy rust changes component appearance dramatically

**Results:**
- Confidence 0.25: 0 detections ❌
- Confidence 0.15: **2 detections** ✅ (defective fishplate: 16.2%, 15.5%)

**Interpretation:**
The model correctly identified corrosion but with low confidence due to:
- Limited extreme rust examples in training data
- Significant visual deviation from typical defects
- Appropriate model uncertainty

**Production Recommendation:**
Implement **tiered alert system**:
- **High confidence (>40%):** Immediate action
- **Medium confidence (15-40%):** Flag for human review
- **Low confidence (<15%):** Monitor

This approach ensures:
1. No critical defects are missed (high recall)
2. Human expertise validates uncertain cases
3. False alarm rate remains manageable