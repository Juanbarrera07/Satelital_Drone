"""
Pytest suite for TerraForge Mining Intelligence Quality Gates.
"""

import pytest
import numpy as np
from pipeline.preprocess import Preprocessor
from pipeline.report import ReportEngine

@pytest.fixture
def base_config():
    return {
        'preprocessing': {'target_level': 'BOA', 'coreg_rmse_threshold': 0.5},
        'reporting': {'target_standard': 'IPCC_Tier_3', 'min_accuracy': 0.85, 'min_precision': 0.80, 'max_uncertainty': 0.15}
    }

def test_boa_standardization(base_config):
    """Test if raw data is correctly standardized and clipped."""
    prep = Preprocessor(base_config)
    raw_data = np.array([5000, 15000, 2000], dtype=np.uint16)
    boa = prep.standardize_to_boa(raw_data, scaling_factor=10000.0)
    
    # Check bounds (clipping at 1.0)
    assert np.all(boa >= 0.0)
    assert np.all(boa <= 1.0)
    assert np.isclose(boa[0], 0.5)
    assert np.isclose(boa[1], 1.0) # Should be clipped

def test_coregistration_quality_gate(base_config):
    """Test if RMSE threshold correctly triggers pass/fail."""
    prep = Preprocessor(base_config)
    assert prep.validate_coregistration(0.4) is True
    assert prep.validate_coregistration(0.6) is False

def test_report_compliance_success(base_config):
    """Test if APU metrics pass when values are above Tier 3 thresholds."""
    engine = ReportEngine(base_config)
    is_compliant = engine.evaluate_compliance(accuracy=0.88, precision=0.85, uncertainty=0.10)
    assert is_compliant is True

def test_report_compliance_failure(base_config):
    """Test if APU metrics fail when values underperform."""
    engine = ReportEngine(base_config)
    is_compliant = engine.evaluate_compliance(accuracy=0.80, precision=0.85, uncertainty=0.10)
    assert is_compliant is False # Fails due to low accuracy

def test_unet_segmentation_shape():
    """Test if U-Net segmentation mock engine returns expected array shapes."""
    from pipeline.classify import CriticalBoundarySegmenter
    segmenter = CriticalBoundarySegmenter(config={'pipeline': {'processing_mode': 'fast'}})
    mock_input = np.random.rand(4, 256, 256) # 4 bands, 256x256
    
    mask, probs = segmenter.segment(mock_input)
    assert mask.shape == (256, 256)
    assert probs.shape == (256, 256)
