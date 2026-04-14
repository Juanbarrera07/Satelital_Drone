import pytest
import numpy as np
import logging
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path for pipeline imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.report import ReportEngine
from pipeline.preprocess import Preprocessor
from pipeline.analytics import SpectralAnalyzer

# Configuration tier-1 limits
CONFIG_MOCK = {
    'preprocessing': {
        'coreg_rmse_threshold': 0.3,
    },
    'reporting': {
        'target_standard': 'IPCC_Tier_3',
        'min_accuracy': 0.90,
        'min_precision': 0.88,
        'max_uncertainty': 0.10
    }
}

# Audit Logger Setup
project_root = Path(__file__).resolve().parent.parent
logs_dir = project_root / "logs"
logs_dir.mkdir(exist_ok=True)
audit_log_file = logs_dir / "audit_results.log"

def log_audit(test_id, metric, limit):
    """Writes PASS log to audit_results.log"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    msg = f"[{timestamp}] [PASS] TestID: {test_id} - Metric: {metric} - Threshold: {limit}\n"
    with open(audit_log_file, "a", encoding="utf-8") as f:
        f.write(msg)

# 1. GENERATE 100 COMBINATIONS FOR REPORT ENGINE
acc_vals = np.linspace(0.890, 0.950, 5) # 5 elements
prec_vals = np.linspace(0.870, 0.920, 5) # 5 elements
unc_vals = np.linspace(0.080, 0.110, 4) # 4 elements -> 100 total
metric_combinations = [(a, p, u) for a in acc_vals for p in prec_vals for u in unc_vals]

@pytest.mark.parametrize("accuracy, precision, uncertainty", metric_combinations)
def test_report_engine_compliance_stress(accuracy, precision, uncertainty):
    engine = ReportEngine(CONFIG_MOCK)
    
    strict_acc = round(accuracy, 3) >= 0.900
    strict_prec = round(precision, 3) >= 0.880
    strict_unc = round(uncertainty, 3) <= 0.100
    expected_compliance = strict_acc and strict_prec and strict_unc
    
    compliance = engine.evaluate_compliance(accuracy, precision, uncertainty)
    
    if not expected_compliance and compliance:
        assert False, f"CRITICAL LEAK: System accepted failing metrics. Acc:{accuracy}, Prec:{precision}, Unc:{uncertainty}"
    elif expected_compliance and not compliance:
        assert False, f"CRITICAL REJECT: System blocked valid metrics. Acc:{accuracy}, Prec:{precision}, Unc:{uncertainty}"
        
    log_audit("ReportEngine", f"Acc:{accuracy:.3f},Prec:{precision:.3f},Unc:{uncertainty:.3f}", "0.90-0.88-0.10")

# 2. GENERATE 100 VARIATIONS FOR PREPROCESSOR TO BOA
array_variations = []
for i in range(100):
    if i < 25:
        # Standard noise array
        array_variations.append(np.random.randint(0, 5000, (10, 10)))
    elif i < 50:
        # Out of bounds (> 10000) checking scaling clip behavior
        array_variations.append(np.random.randint(9000, 20000, (10, 10)))
    elif i < 75:
        # Negative bounds
        array_variations.append(np.random.randint(-5000, 500, (10, 10)))
    else:
        # NaN, Inf injections
        arr = np.random.rand(10, 10) * 10000
        arr[0, 5] = np.nan
        arr[5, 0] = np.inf
        arr[9, 9] = -np.inf
        array_variations.append(arr)

@pytest.mark.parametrize("input_array", array_variations)
def test_preprocessor_boa_stress(input_array):
    preprocessor = Preprocessor(CONFIG_MOCK)
    
    with np.errstate(invalid='ignore', over='ignore', divide='ignore'):
        result = preprocessor.standardize_to_boa(input_array)
        
    assert result.dtype == np.float32, "Security Gate Failed: Array did not enforce float32 casting"
    
    valid_mask = ~np.isnan(result) & ~np.isinf(result)
    if np.any(valid_mask):
        max_val = np.nanmax(result[valid_mask])
        min_val = np.nanmin(result[valid_mask])
        
        if max_val > 1.0 or min_val < 0.0:
            assert False, f"BOA Threshold Breach: Values escaped the [0.0, 1.0] spectrum. Max: {max_val}, Min: {min_val}"

    # Generate a dummy hash ID to separate logs
    test_id = f"BOA_Matrix_{hash(input_array.tobytes()) % 10000}"
    log_audit(test_id, "Array Standardization", "[0.0 - 1.0]")
