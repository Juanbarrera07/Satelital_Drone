import pytest
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.io import MemoryFile
from pathlib import Path
from datetime import datetime

from pipeline.ingest import DataIngestionEngine
from pipeline.analytics import SpectralAnalyzer
from pipeline.classify import LULCClassifier, CriticalBoundarySegmenter
from pipeline.report import ReportEngine
from pipeline.export import Exporter
from pipeline.preprocess import Preprocessor

# Constants
CONFIG_MOCK = {
    'preprocessing': {
        'coreg_rmse_threshold': 0.3,
        'target_level': 'BOA',
    },
    'reporting': {
        'target_standard': 'IPCC_Tier_3',
        'min_accuracy': 0.90,
        'min_precision': 0.88,
        'max_uncertainty': 0.10
    },
    'classification': {
        'rf_estimators': 100,
        'rf_max_depth': 20,
        'confidence_threshold': 0.85
    },
    'pipeline': {
        'processing_mode': 'precision'
    }
}

# ---
# FIXTURES
# ---
@pytest.fixture
def memory_raster():
    # Helper to generate a dummy TIF in memory for testing I/O
    profile = {
        'driver': 'GTiff',
        'dtype': 'uint16',
        'width': 256,
        'height': 256,
        'count': 1,
        'crs': 'EPSG:32719',
        'transform': rasterio.transform.from_origin(0, 0, 10, 10),
        'tiled': True,
        'blockxsize': 128,
        'blockysize': 128,
    }
    data = np.random.randint(0, 5000, (256, 256), dtype='uint16')
    
    with MemoryFile() as memfile:
        with memfile.open(**profile) as dataset:
            dataset.write(data, 1)
        yield memfile

# ---
# 1. INGESTION
# ---
def test_data_ingestion_engine_memory_blocks(memory_raster):
    """Test standard windowed ingestion from COG."""
    engine = DataIngestionEngine(memory_raster.name, window_size=128)
    
    metadata = engine.get_metadata()
    assert metadata['driver'] == 'GTiff'
    assert metadata['width'] == 256
    
    # Check streaming blocks
    blocks = list(engine.stream_blocks())
    assert len(blocks) == 4 # 256x256 divided by 128x128 blocks = 4 chunks
    
    for window, data in blocks:
        assert isinstance(window, Window)
        assert data.shape == (1, 128, 128) # channel, height, width

# ---
# 2. CLASSIFICATION 
# ---
def test_lulc_classification():
    classifier = LULCClassifier(config=CONFIG_MOCK)
    # create dummy BOA reflectance (Bands, H, W) -> let's say 4 bands
    dummy_features = np.random.rand(4, 128, 128).astype(np.float32)
    output = classifier.predict(dummy_features)
    
    assert output.shape == (128, 128), "Output LULC must match Height, Width"
    assert output.dtype == np.uint8, "Output LULC must be unint8 labels"
    assert np.max(output) <= 5 and np.min(output) >= 1, "Expected mock classes between 1 and 5"

def test_critical_boundary_segmentation():
    segmenter = CriticalBoundarySegmenter(config=CONFIG_MOCK)
    dummy_tensor = np.random.rand(4, 128, 128).astype(np.float32)
    
    mask, probs = segmenter.segment(dummy_tensor)
    
    assert mask.shape == (128, 128)
    assert probs.shape == (128, 128)
    assert mask.dtype == np.uint8
    # confidence threshold rule check
    assert np.all(mask[probs >= 0.85] == 1), "Pixels over confidence threshold must be flagged 1"
    assert np.all(mask[probs < 0.85] == 0), "Pixels below confidence threshold must be flagged 0"

# ---
# 3. ANALYTICS (Mathematical Bounds with NaNs/ZeroDivs)
# ---
@pytest.mark.parametrize("idx_type, b1_val, b2_val, expected", [
    ("NDVI", 0.8, 0.4, 0.333),
    ("NDWI", 0.8, 0.4, 0.333),
    ("CLAY", 0.8, 0.4, 2.0),
    ("IRON_OXIDE", 0.6, 0.2, 3.0),
])
def test_analytics_mathematical_stability(idx_type, b1_val, b2_val, expected, tmp_path):
    analyzer = SpectralAnalyzer()
    
    prof = {'driver': 'GTiff', 'width': 16, 'height': 16, 'count': 1, 'dtype': 'float32', 'tiled':True, 'blockxsize':16, 'blockysize':16}
    
    b1_path = str(tmp_path / "b1.tif")
    b2_path = str(tmp_path / "b2.tif")
    
    arr1 = np.full((16, 16), b1_val, dtype='float32')
    arr2 = np.full((16, 16), b2_val, dtype='float32')
    arr1[0,0] = 0.0
    arr2[0,0] = 0.0
    
    with rasterio.open(b1_path, 'w', **prof) as d1:
        d1.write(arr1, 1)
    with rasterio.open(b2_path, 'w', **prof) as d2:
        d2.write(arr2, 1)
        
    out_file = tmp_path / "out_temp.tif"
    analyzer.calculate_index_by_blocks(b1_path, b2_path, out_file, index_type=idx_type)
    
    with rasterio.open(str(out_file)) as res:
        res_arr = res.read(1)
        assert np.isclose(res_arr[1,1], expected, atol=0.01), f"Math failure: {idx_type}"
        assert res_arr[0,0] == -9999.0, "Zero Division was not caught and replaced with nodata"

# ---
# 4. EXPORT
# ---
def test_exporter_writes_cog(tmp_path):
    out_path = tmp_path / "export_test.tif"
    profile = {
        'driver': 'GTiff',
        'width': 100, 'height': 100, 'count': 1, 'dtype': 'uint8'
    }
    
    dummy_data = np.ones((100, 100), dtype='uint8')
    with Exporter(str(out_path), profile) as exp:
        exp.write_window(dummy_data, window=Window(0, 0, 100, 100))
        
    assert out_path.exists()
    
    with rasterio.open(str(out_path)) as dst:
        assert dst.profile['compress'] == 'deflate'
        assert dst.profile['tiled'] == True
        assert dst.read(1).shape == (100, 100)
