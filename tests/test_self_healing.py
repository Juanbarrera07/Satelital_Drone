import os
import pytest
import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio.crs import CRS
import sys
import os
from pathlib import Path

# Fix relative path injection for pytest discovering pipeline root modules
TEST_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = TEST_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Local pipeline hooks
from pipeline.preprocess import Preprocessor
from pipeline.analytics import SpectralAnalyzer

TEMP_DIR = TEST_ROOT / "temp_artifacts"

@pytest.fixture(scope="session", autouse=True)
def setup_teardown():
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    yield
    # We do NOT delete TEMP_DIR automatically to allow manual inspection if a test fails
    pass

def create_mock_raster(path: Path, crs_epsg: int = 4326, has_nan: bool = False):
    """Generates a reliable dummy raster dynamically."""
    arr = np.random.rand(1, 256, 256).astype(np.float32)
    if has_nan:
        # Inject structural voids
        arr[0, 50:100, 50:100] = np.nan
        arr[0, 150:200, 150:200] = float('inf')
        
    if crs_epsg == 4326:
        # Realistic bounding geometry protecting PROJ valid EPSG ranges
        transform = from_origin(-68.9, -22.4, 0.0001, 0.0001)
    else:
        transform = from_origin(0, 0, 10, 10)
        
    meta = {
        'driver': 'GTiff',
        'height': 256,
        'width': 256,
        'count': 1,
        'dtype': 'float32',
        'crs': CRS.from_epsg(crs_epsg),
        'transform': transform,
        'nodata': -9999.0
    }
    with rasterio.open(str(path), 'w', **meta) as dst:
        dst.write(arr)

def test_crs_projection_healing():
    """Autoreprojection assertion forcing EPSG:32719 limits."""
    raw_path = TEMP_DIR / "dummy_4326.tif"
    cog_path = TEMP_DIR / "dummy_32719_COG.tif"
    
    create_mock_raster(raw_path, crs_epsg=4326)
    
    prep = Preprocessor()
    prep.build_cog(str(raw_path), str(cog_path))
    
    assert cog_path.exists(), "COG was not successfully generated."
    
    with rasterio.open(str(cog_path)) as src:
        assert src.crs == CRS.from_epsg(32719), f"Autosanitation CRS Mismatch. Expected 32719, got {src.crs.to_epsg()}"
        
@pytest.fixture
def mock_bands_for_analytics():
    """Builds a mini-satellite session resolving basic analytical parameters."""
    session_dir = TEMP_DIR / "mock_session"
    session_dir.mkdir(exist_ok=True)
    
    b3 = session_dir / "COG_S2A_B03_10m.tif" # Green
    b4 = session_dir / "COG_S2A_B04_10m.tif" # Red
    b8 = session_dir / "COG_S2A_B08_10m.tif" # NIR
    
    create_mock_raster(b3, crs_epsg=32719, has_nan=True)
    create_mock_raster(b4, crs_epsg=32719, has_nan=True)
    create_mock_raster(b8, crs_epsg=32719, has_nan=True)
    return session_dir

def test_data_integrity_nan_healing(mock_bands_for_analytics):
    """Validates structural cleanup projecting -9999.0 onto undefined math."""
    session_dir = mock_bands_for_analytics
    
    analyzer = SpectralAnalyzer()
    analyzer.generate_analytical_cogs(session_dir, requested_products=["NDVI"])
    
    # Analyze the generated result mapping
    results_dir = session_dir.parent.parent.parent / "data" / "results" / "temp_artifacts" / session_dir.name
    out_ndvi = results_dir / f"{session_dir.name}_NDVI_COG.tif"
    
    assert out_ndvi.exists(), "Analytics Engine failed compiling NDVI COG."
    
    with rasterio.open(str(out_ndvi)) as src:
        bounds_array = src.read(1)
        assert src.nodata == -9999.0, "NODATA Metadata incorrectly aligned."
        # Validate that no NaNs persisted in array matrices mapping
        assert not np.isnan(bounds_array).any(), "NaN values breached the standardization filter."
        assert not np.isinf(bounds_array).any(), "INF values breached the standardization filter."
        
        # We assume values validly rendered exist physically bounded between [-1, 1] structurally (NDVI) or are strictly -9999.0
        valid_pixels = bounds_array[bounds_array != -9999.0]
        assert (valid_pixels >= -1.0).all() and (valid_pixels <= 1.0).all(), "Index math exceeded natural boundaries [-1, 1]."

def test_winerror_32_lock_healing_gc(mock_bands_for_analytics):
    """Simulates a locked file handler asserting that aggressive GC overrides unlinks natively."""
    session_dir = mock_bands_for_analytics
    analyzer = SpectralAnalyzer()
    
    # We will invoke standard generation but hold a secret lock on what will map to the temp file
    results_dir = session_dir.parent.parent.parent / "data" / "results" / "temp_artifacts" / session_dir.name
    results_dir.mkdir(parents=True, exist_ok=True)
    
    temp_target = results_dir / "temp_NDWI.tif"
    temp_target.touch()
    
    try:
        # Mocking Windows HANDLE Lock natively
        with open(str(temp_target), "r") as locked_handle:
            # We enforce generation knowing the analyzer routinely cleans up `temp_`
            analyzer.generate_analytical_cogs(session_dir, requested_products=["NDWI"])
            
            # The analyzer MUST not raise a crash and should proceed catching the lock internally.
    except Exception as e:
        pytest.fail(f"Garbage Collection Self-Healing failed bypassing WinError 32 lock: {e}")
