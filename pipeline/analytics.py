"""
TMI Analytics Module.

Responsible for computing base spectral indices (NDVI, NDWI) relevant for mining audits.
Implements block window processing to drastically reduce RAM consumption on large Tier-1 acquisitions.
"""

import os
import glob
import logging
import numpy as np
import rasterio
from rio_cogeo.cogeo import cog_translate
from rio_cogeo.profiles import cog_profiles

logger = logging.getLogger(__name__)

class SpectralAnalyzer:
    def __init__(self):
        """Initializes the spectral analyzer engine."""
        pass

    def identify_satellite(self, folder_path: str) -> str:
        """Determines satellite type based on explicit COG file nomenclature."""
        files = os.listdir(folder_path)
        for f in files:
            if any(x in f for x in ["LC08", "LC09", "LT05", "LE07"]):
                return "LANDSAT"
            if any(x in f for x in ["S2A", "S2B", "_B04_"]):
                return "SENTINEL"
        
        # Fallback heuristic using Landsat Collection 2 suffixes
        if any("SR_B4" in f for f in files):
            return "LANDSAT"
        return "UNKNOWN"

    def get_band_paths(self, folder_path: str, satellite: str) -> dict:
        """Dynamically isolates specific bands needed for indices (Green, Red, NIR)."""
        paths = {}
        files = glob.glob(os.path.join(folder_path, "COG_*.TIF")) + glob.glob(os.path.join(folder_path, "COG_*.tif"))
        
        if satellite == "LANDSAT":
            # Landsat 8/9 OLI configuration
            for f in files:
                if "SR_B3" in f: paths['green'] = f
                elif "SR_B4" in f: paths['red'] = f
                elif "SR_B5" in f: paths['nir'] = f
        elif satellite == "SENTINEL":
            # Sentinel-2 MSI configuration
            for f in files:
                if "B03" in f: paths['green'] = f
                elif "B04" in f: paths['red'] = f
                elif "B08" in f: paths['nir'] = f

        if not all(k in paths for k in ['green', 'red', 'nir']):
            logger.error(f"Missing essential bands for {satellite}. Found: {list(paths.keys())}")
            raise FileNotFoundError("Could not find required bands (Green, Red, NIR) in processed directory.")
            
        return paths

    def calculate_index_by_blocks(self, band1_path: str, band2_path: str, out_temp_path: str, index_type: str = "NDVI"):
        """
        Calculates normalized difference indices extracting chunks synchronously to avert RAM bloat.
        Uses windowed I/O from Rasterio.
        """
        with rasterio.open(band1_path) as src1, rasterio.open(band2_path) as src2:
            meta = src1.meta.copy()
            meta.update(dtype=rasterio.float32, nodata=-9999.0, count=1)
            
            with rasterio.open(out_temp_path, 'w', **meta) as dst:
                for ji, window in src1.block_windows(1):
                    # Array isolation per block
                    b1 = src1.read(1, window=window).astype(np.float32)
                    b2 = src2.read(1, window=window).astype(np.float32)
                    
                    # Strict division logic masking ZeroDivisionError behavior natively
                    with np.errstate(divide='ignore', invalid='ignore'):
                        if index_type == "NDVI":
                            # NDVI = (NIR - Red) / (NIR + Red) -> b1: NIR, b2: Red
                            idx_array = (b1 - b2) / (b1 + b2)
                        elif index_type == "NDWI":
                            # NDWI = (Green - NIR) / (Green + NIR) -> b1: Green, b2: NIR
                            idx_array = (b1 - b2) / (b1 + b2)
                        else:
                            raise ValueError(f"Unknown Index mapping: {index_type}")
                            
                    # Cleans up NaN space or divergent infinities resulting from bounds limits
                    idx_array = np.nan_to_num(idx_array, nan=-9999.0, posinf=-9999.0, neginf=-9999.0)
                    
                    # Flush chunk out
                    dst.write(idx_array, 1, window=window)

    def generate_analytical_cogs(self, session_folder: str):
        """Coordinador of geospatial derivation injecting temp states directly towards final Cloud structure."""
        satellite = self.identify_satellite(session_folder)
        logger.info(f"Analytical Engine: Identified {satellite} footprint in {os.path.basename(session_folder)}")
        
        # Pull appropriate payload paths mapped dynamically
        bands = self.get_band_paths(session_folder, satellite)
        
        session_id = os.path.basename(session_folder)
        # Calculate dynamic project root securely
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        results_dir = os.path.join(project_root, "data", "results", session_id)
        os.makedirs(results_dir, exist_ok=True)
        
        # Tuple sequence rules: (Index_name, operand_b1_path, operand_b2_path)
        indices_to_calc = [
            ("NDVI", bands['nir'], bands['red']),
            ("NDWI", bands['green'], bands['nir'])
        ]
        
        for idx_name, b1, b2 in indices_to_calc:
            logger.info(f"Computing {idx_name} by streaming blocks...")
            temp_tif = os.path.join(results_dir, f"temp_{idx_name}.tif")
            final_cog = os.path.join(results_dir, f"{session_id}_{idx_name}_COG.tif")
            
            # Step 1: Chunked extraction calculation preventing array overload
            self.calculate_index_by_blocks(b1, b2, temp_tif, index_type=idx_name)
            
            # Step 2: Push structural COG format natively wrapping resulting raw output
            logger.info(f"Optimizing {idx_name} temp file into standard COG...")
            dst_profile = cog_profiles.get("deflate")
            dst_profile.update({"tiled": True, "blockxsize": 256, "blockysize": 256})
            config = dict(GDAL_NUM_THREADS="ALL_CPUS", GDAL_TIFF_INTERNAL_MASK=True)
            
            try:
                # Translating strictly on valid RAM cache
                cog_translate(
                    temp_tif,
                    final_cog,
                    dst_profile,
                    config=config,
                    in_memory=True, 
                    overview_level=5,
                    overview_resampling="nearest"
                )
                logger.info(f"✅ {idx_name} Analytics COG successfully assembled: {final_cog}")
            except Exception as e:
                logger.error(f"Failed configuring COG for {idx_name}: {e}")
            finally:
                # Cleanup volatile IOs footprint
                if os.path.exists(temp_tif):
                    os.remove(temp_tif)

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    print("🚀 Iniciando Motor de Análisis Espectral (TMI Analytics)...")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    processed_dir = os.path.join(project_root, "data", "processed")
    
    if not os.path.exists(processed_dir):
        print(f"❌ Directorio puente inactivo. Faltan datos COG en: {processed_dir}")
        sys.exit(1)
        
    sessions = [os.path.join(processed_dir, d) for d in os.listdir(processed_dir) if os.path.isdir(os.path.join(processed_dir, d))]
    
    if not sessions:
        print("ℹ️ Procesamiento en blanco: Ejecutar ingest/preprocess antes de invocar engine analítico.")
        sys.exit(0)
        
    analyzer = SpectralAnalyzer()
    
    for session in sessions:
        print(f"📊 Ejecutando analítica de minería en la captura: {os.path.basename(session)}")
        analyzer.generate_analytical_cogs(session)
        print("-" * 50)
        
    print("✅ Base analítica iterada satisfactoriamente. Todo el pool ha sido persistido sobre data/results/ en formato nativo COG.")
