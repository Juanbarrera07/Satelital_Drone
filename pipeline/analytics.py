"""
TMI Analytics Module.

Responsible for computing base spectral indices (NDVI, NDWI) 
and advanced geological indices (CLAY, IRON_OXIDE) for mining audits.
Implements block window processing to drastically reduce RAM consumption on large Tier-1 acquisitions.
"""

import logging
import numpy as np
import rasterio
from pathlib import Path
from rio_cogeo.cogeo import cog_translate
from rio_cogeo.profiles import cog_profiles

logger = logging.getLogger(__name__)

class SpectralAnalyzer:
    def __init__(self):
        """Initializes the spectral analyzer engine."""
        pass

    def identify_satellite(self, folder_path: Path) -> str:
        """Determines satellite type based on explicit COG file nomenclature."""
        files = [f.name for f in folder_path.iterdir() if f.is_file()]
        for f in files:
            if any(x in f for x in ["LC08", "LC09", "LT05", "LE07"]):
                return "LANDSAT"
            if any(x in f for x in ["S2A", "S2B", "_B04_"]):
                return "SENTINEL"
        
        # Fallback heuristic using Landsat Collection 2 suffixes
        if any("SR_B4" in f for f in files):
            return "LANDSAT"
        return "UNKNOWN"

    def get_band_paths(self, folder_path: Path, satellite: str) -> dict:
        """Dynamically isolates specific bands needed for indices."""
        paths = {}
        files = list(folder_path.glob("COG_*.TIF")) + list(folder_path.glob("COG_*.tif"))
        
        if satellite == "LANDSAT":
            # Landsat 8/9 OLI configuration
            for f in files:
                f_name = f.name
                if "SR_B2" in f_name: paths['blue'] = str(f)
                elif "SR_B3" in f_name: paths['green'] = str(f)
                elif "SR_B4" in f_name: paths['red'] = str(f)
                elif "SR_B5" in f_name: paths['nir'] = str(f)
                elif "SR_B6" in f_name: paths['swir1'] = str(f)
                elif "SR_B7" in f_name: paths['swir2'] = str(f)
        elif satellite == "SENTINEL":
            # Sentinel-2 MSI configuration
            for f in files:
                f_name = f.name
                if "B03" in f_name: paths['green'] = str(f)
                elif "B04" in f_name: paths['red'] = str(f)
                elif "B08" in f_name: paths['nir'] = str(f)

        # Basic bands sanity check
        if not all(k in paths for k in ['green', 'red', 'nir']):
            logger.error(f"Missing essential bands for {satellite}. Found: {list(paths.keys())}")
            raise FileNotFoundError("Could not find required base bands (Green, Red, NIR) in processed directory.")
            
        return paths

    def calculate_index_by_blocks(self, band1_path: str, band2_path: str, out_temp_path: Path, index_type: str = "NDVI"):
        """
        Calculates normalized difference indices extracting chunks synchronously to avert RAM bloat.
        Uses windowed I/O from Rasterio.
        """
        with rasterio.open(band1_path) as src1, rasterio.open(band2_path) as src2:
            meta = src1.meta.copy()
            meta.update(dtype=rasterio.float32, nodata=-9999.0, count=1)
            
            with rasterio.open(str(out_temp_path), 'w', **meta) as dst:
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
                        elif index_type == "CLAY":
                            # Clay = SWIR1 / SWIR2 -> b1: SWIR1, b2: SWIR2
                            idx_array = b1 / b2
                        elif index_type == "IRON_OXIDE":
                            # Iron Oxide = Red / Blue -> b1: Red, b2: Blue
                            idx_array = b1 / b2
                        else:
                            raise ValueError(f"Unknown Index mapping: {index_type}")
                            
                    # Cleans up NaN space or divergent infinities resulting from bounds limits
                    idx_array = np.nan_to_num(idx_array, nan=-9999.0, posinf=-9999.0, neginf=-9999.0)
                    
                    # Flush chunk out
                    dst.write(idx_array, 1, window=window)

    def generate_analytical_cogs(self, session_folder: Path):
        """Coordinador of geospatial derivation injecting temp states directly towards final Cloud structure."""
        satellite = self.identify_satellite(session_folder)
        logger.info(f"Analytical Engine: Identified {satellite} footprint in {session_folder.name}")
        
        # Pull appropriate payload paths mapped dynamically
        bands = self.get_band_paths(session_folder, satellite)
        
        session_id = session_folder.name
        
        # Calculate dynamic project root securely
        project_root = Path(__file__).resolve().parent.parent
        results_dir = project_root / "data" / "results" / session_id
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Tuple sequence rules: (Index_name, operand_b1_path, operand_b2_path)
        indices_to_calc = [
            ("NDVI", bands['nir'], bands['red']),
            ("NDWI", bands['green'], bands['nir'])
        ]
        
        if satellite == "LANDSAT":
            # Add strict geological mappings based on landsat data bounds
            if all(k in bands for k in ['swir1', 'swir2', 'red', 'blue']):
                indices_to_calc.extend([
                    ("CLAY", bands['swir1'], bands['swir2']),
                    ("IRON_OXIDE", bands['red'], bands['blue'])
                ])
            else:
                logger.warning("Landsat payload missing specialized bands. Skipping CLAY and IRON_OXIDE indices.")
        
        for idx_name, b1, b2 in indices_to_calc:
            logger.info(f"Computing {idx_name} by streaming blocks...")
            temp_tif = results_dir / f"temp_{idx_name}.tif"
            final_cog = results_dir / f"{session_id}_{idx_name}_COG.tif"
            
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
                    str(temp_tif),
                    str(final_cog),
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
                # Cleanup volatile IOs footprint securely via pathlib
                try:
                    if temp_tif.exists():
                        temp_tif.unlink()
                except Exception as e:
                    logger.warning(f"Could not cleanly unlink volatile temp file {temp_tif}: {e}")

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    print("🚀 Iniciando Motor de Análisis Espectral Geoespacial (TMI Analytics)...")
    
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    processed_dir = project_root / "data" / "processed"
    
    if not processed_dir.exists():
        print(f"❌ Directorio puente inactivo. Faltan datos COG en: {processed_dir}")
        sys.exit(1)
        
    sessions = [d for d in processed_dir.iterdir() if d.is_dir()]
    
    if not sessions:
        print("ℹ️ Procesamiento en blanco: Ejecutar ingest/preprocess antes de invocar engine analítico.")
        sys.exit(0)
        
    analyzer = SpectralAnalyzer()
    
    for session in sessions:
        print(f"📊 Ejecutando analítica de minería en la captura: {session.name}")
        analyzer.generate_analytical_cogs(session)
        print("-" * 50)
        
    print("✅ Base analítica iterada satisfactoriamente. Subproductos exportados exitosamente a data/results/ en formato nativo COG.")
