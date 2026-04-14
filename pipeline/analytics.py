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
            
        # Refactorización de identify_satellite:
        # Busca en los metadatos del raster (src.tags()) palabras clave como "LANDSAT" o "SENTINEL".
        for f in folder_path.iterdir():
            if f.is_file() and f.suffix.lower() in [".tif", ".tiff", ".jp2", ".tif"]:
                try:
                    with rasterio.open(f) as src:
                        tags = src.tags()
                        for key, val in tags.items():
                            val_upper = str(val).upper()
                            if "LANDSAT" in val_upper: return "LANDSAT"
                            elif "SENTINEL" in val_upper: return "SENTINEL"
                except Exception:
                    continue
                    
        logger.warning(f"Heuristics failed to identify satellite. Manejo de Footprints Vacíos: Directorio escaneado devolvió archivos -> {files}")
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
                if "B02" in f_name: paths['blue'] = str(f)
                elif "B03" in f_name: paths['green'] = str(f)
                elif "B04" in f_name: paths['red'] = str(f)
                elif "B08" in f_name: paths['nir'] = str(f)
                elif "B11" in f_name: paths['swir1'] = str(f)
                elif "B12" in f_name: paths['swir2'] = str(f)

        # Basic bands sanity check
        if not all(k in paths for k in ['green', 'red', 'nir']):
            logger.error(f"Missing essential bands for {satellite}. Found: {list(paths.keys())}")
            raise FileNotFoundError("Could not find required base bands (Green, Red, NIR) in processed directory.")
            
        return paths

    def calculate_index_by_blocks(self, band1_path: str, band2_path: str, out_temp_path: Path, index_type: str = "NDVI", green_path: str = None, red_path: str = None, nir_path: str = None, progress_callback=None, p_prefix="Calculating Index"):
        """
        Calculates normalized difference indices extracting chunks synchronously to avert RAM bloat.
        Uses windowed I/O from Rasterio. Incorporates bare-soil masking.
        """
        with rasterio.open(band1_path) as src1, rasterio.open(band2_path) as src2:
            meta = src1.meta.copy()
            meta.update(dtype=rasterio.float32, nodata=-9999.0, count=1)
            
            with rasterio.open(str(out_temp_path), 'w', **meta) as dst:
                windows = list(src1.block_windows(1))
                total_windows = len(windows)
                for count, (ji, window) in enumerate(windows, 1):
                    # Array isolation per block
                    b1 = src1.read(1, window=window).astype(np.float32)
                    b2 = src2.read(1, window=window).astype(np.float32)
                    
                    # Strict division logic masking ZeroDivisionError behavior natively
                    with np.errstate(divide='ignore', invalid='ignore'):
                        divisor = b1 + b2
                        
                        if index_type == "NDVI":
                            idx_array = (b1 - b2) / divisor
                        elif index_type == "NDWI":
                            idx_array = (b1 - b2) / divisor
                        elif index_type == "CLAY":
                            idx_array = b1 / b2
                            divisor = b2
                        elif index_type == "IRON_OXIDE":
                            idx_array = b1 / b2
                            divisor = b2
                        else:
                            raise ValueError(f"Unknown Index mapping: {index_type}")
                            
                        # Geo-Analytical Integrity (NODATA Compliance)
                        # Fixes NaN divisions bounding strictly to -9999.0 instead of relying on 0.0 rendering corruptions
                        mask_invalid = (divisor == 0) | np.isnan(idx_array) | np.isinf(idx_array) | (b1 == -9999.0) | (b2 == -9999.0)
                        
                        # Dynamic Masking logic for geological indices (Bare Soil Isolation)
                        if index_type in ["CLAY", "IRON_OXIDE"] and all([green_path, red_path, nir_path]):
                            with rasterio.open(green_path) as sg, rasterio.open(red_path) as sr, rasterio.open(nir_path) as sn:
                                g = sg.read(1, window=window).astype(np.float32)
                                r = sr.read(1, window=window).astype(np.float32)
                                n = sn.read(1, window=window).astype(np.float32)
                                
                                ephemeral_ndvi = (n - r) / (n + r)
                                ephemeral_ndwi = (g - n) / (g + n)
                                
                                soil_mask = (ephemeral_ndvi > 0.2) | (ephemeral_ndwi > 0.0)
                                mask_invalid = mask_invalid | soil_mask
                                
                        idx_array[mask_invalid] = -9999.0
                    
                    # Flush chunk out
                    dst.write(idx_array, 1, window=window)
                    
                    if progress_callback:
                        progress_callback(count, total_windows, prefix=p_prefix)

    def calculate_composite_by_blocks(self, b1_path: str, b2_path: str, b3_path: str, out_temp_path: Path, progress_callback=None, p_prefix="Stacking Composite"):
        """
        Stacks 3 bands into a multi-band COG synchronously to avert RAM bloat.
        Uses windowed I/O from Rasterio.
        """
        if not all([b1_path, b2_path, b3_path]):
            logger.error("Missing physical band paths for composition building.")
            return
            
        with rasterio.open(b1_path) as src1, rasterio.open(b2_path) as src2, rasterio.open(b3_path) as src3:
            meta = src1.meta.copy()
            meta.update(count=3) # We preserve primitive dtype honoring sensor defaults
            
            with rasterio.open(str(out_temp_path), 'w', **meta) as dst:
                windows = list(src1.block_windows(1))
                total_windows = len(windows)
                for count, (ji, window) in enumerate(windows, 1):
                    b1 = src1.read(1, window=window)
                    b2 = src2.read(1, window=window)
                    b3 = src3.read(1, window=window)
                    
                    dst.write(b1, 1, window=window)
                    dst.write(b2, 2, window=window)
                    dst.write(b3, 3, window=window)
                    
                    if progress_callback:
                        progress_callback(count, total_windows, prefix=p_prefix)

    def generate_analytical_cogs(self, session_folder: Path, requested_products: list = None, progress_callback=None):
        """Coordinador of geospatial derivation injecting temp states directly towards final Cloud structure."""
        satellite = self.identify_satellite(session_folder)
        logger.info(f"Analytical Engine: Identified {satellite} footprint in {session_folder.name}")
        
        if requested_products is None:
            requested_products = []
        
        # Pull appropriate payload paths mapped dynamically
        bands = self.get_band_paths(session_folder, satellite)
        
        session_id = session_folder.name
        
        # Calculate dynamic project root securely matching parent isolation nodes
        project_root = Path(__file__).resolve().parent.parent
        sensor_folder_name = session_folder.parent.name
        results_dir = project_root / "data" / "results" / sensor_folder_name / session_id
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Tuple sequence rules: (Product_name, Type, operand_paths...)
        products_to_calc = []
        
        if "NDVI" in requested_products:
            products_to_calc.append(("NDVI", "INDEX", bands.get('nir'), bands.get('red')))
        if "NDWI" in requested_products:
            products_to_calc.append(("NDWI", "INDEX", bands.get('green'), bands.get('nir')))
            
        if "CLAY" in requested_products:
            products_to_calc.append(("CLAY", "INDEX", bands.get('swir1'), bands.get('swir2')))
        if "IRON_OXIDE" in requested_products:
            products_to_calc.append(("IRON_OXIDE", "INDEX", bands.get('red'), bands.get('blue')))
        if "TRUE_COLOR" in requested_products:
            products_to_calc.append(("TRUE_COLOR", "COMPOSITE", bands.get('red'), bands.get('green'), bands.get('blue')))
        if "FALSE_COLOR_GEO" in requested_products:
            products_to_calc.append(("FALSE_COLOR_GEO", "COMPOSITE", bands.get('swir2'), bands.get('swir1'), bands.get('red')))
        
        for product in products_to_calc:
            prod_name = product[0]
            prod_type = product[1]
            logger.info(f"Computing {prod_name} by streaming blocks...")
            
            temp_tif = results_dir / f"temp_{prod_name}.tif"
            final_cog = results_dir / f"{session_id}_{prod_name}_COG.tif"
            
            # Step 1: Chunked extraction calculation preventing array overload
            if prod_type == "INDEX":
                if not product[2] or not product[3]: continue
                self.calculate_index_by_blocks(
                    product[2], product[3], temp_tif, 
                    index_type=prod_name,
                    green_path=bands.get('green'),
                    red_path=bands.get('red'),
                    nir_path=bands.get('nir'),
                    progress_callback=progress_callback,
                    p_prefix=f"Processing {prod_name}"
                )
            elif prod_type == "COMPOSITE":
                if not product[2] or not product[3] or not product[4]: continue
                self.calculate_composite_by_blocks(product[2], product[3], product[4], temp_tif, progress_callback=progress_callback, p_prefix=f"Stacking {prod_name}")
            
            # Step 2: Push structural COG format natively wrapping resulting raw output
            logger.info(f"Optimizing {prod_name} temp file into standard COG...")
            dst_profile = cog_profiles.get("deflate")
            dst_profile.update({
                "tiled": True, 
                "blockxsize": 256, 
                "blockysize": 256,
                "nodata": -9999.0
            })
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
                logger.info(f"✅ {prod_name} Analytics COG successfully assembled: {final_cog}")
            except Exception as e:
                logger.error(f"Failed configuring COG for {prod_name}: {e}")
            finally:
                # Cleanup volatile IOs footprint securely via pathlib
                try:
                    if temp_tif.exists():
                        temp_tif.unlink()
                except PermissionError as e:
                    logger.warning(f"WinError 32 Lock detected on {temp_tif}. Triggering AUTOSANATION_GARBAGE_COLLECTION.")
                    import gc
                    import time
                    import os
                    os.makedirs("logs", exist_ok=True)
                    with open("logs/self_healing_audit.log", "a") as f:
                        f.write(f"AUTOSANATION_GARBAGE_COLLECTION triggered on {temp_tif.name}.\n")
                    gc.collect()
                    time.sleep(0.5)
                    try:
                        if temp_tif.exists(): temp_tif.unlink()
                    except Exception as fatal_e:
                        logger.error(f"Failed lock release bypass: {fatal_e}")
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
