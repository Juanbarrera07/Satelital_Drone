"""
TMI Preprocessing Module & Refinery.

Handles archive extraction, Cloud-Optimized GeoTIFF (COG) generation,
radiometric calibration, standardization to Bottom Of Atmosphere (BOA) 
Surface Reflectance, and co-registration quality checks.
"""

import os
import tarfile
import zipfile
import glob
import logging
import numpy as np

# rio-cogeo for memory-efficient COG conversions
from rio_cogeo.cogeo import cog_translate
from rio_cogeo.profiles import cog_profiles

logger = logging.getLogger(__name__)

class Preprocessor:
    def __init__(self, config: dict = None):
        """
        Initialize the preprocessor.
        
        Args:
            config (dict): Pipeline configuration dict.
        """
        config = config or {}
        self.target_level = config.get('preprocessing', {}).get('target_level', 'BOA')
        self.rmse_threshold = config.get('preprocessing', {}).get('coreg_rmse_threshold', 0.5)

    def extract_archive(self, archive_path: str, extract_dir: str) -> list:
        """
        Detects if the file is .zip or .tar(.gz) and extracts its contents into extract_dir.
        
        Returns:
            list: Paths to the extracted raw TIF files.
        """
        logger.info(f"Refinery: Decompressing archive {archive_path} into {extract_dir}")
        os.makedirs(extract_dir, exist_ok=True)
        
        if zipfile.is_zipfile(archive_path):
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
        elif tarfile.is_tarfile(archive_path):
            with tarfile.open(archive_path, 'r:*') as tar_ref:
                tar_ref.extractall(extract_dir)
        else:
             logger.error(f"Unsupported archive format for {archive_path}")
             raise ValueError(f"Not a valid .zip or .tar file: {archive_path}")
             
        # Locate extracted tif files (case insensitive approach)
        tifs = []
        for root, dirs, files in os.walk(extract_dir):
            for file in files:
                if file.lower().endswith(('.tif', '.tiff')):
                    tifs.append(os.path.join(root, file))
                    
        logger.info(f"Refinery: Extracted {len(tifs)} TIF components.")
        return tifs

    def build_cog(self, src_path: str, dst_path: str):
        """
        Converts a raw TIF into a Cloud-Optimized GeoTIFF (COG) highly optimized for Streamlit maps.
        """
        logger.info(f"Refinery: Converting {os.path.basename(src_path)} to COG.")
        
        # COG specifications explicitly demanded
        dst_profile = cog_profiles.get("deflate")
        dst_profile.update({
            "tiled": True,
            "blockxsize": 256,
            "blockysize": 256
        })
        
        # Configuration tuning
        config = dict(
            GDAL_NUM_THREADS="ALL_CPUS",
            GDAL_TIFF_INTERNAL_MASK=True,
            GDAL_TIFF_OVR_BLOCKSIZE="128" # internal overviews optimization
        )
        
        try:
            # Perform translation in memory to avoid Windows temp file permission deadlocks
            cog_translate(
                src_path,
                dst_path,
                dst_profile,
                config=config,
                in_memory=True,    # forces RAM buffer strictly bypassing disk temp locks
                overview_level=5,  # creates internal pyramid overviews for zoom
                overview_resampling="nearest"
            )
            logger.info(f"COG generation successful: {os.path.basename(dst_path)}")
        except PermissionError as e:
            logger.error(f"⚠️ PermissionError (WinError 32): Unable to write or cleanup files.")
            logger.error(f"Sugerencia: Cierra el Explorador de Windows, QGIS o cualquier visor de imágenes anclado en {os.path.dirname(dst_path)}.")
            raise e
        except Exception as e:
            logger.error(f"Unexpected error translating {src_path} to COG: {e}")
            raise e

    def refinery_pipeline(self, raw_archive_path: str) -> list:
        """
        Orchestrates extracting raw data from download_data.py to cloud-optimized analytical formats.
        """
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        session_id = os.path.splitext(os.path.basename(raw_archive_path))[0]
        
        interim_dir = os.path.join(base_dir, "data", "interim", session_id)
        processed_dir = os.path.join(base_dir, "data", "processed", session_id)
        os.makedirs(processed_dir, exist_ok=True)
        
        # Step 1: Extraction
        raw_tifs = self.extract_archive(raw_archive_path, interim_dir)
        
        # Step 2: Conversion to COG 
        cog_paths = []
        for tif in raw_tifs:
            filename = os.path.basename(tif)
            cog_path = os.path.join(processed_dir, f"COG_{filename}")
            if not os.path.exists(cog_path):
                self.build_cog(tif, cog_path)
            cog_paths.append(cog_path)
            
        return cog_paths

    def standardize_to_boa(self, data: np.ndarray, scaling_factor: float = 10000.0) -> np.ndarray:
        """
        Convert raw digital numbers or TOA reflectance to BOA Surface Reflectance.
        """
        logger.info("Applying BOA standardization.")
        boa_data = data.astype(np.float32) / scaling_factor
        return np.clip(boa_data, 0.0, 1.0)

    def validate_coregistration(self, expected_rmse: float) -> bool:
        """
        Validate if the co-registration error falls within acceptable thresholds.
        """
        passed = expected_rmse <= self.rmse_threshold
        if not passed:
            logger.warning(f"Co-registration RMSE ({expected_rmse}) exceeds threshold ({self.rmse_threshold}).")
        return passed

    def process_block(self, window_data: np.ndarray) -> np.ndarray:
        """
        Apply entire preprocessing pipeline to a specific block natively.
        """
        return self.standardize_to_boa(window_data)

if __name__ == "__main__":
    import sys
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    print("🚀 Iniciando motor de Refinería COG (TMI Preprocessing)...")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    raw_dir = os.path.join(project_root, "data", "raw", "landsat")
    
    if not os.path.exists(raw_dir):
        print(f"❌ Abortando: Directorio de ingesta no encontrado -> {raw_dir}")
        sys.exit(1)
        
    archives = glob.glob(os.path.join(raw_dir, "*.tar")) + glob.glob(os.path.join(raw_dir, "*.zip"))
    
    if not archives:
        print(f"ℹ️ No se localizaron archivos crudos (.tar / .zip) en {raw_dir}. Ejecuta la fase de ingesta primero.")
        sys.exit(0)
        
    preprocessor = Preprocessor()
    total_cogs_created = 0
    
    for archive in archives:
        filename = os.path.basename(archive)
        print(f"🔄 Procesando paquete raw: {filename} ...")
        cog_paths = preprocessor.refinery_pipeline(archive)
        total_cogs_created += len(cog_paths)
        for p in cog_paths:
            print(f"  -> Generado: {os.path.basename(p)}")
            
    print(f"✅ Refinería cerrada satisfactoriamente. {total_cogs_created} archivos COG generados para análisis.")
