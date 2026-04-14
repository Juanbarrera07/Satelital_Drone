"""
TMI Data Ingestion Module.

Responsible for reading Cloud-Optimized GeoTIFFs (COGs) and other geospatial data sources
optimally without loading the entire raster into memory.
"""

import rasterio
from rasterio.windows import Window
from typing import Iterator, Tuple
import numpy as np
import logging

logger = logging.getLogger(__name__)

class DataIngestionEngine:
    def __init__(self, file_path: str, window_size: int = 1024):
        """
        Initialize the ingestion engine for a specific COG.
        
        Args:
            file_path (str): URI or local path to the Cloud-Optimized GeoTIFF.
            window_size (int): Size of the square block for windowed I/O.
        """
        self.file_path = file_path
        self.window_size = window_size
        
    def stream_blocks(self) -> Iterator[Tuple[Window, np.ndarray]]:
        """
        Generator that yields blocks of data using Windowed I/O.
        
        Yields:
            Tuple[Window, np.ndarray]: Custom rasterio window and the data array for that window.
        """
        logger.info(f"Starting windowed ingestion from {self.file_path} (Window: {self.window_size})")
        with rasterio.open(self.file_path) as src:
            for block_index, window in src.block_windows(1):
                data = src.read(window=window)
                yield window, data
                
    def get_metadata(self) -> dict:
        """
        Retrieve source CRS, transform, and profile for downstream processing.
        
        Returns:
            dict: Rasterio profile and metadata.
        """
        with rasterio.open(self.file_path) as src:
            return src.profile
