"""
TMI Export Module.

Handles saving output masks and calculated metrics to efficient geospatial formats.
"""

import rasterio
import numpy as np
from rasterio.windows import Window
import logging

logger = logging.getLogger(__name__)

class Exporter:
    def __init__(self, output_path: str, profile: dict):
        """
        Initialize the spatial exporter.
        
        Args:
            output_path (str): Resulting file path.
            profile (dict): Rasterio profile dictionary for writing.
        """
        self.output_path = output_path
        # Update profile to standard output formats (e.g., COG, Compression)
        self.profile = profile.copy()
        self.profile.update({
            'driver': 'GTiff',
            'compress': 'deflate',
            'tiled': True,
            'blockxsize': 512,
            'blockysize': 512
        })
        self._dataset = None

    def __enter__(self):
        self._dataset = rasterio.open(self.output_path, 'w', **self.profile)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._dataset is not None:
            self._dataset.close()

    def write_window(self, data: np.ndarray, window: Window, band: int = 1):
        """
        Write processed data block to output raster.
        """
        if self._dataset is None:
            raise RuntimeError("Exporter must be used within a context manager.")
        self._dataset.write(data, window=window, indexes=band)
