import numpy as np
import rasterio
from rasterio.windows import Window
import sys
from pathlib import Path

# Add project root to path for pipeline imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.analytics import SpectralAnalyzer

idx_type = "NDVI"
b1_val = 0.8
b2_val = 0.4
width = 16

prof = {'driver': 'GTiff', 'width': 16, 'height': 16, 'count': 1, 'dtype': 'float32', 'tiled':True, 'blockxsize':16, 'blockysize':16}

b1_path = "b1_debug.tif"
b2_path = "b2_debug.tif"

arr1 = np.full((16, 16), b1_val, dtype='float32')
arr2 = np.full((16, 16), b2_val, dtype='float32')

arr1[0,0] = 0.0
arr2[0,0] = 0.0

with rasterio.open(b1_path, 'w', **prof) as d1:
    d1.write(arr1, 1)
with rasterio.open(b2_path, 'w', **prof) as d2:
    d2.write(arr2, 1)

analyzer = SpectralAnalyzer()
out_file = "out_debug.tif"
analyzer.calculate_index_by_blocks(b1_path, b2_path, out_file, index_type=idx_type)

with rasterio.open(str(out_file)) as res:
    res_arr = res.read(1)
    print("MOCK B1 VAL:", b1_val)
    print("MOCK B2 VAL:", b2_val)
    print("RES ARR AT [1,1]:", res_arr[1,1])
    print("RES ARR AT [0,0] (EXPECT -9999.0):", res_arr[0,0])

with rasterio.open(b1_path) as src1:
    print("ACTUAL B1 AT [1,1]:", src1.read(1)[1,1])
