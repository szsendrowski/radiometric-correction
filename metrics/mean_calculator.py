import geopandas as gpd
import numpy as np
from rasterio.features import geometry_mask

class MeanCalculator:
    def __init__(self, shapefile_path):
        self.shape = gpd.read_file(shapefile_path).geometry.iloc[0]

    def calculate_mean(self, band_image, transform):
        mask_array = geometry_mask([self.shape], transform=transform, invert=True, out_shape=band_image.shape)
        masked_image = np.ma.masked_array(band_image, mask_array)
        return masked_image.mean()
