import rasterio

def read_image(file_path):
    with rasterio.open(file_path) as src:
        image = src.read()
        profile = src.profile
        transform = src.transform
        raster_crs = src.crs
    return image, profile, transform, raster_crs

def save_image(image, profile, output_path):
    profile.update(dtype=rasterio.float32, count=image.shape[0])
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(image.astype(rasterio.float32))
