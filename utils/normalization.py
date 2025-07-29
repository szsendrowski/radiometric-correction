import numpy as np

def normalize_to_12bit_tiled(image, tile_size=512):
    normalized_image = np.zeros_like(image, dtype=np.uint16)
    for band in range(image.shape[0]):
        min_val = image[band].min()
        max_val = image[band].max()
        if max_val > min_val:
            for i in range(0, image.shape[1], tile_size):
                for j in range(0, image.shape[2], tile_size):
                    tile = image[band, i:i+tile_size, j:j+tile_size]
                    scaled = 4095 * (tile - min_val) / (max_val - min_val)
                    normalized_image[band, i:i+tile_size, j:j+tile_size] = scaled.clip(0, 4095).astype(np.uint16)
    return normalized_image
