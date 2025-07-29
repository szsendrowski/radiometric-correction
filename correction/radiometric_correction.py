import numpy as np

def method1_sensorcalibration(image, radiometric_scale_factors, coeffs):
    corrected_image = np.zeros_like(image, dtype=np.float32)
    for band in range(image.shape[0]):
        radiance = image[band] * radiometric_scale_factors[band]
        corrected_image[band] = radiance * coeffs[band]
    return corrected_image

def method2_dark_object_subtraction(image, radiometric_scale_factors):
    corrected_image = np.zeros_like(image, dtype=np.float32)
    for band in range(image.shape[0]):
        radiance = image[band] * radiometric_scale_factors[band]
        min_value = np.percentile(radiance, 0.5)
        corrected_image[band] = np.maximum(radiance - min_value, 0)
    return corrected_image
