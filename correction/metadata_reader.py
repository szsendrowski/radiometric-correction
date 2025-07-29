import pandas as pd

def read_metadata(csv_path):
    metadata = pd.read_csv(csv_path)
    radiometric_scale_factors = metadata["RadiometricScaleFactor"].values
    coeffs = metadata["ReflectanceCoefficient"].values
    sun_azimuth = metadata["SunAzimuth"].iloc[0]
    sun_elevation = metadata["SunElevation"].iloc[0]
    return radiometric_scale_factors, coeffs, sun_azimuth, sun_elevation
