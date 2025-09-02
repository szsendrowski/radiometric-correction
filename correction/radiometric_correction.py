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


def method3_atmospheric_correction_6s(image, radiometric_scale_factors,
                                      sun_azimuth, sun_elevation):

    try:  # Import locally so that Py6S is only required when this method runs
        from Py6S import SixS, AtmosProfile, AeroProfile, Geometry, Wavelength
    except ImportError as exc:  # pragma: no cover - best effort for missing lib
        raise ImportError("Py6S library is required for 6S atmospheric "
                          "correction") from exc

    corrected_image = np.zeros_like(image, dtype=np.float32)

    # Typical central wavelengths for a 4-band PlanetScope scene [micrometres]
    central_wavelengths = [0.485, 0.56, 0.66, 0.83]

    for band in range(image.shape[0]):
        radiance = image[band] * radiometric_scale_factors[band]

        s = SixS()
        s.atmos_profile = AtmosProfile.PredefinedType(
            AtmosProfile.MidlatitudeSummer)
        s.aero_profile = AeroProfile.PredefinedType(AeroProfile.NoAerosols)
        s.geometry = Geometry.User()
        s.geometry.solar_z = 90 - sun_elevation  # convert elevation to zenith
        s.geometry.solar_a = sun_azimuth
        s.geometry.view_z = 0  # assume nadir looking sensor
        s.geometry.view_a = 0
        s.geometry.month = 7   # default values; ideally read from metadata
        s.geometry.day = 1

        wavelength = (central_wavelengths[band] if band < len(central_wavelengths)
                       else central_wavelengths[-1])
        s.wavelength = Wavelength(wavelength)

        s.run()

        # Retrieve model outputs with fallbacks to keep the function robust to
        # Py6S API changes.  These quantities are used in the classic 6S
        # formula to convert radiance at sensor to surface reflectance.
        L_path = getattr(s.outputs, "atmospheric_intrinsic_radiance", 0)
        trans_total = getattr(getattr(s.outputs, "transmittance_total_scattering", object()),
                              "upward", 1)
        trans_down = getattr(getattr(s.outputs, "transmittance_total_scattering", object()),
                              "downward", 1)
        E0 = getattr(s.outputs, "solar_irradiance", 1)
        mu_s = np.cos(np.radians(90 - sun_elevation))

        denom = max(E0 * mu_s * trans_total * trans_down, 1e-6)
        corrected_image[band] = np.pi * (radiance - L_path) / denom

    return corrected_image