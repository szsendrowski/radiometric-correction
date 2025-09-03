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
        from Py6S.sixs_exceptions import OutputParsingError
    except ImportError as exc:  # pragma: no cover - best effort for missing lib
        raise ImportError("Py6S library is required for 6S atmospheric "
                          "correction") from exc

    corrected_image = np.zeros_like(image, dtype=np.float32)

    # Typical central wavelengths for a 4-band PlanetScope scene [micrometres]
    central_wavelengths = [0.485, 0.56, 0.66, 0.83]

    for band in range(image.shape[0]):
        radiance = image[band] * radiometric_scale_factors[band]
        print("Radiance min/max:", radiance.min(), radiance.max())

        s = SixS()
        s.atmos_profile = AtmosProfile.PredefinedType(
            AtmosProfile.MidlatitudeSummer)
        s.aero_profile = AeroProfile.PredefinedType(AeroProfile.Continental)
        s.geometry = Geometry.User()
        s.geometry.solar_z = 90 - sun_elevation
        s.geometry.solar_a = sun_azimuth
        s.geometry.view_z = 0
        s.geometry.view_a = 0
        s.geometry.month = 8
        s.geometry.day = 31

        wavelength = (central_wavelengths[band] if band < len(central_wavelengths)
                       else central_wavelengths[-1])
        s.wavelength = Wavelength(wavelength)

        s.run()

        print("Available outputs:", dir(s.outputs))

        def _safe_output(name, default):
            try:
                return getattr(s.outputs, name)
            except (AttributeError, OutputParsingError):
                return default

        L_path = _safe_output("atmospheric_intrinsic_radiance", 0)
        print("L_path:", L_path)

        trans = _safe_output("transmittance_total_scattering", None)
        if trans is None:
            trans = _safe_output("transmittance_total", None)

        if trans is not None:
            trans_total = getattr(trans, "upward", 1.0)
            trans_down = getattr(trans, "downward", 1.0)
        else:
            trans_total, trans_down = 1.0, 1.0
        print("Trans total:", trans_total)
        print("Trans down:", trans_down)

        if hasattr(s.outputs, "solar_spectrum"):
            spectrum = s.outputs.solar_spectrum
            if isinstance(spectrum, dict):
                # wersja Py6S, w której spectrum jest słownikiem {λ: wartość}
                closest_wl = min(spectrum.keys(), key=lambda k: abs(k - wavelength))
                E0 = spectrum[closest_wl]
                print(f"E0 at {wavelength} µm (closest {closest_wl}):", E0)
            elif isinstance(spectrum, (float, int)):
                # wersja Py6S, w której spectrum to już pojedyncza liczba
                E0 = spectrum
                print(f"E0 at {wavelength} µm (direct):", E0)
            else:
                # fallback
                E0 = 1800.0
                print(f"E0 fallback:", E0)
        else:
            E0 = 1800.0
            print(f"E0 fallback:", E0)

        mu_s = np.cos(np.radians(90 - sun_elevation))
        denom = max(E0 * mu_s * trans_total * trans_down, 1e-6)

        corrected_image[band] = np.pi * (radiance - L_path) / denom

    return corrected_image
