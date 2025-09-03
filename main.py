import numpy as np
from utils.image_import import read_image, save_image
from correction.radiometric_correction import (
    method1_sensorcalibration,
    method2_dark_object_subtraction,
    method3_atmospheric_correction_6s,
)
from utils.normalization import normalize_to_12bit_tiled
from correction.metadata_reader import read_metadata
from metrics.mean_calculator import MeanCalculator
from metrics.quality_metrics import QualityMetricsEvaluator
from visualization.histogram_plotter import HistogramPlotter

metadata = read_metadata("C:/Users/kipki/Downloads/la_crau_psscene_analytic_udm2/PSScene/metadane_planetscope.txt")
print(metadata)

def main():
    # === Ścieżki wejściowe ===
    input_image_path = "C:/Users/kipki/Downloads/la_crau_psscene_analytic_udm2/PSScene/20250831_105805_23_2518_3B_AnalyticMS_clip.tif"
    metadata_csv_path = "C:/Users/kipki/Downloads/la_crau_psscene_analytic_udm2/PSScene/metadane_planetscope.txt"
    shapefile_path = "C:/Users/kipki/Desktop/projekty/INŻYNIERKA'25/matlab/Lacrau.shp"

    # === Ścieżki wyjściowe ===
    output_image_paths = [
        "P_output_method1_sensorcalibration.tif",
        "P_output_method2_dark_object_subtraction.tif",
        "P_output_method3_6s.tif",
    ]

    # === Wczytanie obrazu i metadanych ===
    image, profile, transform, raster_crs = read_image(input_image_path)
    radiometric_scale_factors, coeffs, sun_azimuth, sun_elevation = read_metadata(metadata_csv_path)

    # === Korekcja radiometryczna ===
    corrected_images = [
        method1_sensorcalibration(image, radiometric_scale_factors, coeffs),
        method2_dark_object_subtraction(image, radiometric_scale_factors),
    ]

    try:
        corrected_images.append(
            method3_atmospheric_correction_6s(
                image, radiometric_scale_factors, sun_azimuth, sun_elevation
            )
        )
    except ImportError as exc:  # pragma: no cover - optional dependency missing
        print(f"6S atmospheric correction skipped: {exc}")

    # === Normalizacja do 12 bit ===
    normalized_images = [normalize_to_12bit_tiled(img) for img in corrected_images]

    # === Zapis obrazów ===
    for output_path, corrected in zip(output_image_paths, corrected_images):
        save_image(corrected, profile, output_path)

    # === Obliczanie średnich wartości w masce SHP ===
    mean_calc = MeanCalculator(shapefile_path)
    target_means_all_images = []

    for norm_image in normalized_images:
        target_means = []
        for band in range(norm_image.shape[0]):
            mean_val = mean_calc.calculate_mean(norm_image[band], transform)
            target_means.append(mean_val)
        target_means_all_images.append(target_means)

    # === Wartości referencyjne (RadCalNet) ===
    simulated_image_manual = np.array([0.0543, 0.0837, 0.0837, 0.2228], dtype=np.float32)

    # === Ewaluacja jakości (SNR i dokładność radiometryczna) ===
    metrics_eval = QualityMetricsEvaluator(simulated_image=simulated_image_manual)

    absolute_accuracy = [
        metrics_eval.evaluate_absolute_accuracy(means)
        for means in target_means_all_images
    ]

    snr_original = metrics_eval.compute_snr(image)
    snr_corrected = metrics_eval.compute_snr_for_images(normalized_images)

    # === Wyniki ===
    print("== Dokładność radiometryczna względem RadCalNet ==")
    for i, acc in enumerate(absolute_accuracy):
        print(f"Metoda {i+1}: {acc['absolut']}")

    print("\n== SNR oryginalnego obrazu ==")
    print(snr_original)

    print("\n== SNR po korekcjach ==")
    for method, snrs in snr_corrected.items():
        print(f"{method}: {snrs}")

    # === Wizualizacja histogramów ===
    method_names = [
        "Kalibracja sensorów",
        "Korekcja atmosferyczna (DOS)",
        "Korekcja atmosferyczna 6S",
    ][: len(normalized_images)]
    HistogramPlotter().plot(image, normalized_images, method_names)

if __name__ == "__main__":
    main()
