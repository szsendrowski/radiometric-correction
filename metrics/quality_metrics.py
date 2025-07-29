import numpy as np
from scipy.ndimage import uniform_filter, sobel

class QualityMetricsEvaluator:
    def __init__(self, simulated_image=None, window_size=9, sobel_threshold=10):
        """
        simulated_image: tablica 1D (np.array) z wartościami referencyjnymi (RadCalNet)
        """
        self.simulated_image = simulated_image
        self.window_size = window_size
        self.sobel_threshold = sobel_threshold

    def evaluate_absolute_accuracy(self, target_means):
        """
        Porównuje target_means (średnie z maski) do wartości referencyjnych simulated_image.
        """
        if self.simulated_image is None:
            raise ValueError("Brakuje simulated_image do obliczenia dokładności radiometrycznej.")

        accuracy = {'absolut': []}
        for band, target_mean in enumerate(target_means):
            simulated_value = self.simulated_image[band]
            difference = abs(simulated_value - target_mean)
            percentage_diff = (difference / simulated_value) * 100
            accuracy['absolut'].append(percentage_diff)
        return accuracy

    def compute_snr(self, image):
        """
        Oblicza SNR dla pojedynczego obrazu (4 kanały).
        """
        if image.shape[0] != 4:
            raise ValueError("Obraz musi mieć dokładnie 4 kanały spektralne.")

        snr_values = {}
        for band in range(4):
            band_data = image[band].astype(np.float32)

            sobel_x = sobel(band_data, axis=1)
            sobel_y = sobel(band_data, axis=0)
            sobel_magnitude = np.hypot(sobel_x, sobel_y)

            mask = sobel_magnitude < self.sobel_threshold

            mean_filtered = uniform_filter(band_data, size=self.window_size, mode='reflect')
            std_filtered = np.sqrt(uniform_filter(band_data**2, size=self.window_size, mode='reflect') - mean_filtered**2)

            valid_pixels = mask & (std_filtered > 0)

            if np.any(valid_pixels):
                mean_signal = np.mean(mean_filtered[valid_pixels])
                noise = np.mean(std_filtered[valid_pixels])
                snr = mean_signal / noise if noise > 0 else np.nan
            else:
                snr = np.nan

            snr_values[f'Band_{band+1}'] = snr
        return snr_values

    def compute_snr_for_images(self, image_list):
        """
        Oblicza SNR dla listy obrazów.
        """
        results = {}
        for idx, image in enumerate(image_list):
            results[f'Metoda_{idx+1}'] = self.compute_snr(image)
        return results
