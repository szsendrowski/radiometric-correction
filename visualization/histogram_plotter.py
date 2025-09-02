import matplotlib.pyplot as plt

class HistogramPlotter:
    def __init__(self, bins=100):
        self.bins = bins

    def plot(self, original_image, corrected_images_list, method_names):
        num_methods = len(corrected_images_list) + 1
        plt.figure(figsize=(18, 12))

        # Histogram oryginalnego obrazu
        plt.subplot(2, 3, 1)
        for i, band in enumerate(original_image, start=1):
            plt.hist(band.flatten(), bins=self.bins, alpha=0.5, label=f"Pasmo {i}")
        plt.xlabel("Wartości pikseli")
        plt.ylabel("Liczba pikseli")
        plt.title("Oryginalny obraz")
        plt.legend()

        # Histogramy dla skorygowanych obrazów
        for idx, (corrected_image, method_name) in enumerate(zip(corrected_images_list, method_names), start=2):
            plt.subplot(2, 3, idx)
            for i, band in enumerate(corrected_image, start=1):
                plt.hist(band.flatten(), bins=self.bins, alpha=0.5, label=f"Pasmo {i}")
            plt.xlabel("Wartości pikseli")
            plt.ylabel("Liczba pikseli")
            plt.title(method_name)
            plt.legend()

        plt.tight_layout()
        plt.show()


def histogram_plotter(original_image, corrected_images_list, method_names):
    """Convenience wrapper for plotting histograms.

    Creates a :class:`HistogramPlotter` instance and calls its
    :meth:`plot` method with the provided images.

    Parameters
    ----------
    original_image : np.ndarray
        The original image array.
    corrected_images_list : list[np.ndarray]
        List of corrected images to compare with the original.
    method_names : list[str]
        Names of the correction methods corresponding to the images.
    """
    plotter = HistogramPlotter()
    plotter.plot(original_image, corrected_images_list, method_names)
