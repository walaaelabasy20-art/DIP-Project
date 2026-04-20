import cv2
import numpy as np
import matplotlib.pyplot as plt

class ImageProcessor:
    def __init__(self):
        pass

    # --- 1. Point Operations ---
    def addition(self, image, value=50):
        """Increase brightness by adding a constant value."""
        return cv2.add(image, np.full(image.shape, value, dtype=np.uint8))

    def subtraction(self, image, value=50):
        """Decrease brightness by subtracting a constant value."""
        return cv2.subtract(image, np.full(image.shape, value, dtype=np.uint8))

    def division(self, image, factor=2):
        """Divide pixel values to reduce contrast/intensity."""
        factor = factor if factor != 0 else 1
        res = image.astype(float) / factor
        return np.clip(res, 0, 255).astype(np.uint8)

    def complement(self, image):
        """Invert image colors (Negative)."""
        return cv2.bitwise_not(image)

    # --- 2. Color Image Operations ---
    def change_red_lighting(self, image, value=50):
        """Specifically increase the intensity of the Red channel."""
        b, g, r = cv2.split(image)
        r = cv2.add(r, value)
        return cv2.merge((b, g, r))

    def swap_r_to_g(self, image):
        """Swap Red and Green channels for artistic effect."""
        b, g, r = cv2.split(image)
        return cv2.merge((b, r, g))

    def eliminate_red(self, image):
        """Remove the Red channel entirely (set to zero)."""
        b, g, r = cv2.split(image)
        r = np.zeros_like(r)
        return cv2.merge((b, g, r))

    # --- 3. Image Histogram ---
    def histogram_stretching(self, image):
        """Expand the range of intensity levels to improve contrast."""
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        min_v, max_v, _, _ = cv2.minMaxLoc(image)
        if max_v == min_v: return image
        stretched = (image - min_v) * (255.0 / (max_v - min_v))
        return stretched.astype(np.uint8)

    def histogram_equalization(self, image):
        """Distribute pixel intensities uniformly across the histogram."""
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.equalizeHist(image)

    # --- 4. Neighborhood Processing ---
    def average_filter(self, image, size=3):
        """Apply a simple box blur by averaging local neighbors."""
        return cv2.blur(image, (size, size))

    def laplacian_filter(self, image):
        """Highlight edges using the Laplacian second-order derivative."""
        lap = cv2.Laplacian(image, cv2.CV_64F)
        return cv2.convertScaleAbs(lap)

    def maximum_filter(self, image, size=3):
        """Dilation: Assign the maximum neighbor value to the pixel."""
        return cv2.dilate(image, np.ones((size, size), np.uint8))

    def minimum_filter(self, image, size=3):
        """Erosion: Assign the minimum neighbor value to the pixel."""
        return cv2.erode(image, np.ones((size, size), np.uint8))

    def median_filter(self, image, size=3):
        """Replace pixel with the median of neighbors (Great for Salt & Pepper noise)."""
        if size % 2 == 0: size += 1
        return cv2.medianBlur(image, size)

    def mode_filter(self, image, size=3):
        """Replace pixel with the most frequent value in the neighborhood."""
        pad = size // 2
        padded = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
        out = np.zeros_like(image)
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                region = padded[i:i+size, j:j+size]
                values, counts = np.unique(region, return_counts=True)
                out[i, j] = values[np.argmax(counts)]
        return out

    # --- Display in Subplots ---
    def show_results(self, original, processed, title="Processed Result"):
        plt.figure(figsize=(10, 5))
        
        # Display Original
        plt.subplot(1, 2, 1)
        if len(original.shape) == 3:
            plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
        else:
            plt.imshow(original, cmap='gray')
        plt.title("Original Image")
        plt.axis('off')
        
        # Display Processed
        plt.subplot(1, 2, 2)
        if len(processed.shape) == 3:
            plt.imshow(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB))
        else:
            plt.imshow(processed, cmap='gray')
        plt.title(title)
        plt.axis('off')
        
        plt.show()

