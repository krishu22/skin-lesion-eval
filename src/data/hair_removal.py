import cv2
import numpy as np
from PIL import Image


class DullRazor:
    """Removes hair artifacts from dermoscopic images via morphological
    blackhat filtering + inpainting (Lee et al., DullRazor)."""

    def __init__(self, kernel_size=9, inpaint_radius=6, threshold=10):
        self.kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (kernel_size, kernel_size))
        self.inpaint_radius = inpaint_radius
        self.threshold = threshold

    def __call__(self, img):
        image = np.array(img.convert("RGB"))

        grayscale_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        blackhat = cv2.morphologyEx(grayscale_image, cv2.MORPH_BLACKHAT, self.kernel)
        gaussian_blur_image = cv2.GaussianBlur(blackhat, (3, 3), cv2.BORDER_DEFAULT)
        _, mask = cv2.threshold(gaussian_blur_image, self.threshold, 255, cv2.THRESH_BINARY)
        final_image = cv2.inpaint(image, mask, self.inpaint_radius, cv2.INPAINT_TELEA)

        return Image.fromarray(final_image)
