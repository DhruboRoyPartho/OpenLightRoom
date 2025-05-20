import cv2
import numpy as np

def load_image(path: str) -> np.ndarray:
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Failed to load image.")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def save_image(path: str, image: np.ndarray, format: str="JPEG", quality: int = 95):
    ext = format.lower()
    img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    params = []

    if ext == 'jpeg' or ext == 'jpg':
        params = [cv2.IMWRITE_JPEG_QUALITY, quality]
        path = path if path.lower().endswith('.jpg') else path + ".jpg"
    elif ext == 'png':
        compression_level = 9 - int(quality / 11.2)
        params = [cv2.IMWRITE_PNG_COMPRESSION, compression_level]
        path = path if path.lower().endswith('.png') else path + ".png"
    elif ext == 'tiff' or ext == 'tif':
        path = path if path.lower().endswith('.tif') else path + ".tif"

    success = cv2.imwrite(path, img_bgr, params)

    if not success:
        raise IOError("Failed to save image.")