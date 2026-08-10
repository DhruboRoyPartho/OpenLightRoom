import io
import cv2
import numpy as np
from PIL import Image, ImageCms
from core.processing.color_space import srgb_to_linear

_SRGB_PROFILE = ImageCms.createProfile("sRGB")
_SRGB_ICC_BYTES = ImageCms.ImageCmsProfile(_SRGB_PROFILE).tobytes()

_PIL_FORMAT = {"jpeg": "JPEG", "jpg": "JPEG", "png": "PNG", "tiff": "TIFF", "tif": "TIFF", "webp": "WEBP"}
_EXTENSION = {"jpeg": ".jpg", "jpg": ".jpg", "png": ".png", "tiff": ".tif", "tif": ".tif", "webp": ".webp"}


def load_and_linearize(path: str) -> np.ndarray:
    """Decode a standard (non-RAW) image file into this app's scene-linear
    float32 working space. If the file carries an embedded ICC profile
    other than sRGB, it's converted to sRGB first via LittleCMS, so the
    linearization step matches the color space the pixels are actually
    encoded in rather than assuming sRGB blindly."""
    pil_img = Image.open(path)
    pil_img = _normalize_to_srgb(pil_img)

    srgb_u8 = np.asarray(pil_img).astype(np.float32) / 255.0
    return srgb_to_linear(srgb_u8)


def _normalize_to_srgb(pil_img: Image.Image) -> Image.Image:
    icc_bytes = pil_img.info.get("icc_profile")
    if icc_bytes:
        try:
            src_profile = ImageCms.ImageCmsProfile(io.BytesIO(icc_bytes))
            pil_img = ImageCms.profileToProfile(pil_img, src_profile, _SRGB_PROFILE, outputMode="RGB")
        except Exception:
            pil_img = pil_img.convert("RGB")  # malformed profile - fall back to raw pixels
    else:
        pil_img = pil_img.convert("RGB")
    return pil_img


def save_image(path: str, image: np.ndarray, format: str = "JPEG", quality: int = 95, bit_depth: int = 8) -> str:
    """image: float32, [0, 1], display-referred (already gamma-encoded by
    ImageDocument.render()). Returns the actual path written (the correct
    extension is appended if missing). Embeds a real sRGB ICC profile for
    8-bit exports (JPEG/PNG/TIFF/WebP, via LittleCMS/Pillow in a single
    encoding pass - no lossy double-compression). 16-bit output (PNG/TIFF
    only - JPEG/WebP have no 16-bit mode) goes through OpenCV instead,
    which is the one path proven to preserve 16-bit depth correctly; it
    isn't ICC-tagged, so it's implicitly sRGB like an untagged image.
    """
    ext = format.lower()
    if ext not in _PIL_FORMAT:
        raise ValueError(f"Unsupported export format: {format}")
    if ext in ("jpeg", "jpg", "webp"):
        bit_depth = 8  # these formats have no 16-bit mode

    if not path.lower().endswith(_EXTENSION[ext]):
        path += _EXTENSION[ext]

    clipped = np.clip(image, 0.0, 1.0)

    if bit_depth == 16:
        img16 = np.round(clipped * 65535.0).astype(np.uint16)
        img_bgr = cv2.cvtColor(img16, cv2.COLOR_RGB2BGR)
        success = cv2.imwrite(path, img_bgr)
        if not success:
            raise IOError("Failed to save image.")
        return path

    img8 = np.round(clipped * 255.0).astype(np.uint8)
    pil_img = Image.fromarray(img8, mode="RGB")

    save_kwargs = {"icc_profile": _SRGB_ICC_BYTES}
    if ext in ("jpeg", "jpg"):
        save_kwargs.update(quality=quality, optimize=True)
    elif ext == "webp":
        save_kwargs.update(quality=quality)
    elif ext == "png":
        compression_level = 9 - int(quality / 11.2)
        save_kwargs.update(compress_level=max(0, min(9, compression_level)))

    pil_img.save(path, format=_PIL_FORMAT[ext], **save_kwargs)
    return path
