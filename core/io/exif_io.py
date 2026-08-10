import exifread
import piexif

# Curated set of the most broadly meaningful EXIF fields. Full byte-exact
# EXIF block round-tripping - including proprietary per-manufacturer
# MakerNote data - is out of scope; this preserves the human-readable
# shooting information that's actually useful once a photo is exported.
_TAG_MAP = {
    "Image Make": ("0th", piexif.ImageIFD.Make),
    "Image Model": ("0th", piexif.ImageIFD.Model),
    "EXIF DateTimeOriginal": ("Exif", piexif.ExifIFD.DateTimeOriginal),
    "EXIF ISOSpeedRatings": ("Exif", piexif.ExifIFD.ISOSpeedRatings),
    "EXIF FocalLength": ("Exif", piexif.ExifIFD.FocalLength),
    "EXIF LensModel": ("Exif", piexif.ExifIFD.LensModel),
    "EXIF ExposureTime": ("Exif", piexif.ExifIFD.ExposureTime),
    "EXIF FNumber": ("Exif", piexif.ExifIFD.FNumber),
}

_RATIONAL_TYPES = (5, 10)   # RATIONAL, SRATIONAL
_INT_TYPES = (3, 4, 8, 9)   # SHORT, LONG, SSHORT, SLONG


def _to_piexif_value(tag):
    """Convert an exifread IfdTag to the plain Python value piexif expects
    for the same EXIF field type (ASCII bytes, an (numerator, denominator)
    tuple for rationals, or a plain int)."""
    field_type = tag.field_type
    if field_type == 2:  # ASCII
        return str(tag.values).encode("utf-8")

    values = tag.values if isinstance(tag.values, list) else [tag.values]
    if not values:
        return None

    if field_type in _RATIONAL_TYPES:
        ratio = values[0]
        return (int(ratio.num), int(ratio.den))
    if field_type in _INT_TYPES:
        return int(values[0])
    return None


def read_exif(path: str) -> dict:
    """Reads the subset of EXIF fields this app preserves across
    import/export, as {tag_name: piexif-ready value}. Works on JPEG/TIFF
    and most TIFF-based RAW formats (CR2, NEF, ARW, DNG, ORF, RW2);
    non-TIFF RAW containers (CR3, RAF) commonly aren't parsed by exifread
    and simply yield no metadata rather than an error."""
    try:
        with open(path, "rb") as f:
            tags = exifread.process_file(f, details=False)
    except Exception:
        return {}

    result = {}
    for name in _TAG_MAP:
        tag = tags.get(name)
        if tag is None:
            continue
        value = _to_piexif_value(tag)
        if value is not None:
            result[name] = value
    return result


def write_exif(path: str, exif_data: dict):
    """Embeds previously-read EXIF fields into an exported JPEG/TIFF file
    that already exists on disk. No-op if there's nothing to write, or the
    target format doesn't support EXIF (PNG/WebP)."""
    if not exif_data:
        return
    if not path.lower().endswith((".jpg", ".jpeg", ".tif", ".tiff")):
        return

    exif_dict = {"0th": {}, "Exif": {}, "GPS": {}, "1st": {}, "thumbnail": None}
    for name, value in exif_data.items():
        mapping = _TAG_MAP.get(name)
        if mapping is None:
            continue
        ifd, tag_id = mapping
        exif_dict[ifd][tag_id] = value

    # The image has already been through this app's own crop/rotate tool,
    # so any orientation baked into the original EXIF no longer applies -
    # a stale Orientation tag would cause viewers to rotate an
    # already-correctly-oriented image a second time.
    exif_dict["0th"][piexif.ImageIFD.Orientation] = 1

    try:
        exif_bytes = piexif.dump(exif_dict)
        piexif.insert(exif_bytes, path)
    except Exception:
        pass  # best-effort - never fail an export over metadata
