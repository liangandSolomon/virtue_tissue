"""TIFF reading helpers tolerant of a missing ``imagecodecs`` package.

``tifffile.imread`` delegates JPEG / JPEG2000 decoding to the optional
``imagecodecs`` package. Whole-slide H&E images (Aperio, Hamamatsu, ...) are
commonly stored as JPEG compressed tiled TIFFs, so without that package the read
fails with::

    ValueError: <COMPRESSION.JPEG: 7> requires the 'imagecodecs' package

Pillow decodes those same files natively, so :func:`imread_tiff` falls back to
it instead of aborting the pipeline. ``imagecodecs`` is still preferred when
installed because it is faster.
"""

import numpy as np
import tifffile


def _is_missing_decoder_error(exc):
    """True when tifffile refused to decode because a codec is not installed."""
    msg = str(exc).lower()
    return "imagecodecs" in msg or "requires the" in msg


def _read_with_pillow(fp):
    """Decode the base (full resolution) level of a TIFF using Pillow."""
    from PIL import Image

    # Whole-slide images exceed Pillow's default decompression bomb limit.
    Image.MAX_IMAGE_PIXELS = None

    with Image.open(fp) as img:
        # Pyramidal slides expose each level as a frame; frame 0 is level 0.
        if getattr(img, "n_frames", 1) > 1:
            img.seek(0)
        if img.mode not in ("RGB", "RGBA", "L", "I", "I;16", "F"):
            img = img.convert("RGB")
        arr = np.asarray(img)

    if arr.ndim == 3 and arr.shape[-1] == 4:
        arr = arr[..., :3]

    return arr


def imread_tiff(fp, **kwargs):
    """Read a TIFF, falling back to Pillow when a codec is unavailable.

    Parameters
    ----------
    fp : str
        Path to the TIFF file.
    **kwargs
        Forwarded to ``tifffile.imread``.
    """
    try:
        return tifffile.imread(fp, **kwargs)
    except Exception as exc:
        if kwargs or not _is_missing_decoder_error(exc):
            raise
        try:
            arr = _read_with_pillow(fp)
        except Exception:
            # Neither decoder is usable: keep the original tifffile error
            # but point at the fix.
            raise ValueError(
                f"{exc}. Install imagecodecs (pip install imagecodecs) or "
                "Pillow to read compressed TIFFs"
            ) from None
        print(f"imagecodecs unavailable, read {fp} with Pillow instead")
        return arr