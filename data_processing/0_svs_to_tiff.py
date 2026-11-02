"""Convert a whole-slide image (.svs/.ndpi/.mrxs/.tif ...) into a plain TIFF.

GHIST expects the H&E image to be a plain RGB uint8 TIFF whose pixel grid
matches the nuclei segmentation mask, unlike a pyramidal whole-slide image.

Examples
--------
Native resolution, JPEG compressed, with pyramid levels:

    python 0_svs_to_tiff.py --fp_svs lung/slide.svs --fp_out lung/he_image.tif --pyramid 3

Downsample to a 0.5 MPP grid, lossless:

    python 0_svs_to_tiff.py --fp_svs slide.svs --fp_out he_image.tif --mpp 0.5 --compression deflate

Export a region of interest only (x y w h, at the read level):

    python 0_svs_to_tiff.py --fp_svs slide.svs --fp_out roi.tif --roi 4000 6000 4000 4000

Reading Aperio JPEG2000/J2K tiles requires either ``openslide`` or
``tifffile`` together with ``imagecodecs``; without them the tiles cannot be
decoded at all.
"""

import argparse
import os
import re
import sys

import numpy as np
import tifffile


class OpenSlideReader:
    """Tile-wise reader backed by openslide; memory efficient."""

    def __init__(self, fp):
        import openslide

        self.slide = openslide.OpenSlide(fp)
        self.name = os.path.basename(fp)
        self.level_dims = [tuple(int(v) for v in d) for d in self.slide.level_dimensions]
        self.level_downsamples = [float(d) for d in self.slide.level_downsamples]
        mpp_x = self.slide.properties.get(openslide.PROPERTY_NAME_MPP_X)
        mpp_y = self.slide.properties.get(openslide.PROPERTY_NAME_MPP_Y)
        self.mpp = (float(mpp_x), float(mpp_y)) if mpp_x and mpp_y else None

    def read_region(self, x, y, w, h, level):
        region = self.slide.read_region((int(x), int(y)), level, (int(w), int(h)))
        return np.asarray(region.convert("RGB"))

    def close(self):
        self.slide.close()


class TiffFileReader:
    """Fallback reader using tifffile + imagecodecs.

    Decodes a whole pyramid level into memory on first use, so full resolution
    level 0 of a large slide needs several GB of RAM.
    """

    def __init__(self, fp):
        self.tf = tifffile.TiffFile(fp)
        self.name = os.path.basename(fp)
        self.levels = list(self.tf.series[0].levels)
        self.level_dims = [tuple(int(v) for v in lv.shape[:2]) for lv in self.levels]
        self.level_downsamples = [self.level_dims[0][0] / d[0] for d in self.level_dims]
        self._cache = {}
        self.mpp = self._parse_mpp()

    def _parse_mpp(self):
        """Pull MPP out of the Aperio style ``...|MPP = 0.2520|...`` header."""
        tag = self.tf.pages[0].tags.get("ImageDescription")
        if tag is None:
            return None
        match = re.search(r"\|?MPP\s*=\s*([0-9.]+)", str(tag.value))
        if match:
            return (float(match.group(1)),) * 2
        return None

    def _level_array(self, level):
        if level not in self._cache:
            arr = self.levels[level].asarray()
            if arr.ndim == 2:
                arr = np.stack([arr] * 3, axis=-1)
            self._cache[level] = arr[..., :3]
        return self._cache[level]

    def read_region(self, x, y, w, h, level):
        arr = self._level_array(level)
        out = np.zeros((int(h), int(w), 3), dtype=np.uint8)
        y0, x0 = max(0, int(y)), max(0, int(x))
        y1 = min(y0 + int(h), arr.shape[0])
        x1 = min(x0 + int(w), arr.shape[1])
        if y1 > y0 and x1 > x0:
            out[y0 - int(y) : y1 - int(y), x0 - int(x) : x1 - int(x)] = arr[y0:y1, x0:x1]
        return out

    def close(self):
        self._cache.clear()
        self.tf.close()


def open_reader(fp):
    try:
        import openslide  # noqa: F401
    except ImportError:
        print("openslide not installed, falling back to tifffile")
    else:
        return OpenSlideReader(fp)

    try:
        import imagecodecs  # noqa: F401
    except ImportError:
        print(
            "warning: imagecodecs is missing, compressed tiles may fail to decode;"
            " install openslide or imagecodecs"
        )
    return TiffFileReader(fp)


def choose_level(reader, target_mpp, level):
    """Resolve which pyramid level to read."""
    n_levels = len(reader.level_dims)
    if level is not None:
        if not 0 <= level < n_levels:
            sys.exit("--level %d out of range (0..%d)" % (level, n_levels - 1))
        return level
    if target_mpp is None:
        return 0
    if reader.mpp is None:
        sys.exit("Slide carries no MPP metadata; pass --level instead of --mpp")

    base_mpp = reader.mpp[0]
    wanted = target_mpp / base_mpp
    if wanted < 1.0:
        sys.exit("--mpp %.4f is finer than the native %.4f MPP" % (target_mpp, base_mpp))

    best = min(
        range(n_levels),
        key=lambda i: abs(reader.level_downsamples[i] - wanted) / wanted,
    )
    return best


def compression_kwargs(compression, quality):
    if compression in (None, "none"):
        return {}
    if compression in ("jpeg", "jpeg2000"):
        return {"compression": compression, "compressionargs": {"level": quality}}
    return {"compression": compression}


def valid_tile(tile, w, h):
    """TIFF tile edge must be a multiple of 16 and not exceed the image."""
    edge = max(16, min(tile, w, h) // 16 * 16)
    return (edge, edge)


def plan_levels(reader, level, pyramid):
    """Map each output level to a slide level plus an integer decimation step.

    Reading a coarser slide level whenever its downsample matches the wanted
    resolution keeps the I/O small while staying exact.
    """
    base_ds = reader.level_downsamples[level]
    plan = []
    for lv in range(pyramid):
        target = base_ds * (2.0 ** lv)
        candidates = [
            (j, d) for j, d in enumerate(reader.level_downsamples) if d <= target + 1e-6
        ]
        j, d = max(candidates, key=lambda t: t[1]) if candidates else (level, base_ds)
        step = target / d
        plan.append((lv, j, max(1, int(round(step)))))
    return plan


def tile_stream(reader, slide_level, step, x0, y0, w, h, tile):
    """Yield ``tile`` sized output tiles for one pyramid level.

    ``step`` is the integer decimation applied to the region read at
    ``slide_level``; one output pixel therefore covers ``step`` read pixels.
    """
    tw, th = tile
    n_tx = -(-w // tw)
    n_ty = -(-h // th)
    for ty in range(n_ty):
        for tx in range(n_tx):
            rx = x0 + tx * tw * step
            ry = y0 + ty * th * step
            rw = min(tw * step, max(1, (w - tx * tw) * step))
            rh = min(th * step, max(1, (h - ty * th) * step))
            arr = reader.read_region(rx, ry, rw, rh, slide_level)
            if step > 1:
                arr = arr[::step, ::step]
            arr = arr[:th, :tw]
            if arr.shape[0] != th or arr.shape[1] != tw:
                pad = np.zeros((th, tw, 3), dtype=np.uint8)
                pad[: arr.shape[0], : arr.shape[1]] = arr
                arr = pad
            yield arr


def write_tiff(path, reader, level, x0, y0, w, h, tile, compression, quality, pyramid):
    tile = valid_tile(tile, w, h)
    kwargs = compression_kwargs(compression, quality)
    bigtiff = (w * h * 3) > 3.5 * 1024 ** 3

    with tifffile.TiffWriter(path, bigtiff=bigtiff) as tif:
        for out_lv, slide_level, step in plan_levels(reader, level, pyramid):
            cur_w = max(1, w // (2 ** out_lv))
            cur_h = max(1, h // (2 ** out_lv))
            if cur_w < tile[0] or cur_h < tile[1]:
                break
            tif.write(
                data=tile_stream(reader, slide_level, step, x0, y0, cur_w, cur_h, tile),
                shape=(cur_h, cur_w, 3),
                dtype=np.uint8,
                tile=(tile[1], tile[0]),
                photometric="rgb",
                subfiletype=0 if out_lv == 0 else 1,
                **kwargs,
            )
            print(
                "  level %d: %d x %d px (slide level %d, step %d)"
                % (out_lv, cur_w, cur_h, slide_level, step)
            )

def main(args):
    reader = open_reader(args.fp_svs)
    try:
        print("Slide      : %s" % reader.name)
        print("Levels     : %s" % (reader.level_dims,))
        print("Downsample : %s" % [round(d, 3) for d in reader.level_downsamples])
        print("MPP        : %s" % (reader.mpp,))

        level = choose_level(reader, args.mpp, args.level)
        dims = reader.level_dims[level]
        print("Read level %d -> %d x %d px" % (level, dims[1], dims[0]))

        if args.roi:
            x0, y0, w, h = args.roi
        else:
            x0, y0, w, h = 0, 0, dims[1], dims[0]

        w, h = min(w, dims[1] - x0), min(h, dims[0] - y0)
        if w <= 0 or h <= 0:
            sys.exit("Empty region after clipping to slide bounds")
        print("Output     : %d x %d px -> %s" % (h, w, args.fp_out))

        out_dir = os.path.dirname(os.path.abspath(args.fp_out))
        os.makedirs(out_dir, exist_ok=True)

        write_tiff(
            args.fp_out, reader, level, x0, y0, w, h,
            args.tile, args.compression, args.quality, args.pyramid,
        )
        print("Saved %s (%.1f MB)" % (args.fp_out, os.path.getsize(args.fp_out) / 1024 ** 2))
    finally:
        reader.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert a whole-slide image into a plain (optionally pyramidal) TIFF"
    )
    parser.add_argument("--fp_svs", required=True, help="path to the whole-slide image")
    parser.add_argument("--fp_out", required=True, help="output .tif path")
    parser.add_argument("--mpp", type=float, default=None,
                        help="target microns per pixel (default: native level 0)")
    parser.add_argument("--level", type=int, default=None,
                        help="read this pyramid level directly (overrides --mpp)")
    parser.add_argument("--roi", nargs=4, type=int, default=None, metavar=("X", "Y", "W", "H"),
                        help="region to export, at the read level")
    parser.add_argument("--tile", type=int, default=1024, help="tile size in px")
    parser.add_argument("--pyramid", type=int, default=1,
                        help="pyramid levels to write (1 = flat single level)")
    parser.add_argument("--compression", default="jpeg",
                        choices=["jpeg", "jpeg2000", "deflate", "lzw", "none"])
    parser.add_argument("--quality", type=int, default=90)
    main(parser.parse_args())