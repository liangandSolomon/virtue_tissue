import os
import sys

import imageio
from pathlib import Path
import numpy as np

# Shared reader that falls back to Pillow for TIFFs compressed with a codec
# missing from the environment (e.g. JPEG without imagecodecs).
sys.path.append(
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data_processing"
    )
)
from tiff_io import imread_tiff


def load_image(fp):
    if Path(fp).suffix.lower() in [".tif", ".ome.tif"]:
        image = imread_tiff(fp)
    else:
        image = imageio.imread(fp)

    if len(image.shape) == 4:
        image = image[:, :, :, 0]
    if image.shape[0] == 3:
        image = np.moveaxis(image, 0, -1)

    return image
