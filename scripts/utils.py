from pathlib import Path
from typing import Union

import h5py
import numpy as np
import oiffile
from nd2reader import ND2Reader
from numpy.typing import NDArray


# get image data from ND2 file
def get_ND2_image_data(nd_file, as_nm=False):
    """Retrieve 3D image data as well as the z-scaling factor a confocal images."""
    images = ND2Reader(nd_file)
    channels = []
    for i in range(images.sizes["c"]):
        images.default_coords["c"] = i
        images.bundle_axes = ("z", "y", "x")
        channels.append(images.get_frame(0))
    raw_meta = images.parser._raw_metadata.image_metadata
    meta = images.metadata
    z_scale = raw_meta[b"SLxExperiment"][b"uLoopPars"][b"dZStep"]
    # dimensions in micro meters
    dimensions = np.array([z_scale, meta["pixel_microns"], meta["pixel_microns"]])
    if as_nm:
        dimensions = dimensions * 10 ** 3
    arr = np.array(channels)
    return arr, dimensions


# get image data from oif file
def get_oif_physical_dimensions(oif, as_nm=False):
    info = oiffile.SettingsFile(oif)
    # Z, Y, X direction
    axes = [
        "Axis 3 Parameters Common",
        "Axis 1 Parameters Common",
        "Axis 0 Parameters Common",
    ]
    units_per_pxl = [0, 0, 0]
    for i, axis in enumerate(axes):
        units_per_pxl[i] = (
            info[axis]["EndPosition"] - info[axis]["StartPosition"]
        ) / info[axis]["MaxSize"]
        if info[axis]["PixUnit"] == "um" and as_nm:
            units_per_pxl[i] *= 1 / 10 ** 3
    return np.array(units_per_pxl)


def read_image_file(im_path, as_nm=False):
    limits = [0, 2 ** 12 - 1]
    if im_path.endswith(".oif"):
        images = oiffile.imread(im_path)
        dimensions = get_oif_physical_dimensions(im_path, as_nm)
    elif im_path.endswith(".nd2"):
        images, dimensions = get_ND2_image_data(im_path, as_nm)
    elif im_path.endswith(".h5"):
        images = read_h5ad(im_path)
        dimensions = None
    else:
        raise IOError("Unsupported file type!")
    print(images.shape)
    return images, dimensions


def read_h5ad(filename: Union[str, Path]) -> NDArray:
    """Read h5ad file saved via `write_h5ad()`"""
    dataset = h5py.File(filename, "r")
    return dataset.get("image")[:]


def get_channel_index(channels, channel):
    """Get numerical index for channel of interest by searching a string of ';' separated channel names"""
    print(channels, channel)
    channel_index = [
        i for i, x in enumerate(channels.split(";")) if x.lower() == channel.lower()
    ][0]
    assert int(channel_index) == channel_index
    return int(channel_index)


def to_hdf5(image, filename):
    "write image to .hdf5 file."
    f = h5py.File(filename, "w")
    dataset = f.create_dataset("image", image.shape, h5py.h5t.NATIVE_DOUBLE, data=image)
    f.close()


def pad_to_shape(img, shape):
    for i, s in enumerate(img.shape):
        assert s <= shape[i]

    def get_lr(size):
        if size % 2 == 0:
            return size // 2, size // 2
        else:
            return size // 2, size // 2 + 1

    pad_widths = [get_lr(new_l - l) for l, new_l in zip(img.shape, shape)]
    padded = np.pad(img, pad_widths, mode="constant", constant_values=0)
    assert padded.shape == shape
    return padded
