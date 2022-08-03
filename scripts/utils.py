from nd2reader import ND2Reader
import h5py
import numpy as np

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
    # I don't think this is the correct z-scale btw
    z_scale = raw_meta[b"SLxExperiment"][b"uLoopPars"][b"dZStep"]
    dimensions = np.array([z_scale, meta["pixel_microns"], meta["pixel_microns"]])
    if as_nm:
        dimensions = dimensions * 10 ** 3
    arr = np.array(channels)
    return arr, dimensions


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
