import os
import pathlib
import re
import sys
from glob import glob

import napari
import numpy as np
import pandas as pd
import typer
from napari.layers import Shapes
from nd2reader import ND2Reader
from pyometiff import OMETIFFReader
from rich import print as rprint
from rich import prompt
from skimage import exposure, transform
from typing import Union, Tuple
from pathlib import Path
from numpy.typing import NDArray

sys.path.append("/projectnb/bradham/workflows/indrops_hcr/scripts/")
import template_preprocess
import preprocess_images

COLOR_CYCLE = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]

# --------------------------------- Image IO --------------------------------- #
def get_ND2_image_data(nd_file: Union[str, Path]) -> Tuple[NDArray, Tuple]:
    """Read `.nd2` image file. See `get_image_data()` for more information."""
    images = ND2Reader(nd_file)
    channels = []
    for i in range(images.sizes["c"]):
        images.default_coords["c"] = i
        images.bundle_axes = ("z", "y", "x")
        channels.append(images.get_frame(0))
    zyx = (
        int(np.median(np.diff(images.metadata["z_coordinates"][::-1]))),
        *([images.metadata["pixel_microns"]] * 2),
    )
    arr = np.array(channels)
    return arr, zyx


def get_ome_tiff_image_data(im_file: str) -> Tuple[NDArray, Tuple]:
    """Read `.ome.tiff` image file. See `get_image_data()` for more information."""
    reader = OMETIFFReader(fpath=im_file)
    arr, metadata, xml_metadata = reader.read()
    zyx = (metadata["PhysicalSizeZ"], *([metadata["PhysicalSizeX"]] * 2))
    return arr, zyx


def get_image_data(im_file: str) -> Tuple[NDArray, Tuple]:
    """Read image data from microscope file

    Parameters
    ----------
    im_file : str
        Path to image file

    Returns
    -------
    Collection[NDArray, Tuple]
        Image data and physical pixel sizes
    """
    im_file = Path(im_file)
    if im_file.suffix == ".nd2":
        arr, dimensions = get_ND2_image_data(str(im_file))
    elif ".ome.tiff" in str(im_file):
        arr, dimensions = get_ome_tiff_image_data(im_file)
    else:
        raise ValueError(f"Unsupported image type: {im_file.suffix}")
    return arr, dimensions


def main(
    im_path: str,
    validate_landmarks: bool = False,
    segment: bool = False,
    pmc_channel: int = 0,
    z_start: int = 0,
    z_stop: int = -1,
) -> None:
    """Load image and view in napari.

    Parameters
    ----------
    im_path : str
        Path to image file.
    validate_landmarks : bool
        Whether to validate annotated landmarks.
    segment : bool, optional
        Whether to segment channels by the PMC stain, by default False
    pmc_channel : int, optional
        channel containng PMC stain, by default 0
    """
    stack, dimensions = get_image_data(im_path)
    if z_stop != -1:
        z_stop += 1
    stack = stack[:, z_start:z_stop, :, :]
    z_scale = dimensions[0] / dimensions[1]
    if segment:
        for z in range(stack.shape[1]):
            stack[pmc_channel, z, :, :] = preprocess_images.preprocess_pmc_slice(
                stack[pmc_channel, z, :, :],
                upper_percentile=99.99,
                new_min=0,
                new_max=1,
            )
        fg = template_preprocess.clean_pmc_stain(
            stack[pmc_channel, :, :, :],
            dimensions,
            entropy_percentile=99.9,
        )
        for i in range(stack.shape[0]):
            stack[i, :, :, :][~fg] = 0
    ext = ".nd2"
    landmark_file = pathlib.Path(im_path.replace(ext, "_points.csv"))
    if ".ome.tiff" in im_path:
        ext = ".ome.tiff"
        landmark_file = pathlib.Path(im_path.replace(ext, ".ome_points.csv"))

    # load image and any landmarks that exist
    viewer = napari.view_image(stack, channel_axis=0, scale=[z_scale, 1, 1])
    if landmark_file.exists():
        rprint(
            f"Loading landmarks from: [italic light_sky_blue1]{landmark_file}[/italic light_sky_blue1]"
        )
        landmarks = pd.read_csv(landmark_file, index_col=0).sort_values("label")
        data = landmarks[["axis-0", "axis-1", "axis-2"]].values
        current_properties = {"label": landmarks["label"].values.tolist()}
        points_layer = viewer.add_points(
            data=data,
            properties=current_properties,
            property_choices=current_properties,
            edge_color="label",
            edge_color_cycle=COLOR_CYCLE,
            symbol="o",
            face_color="transparent",
            edge_width=1,
            size=12,
            scale=[z_scale, 1, 1],
            n_dimensional=True,
            text="{label}",
            ndim=3,
        )
        points_layer.edge_color_mode = "cycle"
    napari.run()
    crops = [layer for layer in viewer.layers if isinstance(layer, Shapes)]
    for i, each in enumerate(crops):
        to_replace = ".csv"
        if i > 0:
            to_replace = f"({i}).csv"
        crop_file = pathlib.Path("crops").joinpath(
            im_path.replace("/", "_").replace(ext, to_replace)
        )
        rprint(
            f"Saving cropped selection to [italic light_sky_blue1]{crop_file}[/italic light_sky_blue1]"
        )
        each.save(str(crop_file))
    if landmark_file.exists() and validate_landmarks:
        swapped = False

        def swap_axes(x, ax1, ax2):
            if ax1 in x:
                return x.replace(ax1, ax2)
            elif ax2 in x:
                return x.replace(ax2, ax1)
            return x

        if prompt.Confirm.ask("Switch left-right coordinates?"):
            landmarks.label = landmarks.label.apply(
                lambda x: swap_axes(x, "left", "right")
            )
            swapped = True
        if prompt.Confirm.ask("Switch dorsal-ventral coordinates?"):
            landmarks.label = landmarks.label.apply(
                lambda x: swap_axes(x, "dorsal", "ventral")
            )
            swapped = True
        if swapped:
            rprint(
                f"Saving swapped points to [italic light_sky_blue1]{landmark_file}[/italic light_sky_blue1]"
            )
            landmarks.to_csv(landmark_file)


if __name__ == "__main__":
    typer.run(main)
