import numpy as np
import cv2
import xarray as xr
import pandas as pd
import scipy.ndimage as ndi
import os
from tqdm import tqdm

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import torch
import re


from base64 import b64encode
from IPython.display import HTML
def show_video(video_path):
    video_file = open(video_path, "r+b").read()
    video_url = f"data:video/mp4;base64,{b64encode(video_file).decode()}"
    return HTML(f"""<video width="640" height="480" autoplay loop controls><source src="{video_url}"></video>""")


def get_datetime(path):
    # \d{8} matches exactly 8 digits (YYYYMMDD)
    # T matches the literal 'T' character
    # \d{6} matches exactly 6 digits (HHMMSS)
    # Z matches the literal 'Z' character
    pattern = r'\d{8}T\d{6}Z'
    datetime_str = re.search(pattern, path).group()
    return datetime_str



def normalize(video, scale=255, pmin=0, pmax=99.9):
    video_tensor = torch.from_numpy(video).float()
    vmin, vmax = np.percentile(video, (pmin, pmax))
    normalized = scale * (video_tensor - vmin) / (vmax - vmin)
    return normalized.byte().numpy()


def image_frf_coords():
    x_frf = np.linspace(0, 500, 500)
    y_frf = np.linspace(1500, -100, 1600)
    return x_frf, y_frf


def get_video_shape(path=None, cap=None):
    if cap is None:
        cap = cv2.VideoCapture(path)
    N = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    FPS = int(cap.get(cv2.CAP_PROP_FPS))
    return N, W, H, FPS


def load_frames(file, stop=np.inf, dtype=np.int16, color=False):
    cap = cv2.VideoCapture(file)
    N, W, H, FPS = get_video_shape(cap=cap)

    frames = np.zeros((min(N, stop), H, W), dtype=dtype)
    print(f'{frames.size/1e9}gb allocated')
    for i in tqdm(range(min(N, stop)), desc="Building Timestack"):
        _, frame = cap.read()
        if not color:
            frames[i] = frame[..., 0]  # Saved as RGB but is grayscale

    cap.release()
    cv2.destroyAllWindows()
    return frames


def make_timestack(dir, file, stop=1000):
    return load_frames(os.path.join(dir, file), stop)


def image_to_frf(x, y):
    return x, 1500 - y * (1600 / 1500)

def show_breakers_on_frame(frame, i, breakers):
    if np.any(breakers[i] != 0):
        frame[ndi.binary_dilation(breakers[i]!=0, iterations=2)] = np.array(([0, 0, 255]))
    return frame

import numpy as np
import colorsys

def generate_distinct_colors(n_colors, shuffle=True):
    """
    Generate n_colors distinct colors using HSV color space.
    Returns RGB values scaled to 0-255 range.
    """
    colors = []
    for i in range(n_colors):
        # Use golden ratio to create well-distributed hues
        hue = i * (1.0 / n_colors)
        # Keep saturation and value high for visibility
        saturation = 0.8 + np.random.random() * 0.2
        value = 0.8 + np.random.random() * 0.2
        
        # Convert HSV to RGB
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        # Scale to 0-255 range
        rgb_255 = tuple(int(x * 255) for x in rgb)
        colors.append(rgb_255)
    
    colors = np.array(colors)
    
    if shuffle:
        np.random.shuffle(colors)
    
    return colors

def show_labels_on_frame(labels):
    """
    Optimized version that pre-computes color mapping
    """
    # Get unique labels once
    unique_labels = np.unique(labels)
    unique_labels = unique_labels[unique_labels != 0]

    # Generate color lookup table
    n_labels = unique_labels.max() + 1
    color_lut = np.zeros((n_labels, 3), dtype=np.uint8)
    colors = generate_distinct_colors(len(unique_labels))
    color_lut[unique_labels] = colors

    # Pre-compute structuring element for dilation
    struct = np.zeros((3, 3), dtype=bool)
    struct[1, :]=1
    struct[:, 1]=1

    def _show_labels_on_frame(frame, i, labels):
        # Create mask of all non-zero labels
        mask = labels[i] > 0
        # Dilate all labels at once
        dilated_mask = ndi.binary_dilation(mask, struct, iterations=3)

        # Create output frame
        out_frame = frame.copy()
        # Use advanced indexing to assign colors all at once
        out_frame[dilated_mask] = color_lut[labels[i][dilated_mask]]

        return out_frame

    return _show_labels_on_frame    

def write_sequence_to_mp4(
    sequence,
    fname,
    fn=None,
    data=None,
    FPS=2,
    transpose=True,
    color=True,
    lossless=False,
    fourcc=None
):
    H, W = sequence.shape[1:]
    draw = fn is not None and data is not None
    channels = 3 if color else 1
    if transpose:
        W, H = H, W
    if lossless:
        if '.avi' not in fname:
            raise Exception('Only lossless for avi')
        if fourcc is not None and fourcc != 'MPNG':
            raise Exception('Only MPNG')
        fourcc='MPNG'
    else:
        fourcc='avc1'
    out = cv2.VideoWriter(
        fname, cv2.VideoWriter_fourcc(*fourcc), FPS * 5, (W, H), color
    )
    for i in tqdm(range(sequence.shape[0])):
        if sequence.shape[-1] != channels:
            frame = np.stack((sequence[i].astype(np.uint8),) * channels, axis=-1)
        else:
            frame = sequence[i].astype(np.uint8)
        if draw:
            frame = fn(frame, i, data)
        if transpose:
            frame = np.flip(np.swapaxes(frame, 0, 1), 0)
        out.write(frame)
    out.release()
    cv2.destroyAllWindows()




def write_to_mp4_cmap(
    data, fname, FPS=2, H=1600, W=500, transpose=True, color=True, cmap="jet"
):
    def apply_cmap(data, vmin=25, vmax=50):
        # Create a normalize function to map values to [0, 1]
        norm = plt.Normalize(vmin, vmax)

        # Create the bwr colormap
        cmap = plt.get_cmap(cmap)

        # Apply the colormap to the normalized data
        rgba_colors = cmap(norm(data))

        # Convert RGBA to BGR
        bgr_colors = rgba_colors[:, :, [2, 1, 0]]  # Extract BGR channels

        # Scale to 0-255 range and convert to uint8
        bgr_colors = (bgr_colors * 255).astype(np.uint8)

        return bgr_colors

    write_sequence_to_mp4(data, fname, FPS=FPS, transpose=transpose, color=color)


def argus_sequence_to_xarray(sequence, name, fs=2, start=pd.Timestamp(0), times=None):
    # Convention is that image is of shape 1600, 500 (alongshore, cross-shore)
    T, H, W = sequence.shape
    assert (H, W) == (1600, 500)

    if times is None:
        ns_to_s = 1e9
        start_ns = start.value  # Convert to nanoseconds since epoch
        end_ns = start_ns + int(
            (T - 1) / fs * ns_to_s
        )  # Calculate end time in nanoseconds
        times = np.linspace(start_ns, end_ns, T).astype("datetime64[ns]")

        # times = (np.linspace(start, start + (T - 1) / fs, T) * ns_to_s).astype(
        #     "datetime64[ns]"
        # )  # .astype('datetime64[ns]')

    # Create coordinate arrays mirroring the xs and ys
    x_frf, y_frf = image_frf_coords()

    # Create the xarray DataArray
    da = xr.DataArray(
        data=sequence,
        dims=["time", "y_frf", "x_frf"],
        coords={
            "y_frf": (
                "y_frf",
                y_frf,
                {
                    "long_name": "Alongshore coordinate",
                    "units": "meters",
                    "description": "Y location in meters in the Duck FRF coordinate system",
                },
            ),
            "x_frf": (
                "x_frf",
                x_frf,
                {
                    "long_name": "Cross-shore coordinate",
                    "units": "meters",
                    "description": "X location in meters in the Duck FRF coordinate system",
                },
            ),
            "time": (
                "time",
                times,
                {
                    "long_name": "Time",
                    "description": "Time of image acquisition",
                    "units": "nanoseconds since 1970-01-01",
                },
            ),
        },
        name=name,
        attrs={"long_name": "Argus Image", "units": "pixel intensity"},
    )
    return da


def argus_image_to_xarray(image):
    # Convention is that image is of shape 1600, 500 (alongshore, cross-shore)
    assert image.shape == (1600, 500)

    # Create coordinate arrays mirroring the xs and ys
    x_frf, y_frf = image_frf_coords()

    # Create the xarray DataArray
    da = xr.DataArray(
        data=image,
        dims=["y_frf", "x_frf"],
        coords={
            "y_frf": (
                "y_frf",
                y_frf,
                {
                    "long_name": "Alongshore coordinate",
                    "units": "meters",
                    "description": "Y location in meters in the Duck FRF coordinate system",
                },
            ),
            "x_frf": (
                "x_frf",
                x_frf,
                {
                    "long_name": "Cross-shore coordinate",
                    "units": "meters",
                    "description": "X location in meters in the Duck FRF coordinate system",
                },
            ),
        },
        attrs={"long_name": "Argus Image", "units": "pixel intensity"},
    )

    return da


def plot_frf_image(image, ax=None, transpose=False, colorbar=True, levels=None):
    # Convention is that image is of shape 1600, 500 (alongshore, cross-shore)
    assert image.shape == (1600, 500)

    # Adjust the coordinate arrays to be one larger than the image array
    xs, ys = np.meshgrid(np.linspace(0, 500, 500), np.linspace(1500, -100, 1600))

    # image = np.flip(image, 1)
    xlabel = "Cross Shore [m]"
    ylabel = "Alnog Shore [m]"
    if transpose:
        image = image.T
        xs, ys = ys, xs  # Swap the coordinates for transposing
        xlabel, ylabel = ylabel, xlabel

    # Ensure xs and ys are one element larger than the image array
    nx = image.shape[1] + (1 if levels is None else 0)
    ny = image.shape[0] + (1 if levels is None else 0)

    xs = np.linspace(xs.min(), xs.max(), nx)
    ys = np.linspace(ys.min(), ys.max(), ny)

    if transpose:
        xs = np.flip(xs)
    else:
        ys = np.flip(ys)

    if ax is None:
        fig, ax = plt.subplots(figsize=(20, 5))

    # Use shading='flat' now that dimensions match the requirement
    if levels is not None:
        cax = ax.contourf(xs, ys, image, levels=levels)
    else:
        cax = ax.pcolormesh(xs, ys, image, shading="flat")
    ax.set_aspect("equal")
    ax.set_xlim(xs.min(), xs.max())
    ax.set_ylim(ys.min(), ys.max())

    # Removing any white space
    ax.set_adjustable("box")
    ax.set_xlim(xs.min(), xs.max())
    ax.set_ylim(ys.min(), ys.max())

    if transpose:
        ax.invert_xaxis()

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if ax is None:
        plt.show()

    if colorbar:
        # Create an axis on the right side of the plot to host the colorbar
        divider = make_axes_locatable(ax)
        cbar_ax = divider.append_axes(
            "right", size="2.5%", pad=0.05
        )  # Adjust size and pad as needed

        # Create the colorbar in the new axis
        cb = plt.colorbar(cax, cax=cbar_ax)
        cbar_ticks = np.linspace(
            image.min(), image.max(), num=5
        )  # Adjust num for desired number of ticks
        cb.set_ticks(cbar_ticks, labels=[f"{t:.3f}" for t in cbar_ticks])

    return cax
