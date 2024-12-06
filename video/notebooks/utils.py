import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from tqdm import tqdm
from scipy.ndimage import label
from sklearn.cluster import DBSCAN
from scipy.ndimage import measurements
from scipy import ndimage as ndi
from scipy.spatial.distance import cdist

# from video_io import *
# from video_io import *


def running_average_3d(arr, window_size=120):
    """
    Calculate the running average along the first axis of a 3D array.

    Args:
    arr (numpy.ndarray): Input 3D array
    window_size (int): Size of the moving window (default: 120)

    Returns:
    numpy.ndarray: 3D array with running averages
    """
    # Check if the input is a 3D array
    if arr.ndim != 3:
        raise ValueError("Input must be a 3D array")

    # Check if the first dimension is large enough for the window size
    if arr.shape[0] < window_size:
        raise ValueError(f"First dimension must be at least {window_size}")

    # Calculate the cumulative sum along the first axis
    cumsum = np.cumsum(arr, axis=0).astype(np.float64)

    # Calculate the running average
    result = np.empty_like(arr).astype(np.float64)
    result[:window_size] = (
        cumsum[:window_size] / np.arange(1, window_size + 1)[:, np.newaxis, np.newaxis]
    )
    result[window_size:] = (cumsum[window_size:] - cumsum[:-window_size]) / window_size
    return result


def running_qb(arr, Tp, FPS=2, window_size=120):
    """
    Calculate the running average along the first axis of a 3D array.

    Args:
    arr (numpy.ndarray): Input 3D array
    window_size (int): Size of the moving window (default: 120)

    Returns:
    numpy.ndarray: 3D array with running averages
    """
    # Check if the input is a 3D array
    return running_average_3d(arr) * FPS / Tp


def lambda_c_algorithm(timestack):
    N, M, _ = timestack.shape
    result = np.zeros_like(timestack, dtype=bool)

    for i in tqdm(range(M), desc="Processing alongshore"):
        diff = timestack[1:, i] - timestack[:-1, i]
        result[:, i, :] = remove_loose_pixels_not_in_line(
            detect_leading_edges(
                # Threshold on brightness
                timestack[:, i] >= np.percentile(timestack[:, i], 95),
                axis=0,
            ),
            min_line_length=3,
            eps=5,
            min_samples=5,
        )

        # And optionally on gradient? This is picking up lots of stuff offshore which isn't ideal
        # result[1:, i, :] |= remove_loose_pixels_not_in_line(diff > np.percentile(diff, 90, axis=1)[:, np.newaxis], 15)

    for i in tqdm(range(N), desc="Processing time"):
        result[i] = remove_loose_pixels_not_in_line(
            detect_leading_edges(result[i], axis=1),
            min_line_length=3,
            eps=3,
            min_samples=5,
        )

    # for i in tqdm(range(N), desc="Removing loose pixels"):
    #     result[i] = remove_loose_pixels(result[i], 5)

    breaker_labels = filter_and_label_spatiotemporal_propagation(
        result, min_size=6, time_window=3, space_window=5
    )

    return breaker_labels != 0


def detect_leading_edges(binary_array, axis=0):
    if axis not in [0, 1]:
        raise ValueError("Axis must be 0 or 1.")

    # Shift the array along the specified axis
    shifted_array = np.roll(binary_array, 1, axis=axis)

    # Detect leading edges: where the original array is 1 and the shifted array is 0
    leading_edges = (binary_array == 1) & (shifted_array == 0)

    # Set the first row/column to False to avoid false detection due to wrapping from np.roll
    if axis == 0:
        leading_edges[0, :] = False
    else:
        leading_edges[:, 0] = False

    return leading_edges


def remove_loose_pixels(binary_image, min_size=1):
    s = np.ones((3, 3), dtype=int)
    labels, num_labels = ndi.label(binary_image, structure=s)
    component_sizes = np.bincount(labels.ravel())
    mask = component_sizes >= min_size
    mask[0] = 0  # Ensure background (label 0) is always removed
    return mask[labels]


def remove_loose_pixels_old(binary_image, min_size=1):
    """
    Remove any loose pixels (isolated pixels) from a binary image.

    Args:
        binary_image (numpy.ndarray): A 2D binary image.

    Returns:
        numpy.ndarray: A 2D binary image with loose pixels removed.
    """
    # Label the connected components
    labels, num_labels = measurements.label(binary_image)

    # Get the sizes of the connected components
    component_sizes = np.bincount(labels.ravel())[1:]

    # Create a new binary image with loose pixels removed
    cleaned_image = np.zeros_like(binary_image)
    cleaned_image[labels > 0] = component_sizes[labels[labels > 0] - 1] > min_size

    return cleaned_image


def remove_loose_pixels_not_in_line(
    binary_image, min_line_length=3, eps=1, min_samples=2
):
    """
    Remove loose pixels (isolated pixels) that are not part of a line in a binary image.
    A line can be continuous or discontinuous, and at any angle, but must have a minimum length.

    Args:
        binary_image (numpy.ndarray): A 2D binary image.
        min_line_length (int): The minimum length of a line, default is 3.
        eps (float): The maximum distance between two samples for them to be considered as in the same neighborhood (DBSCAN parameter).
        min_samples (int): The number of samples (or total weight) in a neighborhood for a point to be considered as a core point (DBSCAN parameter).

    Returns:
        numpy.ndarray: A 2D binary image with loose pixels not in lines removed.
    """
    # Label the connected components
    labels, num_labels = label(binary_image)

    # Get the indices of the non-zero elements
    row_indices, col_indices = np.nonzero(binary_image)

    if len(row_indices) == 0:
        return np.zeros_like(binary_image)

    # Perform DBSCAN to cluster the pixels
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(
        np.column_stack((row_indices, col_indices))
    )

    # Create a new binary image with loose pixels not in lines removed
    cleaned_image = np.zeros_like(binary_image)

    # Iterate over the clusters
    for cluster_id in np.unique(clustering.labels_):
        if cluster_id == -1:
            # Noise points
            continue

        cluster_mask = clustering.labels_ == cluster_id
        # TODO(@mleclair), instead do a disstance calculation between the points
        # This is just min_samples twice
        if np.sum(cluster_mask) >= min_line_length:
            # The cluster is part of a line
            cleaned_image[row_indices[cluster_mask], col_indices[cluster_mask]] = 1

    return cleaned_image


def filter_and_label_spatiotemporal_propagation(
    binary_video, min_size=1, time_window=3, space_window=3
):
    """
    Filter a binary video to keep only pixels that show roughly continuous propagation in space and time,
    and label the features consistently through space and time.

    Args:
        binary_video (numpy.ndarray): A 3D binary array where dimensions are (time, height, width).
        min_size (int): The minimum size of connected components to keep.
        time_window (int): The number of frames to consider for temporal continuity.
        space_window (int): The spatial window size to consider for propagation.

    Returns:
        numpy.ndarray: A 3D integer array with labeled features.
    """
    filtered_video = np.zeros_like(binary_video, dtype=int)
    time_frames, height, width = binary_video.shape

    # Define 3D structure element for spatiotemporal connectivity
    s = np.ones((min(time_window, 3), 3, 3))

    next_label = 1
    feature_history = {}  # Dictionary to store feature centroids and labels

    for t in tqdm(range(time_frames)):
        # Extract temporal window
        start_t = max(0, t - time_window // 2)
        end_t = min(time_frames, t + time_window // 2 + 1)
        temp_window = binary_video[start_t:end_t]

        # Label connected components in the temporal window
        labels, _ = ndi.label(temp_window, structure=s)

        # Get sizes of connected components
        component_sizes = np.bincount(labels.ravel())[1:]

        # Filter based on size
        size_mask = np.zeros_like(labels)
        size_mask[labels > 0] = component_sizes[labels[labels > 0] - 1] >= min_size

        # Check for spatial propagation
        prop_mask = np.zeros_like(size_mask)
        current_frame_index = t - start_t  # Index of current frame within the window

        for y in range(height):
            for x in range(width):
                if size_mask[current_frame_index, y, x]:
                    # Check neighborhood in previous and next frames if they exist
                    has_prev = current_frame_index > 0
                    has_next = current_frame_index < size_mask.shape[0] - 1

                    prev_neighborhood = (
                        size_mask[
                            current_frame_index - 1,
                            max(0, y - space_window // 2) : min(
                                height, y + space_window // 2 + 1
                            ),
                            max(0, x - space_window // 2) : min(
                                width, x + space_window // 2 + 1
                            ),
                        ]
                        if has_prev
                        else np.array([])
                    )

                    next_neighborhood = (
                        size_mask[
                            current_frame_index + 1,
                            max(0, y - space_window // 2) : min(
                                height, y + space_window // 2 + 1
                            ),
                            max(0, x - space_window // 2) : min(
                                width, x + space_window // 2 + 1
                            ),
                        ]
                        if has_next
                        else np.array([])
                    )

                    # Propagation condition: either previous or next frame should have an active pixel
                    if np.any(prev_neighborhood) or np.any(next_neighborhood):
                        prop_mask[current_frame_index, y, x] = 1

        # Apply both size and propagation filters
        current_frame = size_mask[current_frame_index] & prop_mask[current_frame_index]

        # Label the current frame
        current_labels, num_features = ndi.label(current_frame)

        # Calculate centroids for current features
        current_centroids = ndi.center_of_mass(
            current_frame, current_labels, range(1, num_features + 1)
        )

        # Create a new labeling for the current frame
        new_labeling = np.zeros_like(current_labels)

        if t > 0:
            # Get centroids from previous frame
            prev_centroids = [
                feature_history[label]["centroid"]
                for label in feature_history
                if feature_history[label]["last_frame"] == t - 1
            ]
            prev_labels = [
                label
                for label in feature_history
                if feature_history[label]["last_frame"] == t - 1
            ]

            if prev_centroids:
                # Calculate distances between current and previous centroids
                distances = cdist(current_centroids, prev_centroids)

                # Match current features to previous features based on minimum distance
                for i, centroid in enumerate(current_centroids):
                    if len(prev_centroids) > 0:
                        min_dist_index = np.argmin(distances[i])
                        if distances[i][min_dist_index] < space_window:
                            matched_label = prev_labels[min_dist_index]
                            mask = current_labels == i + 1
                            new_labeling[mask] = matched_label
                            feature_history[matched_label] = {
                                "centroid": centroid,
                                "last_frame": t,
                            }
                            distances[:, min_dist_index] = (
                                np.inf
                            )  # Prevent this previous feature from being matched again
                        else:
                            # No close match, assign a new label
                            mask = current_labels == i + 1
                            new_labeling[mask] = next_label
                            feature_history[next_label] = {
                                "centroid": centroid,
                                "last_frame": t,
                            }
                            next_label += 1
                    else:
                        # No previous features left to match, assign a new label
                        mask = current_labels == i + 1
                        new_labeling[mask] = next_label
                        feature_history[next_label] = {
                            "centroid": centroid,
                            "last_frame": t,
                        }
                        next_label += 1
            else:
                # No previous features, assign new labels to all
                for i, centroid in enumerate(current_centroids):
                    mask = current_labels == i + 1
                    new_labeling[mask] = next_label
                    feature_history[next_label] = {
                        "centroid": centroid,
                        "last_frame": t,
                    }
                    next_label += 1
        else:
            # First frame, assign new labels to all
            for i, centroid in enumerate(current_centroids):
                mask = current_labels == i + 1
                new_labeling[mask] = next_label
                feature_history[next_label] = {"centroid": centroid, "last_frame": t}
                next_label += 1

        filtered_video[t] = new_labeling

        # Remove labels that haven't been used in the last 'time_window' frames
        for label in list(feature_history.keys()):
            if t - feature_history[label]["last_frame"] >= time_window:
                del feature_history[label]

    return filtered_video
