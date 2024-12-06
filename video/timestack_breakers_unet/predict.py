import torch
import numpy as np
from torch.nn import functional as F
from tqdm.auto import tqdm
import torch
import torchvision.transforms.functional as TF
from patchify import patchify, unpatchify

def predict_video_timestack(model, ts, patch_size=128, overlap=32, temporal_batch_size=16, spatial_batch_size=8, device='cpu'):
    """
    Predicts breakers in a timestack video using batched processing for both temporal and spatial dimensions.
    
    Args:
        model: trained UNet model
        ts: timestack video of shape (height, time, width)
        patch_size: size of spatial patches
        overlap: overlap between spatial patches
        temporal_batch_size: number of timesteps to process at once
        spatial_batch_size: number of spatial patches to process at once
        device: computation device
    
    Returns:
        predictions: array of same shape as input with predictions
    """
    import torch
    import numpy as np
    from torch.nn import functional as F
    from tqdm.auto import tqdm
    
    model.eval()
    
    height, time_steps, width = ts.shape
    print(f"Input shape: {ts.shape}")  # Debug print
    
    # Calculate spatial steps
    stride = patch_size - overlap
    h_steps = int(np.ceil((width - patch_size) / stride) + 1)  # Changed from height to width
    w_steps = int(np.ceil((height - patch_size) / stride) + 1)  # Changed from width to height
    
    # Calculate padding if needed
    pad_h = (h_steps - 1) * stride + patch_size - width  # Swapped height/width
    pad_w = (w_steps - 1) * stride + patch_size - height
    
    # Create weight matrix for blending
    weight = np.zeros((patch_size, patch_size))
    for i in range(patch_size):
        for j in range(patch_size):
            weight[i, j] = (1 - abs(i - patch_size/2)/patch_size) * (1 - abs(j - patch_size/2)/patch_size)
    weight = torch.from_numpy(weight).float().to(device)
    
    # Initialize output array
    predictions = np.zeros_like(ts)
    
    # Process in temporal batches
    for t_start in tqdm(range(0, time_steps, temporal_batch_size), desc='Processing timesteps'):
        t_end = min(t_start + temporal_batch_size, time_steps)
        current_temporal_batch_size = t_end - t_start
        
        # Initialize prediction accumulator for this temporal batch
        batch_prediction = torch.zeros((current_temporal_batch_size, width + pad_h, height + pad_w)).to(device)  # Swapped dimensions
        batch_weights = torch.zeros_like(batch_prediction).to(device)
        
        # Process spatial patches for this temporal batch
        patches = []
        positions = []
        timesteps = []
        
        for t_idx in range(current_temporal_batch_size):
            # Get normalized image for this timestep and flip vertically
            img = np.flip(ts[:, t_start + t_idx, :].T / 255.0, 0)
            
            # Pad if needed
            if pad_h > 0 or pad_w > 0:
                img = np.pad(img, ((0, pad_h), (0, pad_w)), mode='reflect')  # Swapped pad dimensions
            
            for i in range(h_steps):
                for j in range(w_steps):
                    h_start = i * stride
                    w_start = j * stride
                    patch = img[h_start:h_start + patch_size, w_start:w_start + patch_size]
                    
                    # Convert to tensor and add batch and channel dimensions
                    patch = torch.from_numpy(patch).float()
                    patch = patch.unsqueeze(0).unsqueeze(0)
                    patch = patch.repeat(1, 3, 1, 1)
                    
                    patches.append(patch)
                    positions.append((h_start, w_start))
                    timesteps.append(t_idx)
                    
                    # Process batch if it's full or last patch
                    if len(patches) == spatial_batch_size or (
                        t_idx == current_temporal_batch_size-1 and 
                        i == h_steps-1 and 
                        j == w_steps-1
                    ):
                        # Process batch
                        batch = torch.cat(patches).to(device)
                        with torch.no_grad():
                            batch_pred = model(batch)
                            batch_pred = torch.sigmoid(batch_pred)
                        
                        # Add predictions to output
                        for idx, ((h_start, w_start), t_idx) in enumerate(zip(positions, timesteps)):
                            pred = batch_pred[idx, 0]
                            batch_prediction[t_idx, h_start:h_start + patch_size, w_start:w_start + patch_size] += pred * weight
                            batch_weights[t_idx, h_start:h_start + patch_size, w_start:w_start + patch_size] += weight
                        
                        patches = []
                        positions = []
                        timesteps = []
        
        # Average overlapping regions
        batch_prediction = (batch_prediction / (batch_weights + 1e-8))
        
        # Crop to original size and store in output array
        for t_idx in range(current_temporal_batch_size):
            # Flip back vertically and handle dimensions correctly
            pred = batch_prediction[t_idx, :width, :height].cpu().numpy()
            pred = np.flip(pred, 0)
            predictions[:, t_start + t_idx, :] = pred.T
    
    return predictions

import numpy as np
import torch
import torch.nn.functional as F
from patchify import patchify, unpatchify

def predict_large_image_patchify(
    model, image, patch_size=128, overlap=32, batch_size=4, device="cuda", flip=False,
):
    """
    Predicts segmentation for a large image using patchify for sliding window.
    """
    model.eval()
    
    # Format image
    if flip:
        image = np.flip(image, 0)
    if len(image.shape) == 2:
        image = image[None, None, :, :]
    elif len(image.shape) == 3:
        image = image[None, :, :, :]
        
    if image.shape[1] == 1:
        image = np.repeat(image, 3, axis=1) if isinstance(image, np.ndarray) else image.repeat(1, 3, 1, 1)
    
    _, channels, height, width = image.shape
    stride = patch_size - overlap
    
    # Calculate padding
    h_padding = (stride - (height - patch_size) % stride) % stride
    w_padding = (stride - (width - patch_size) % stride) % stride
    
    padded_height = height + h_padding
    padded_width = width + w_padding
    
    # Pad image
    if h_padding > 0 or w_padding > 0:
        image = F.pad(torch.from_numpy(image) if isinstance(image, np.ndarray) else image, 
                     (0, w_padding, 0, h_padding), mode="reflect")
        image = image.cpu().numpy() if torch.is_tensor(image) else image
    
    # Prepare for patchify (H, W, C)
    image = image[0].transpose(1, 2, 0)
    
    # Extract patches and remove the extra dimension
    patches = patchify(image, (patch_size, patch_size, 3), step=stride).squeeze(2)
    
    # Process patches
    n_h, n_w = patches.shape[:2]
    total_patches = n_h * n_w
    patches = patches.reshape(total_patches, patch_size, patch_size, 3)
    patches = torch.from_numpy(patches.transpose(0, 3, 1, 2)).float().to(device)
    
    # Predict in batches
    predictions = []
    for i in range(0, total_patches, batch_size):
        batch = patches[i:i + batch_size]
        with torch.no_grad():
            pred = torch.sigmoid(model(batch))
            predictions.append(pred.cpu())
    
    predictions = torch.cat(predictions, dim=0)
    
    # Reshape predictions
    predictions = predictions.numpy()
    predictions = predictions.transpose(0, 2, 3, 1)  # (N, H, W, C)
    predictions = predictions.reshape(n_h, n_w, patch_size, patch_size, 1)
    
    # Create reconstruction
    reconstructed = np.zeros((padded_height, padded_width, 1))
    weight = np.zeros((padded_height, padded_width, 1))
    
    # Manually blend patches
    for i in range(n_h):
        for j in range(n_w):
            y = i * stride
            x = j * stride
            patch = predictions[i, j]
            patch_weight = np.ones_like(patch)
            
            reconstructed[y:y + patch_size, x:x + patch_size] += patch
            weight[y:y + patch_size, x:x + patch_size] += patch_weight
    
    # Average overlapping regions
    reconstructed = reconstructed / (weight + 1e-8)
    
    # Crop to original size
    result = reconstructed[:height, :width, 0]
    if flip:
        result = np.flip(result, 0)
    
    return result

def predict_large_image(
    model, image, patch_size=128, overlap=32, batch_size=4, device="cuda", flip=False,
):
    """
    Predicts segmentation for a large image using sliding window with overlap.

    Args:
        model: trained UNet model
        image: input image as numpy array (H, W) or (H, W, C)
        patch_size: size of patches to process (assumed square)
        overlap: number of pixels to overlap between patches
        batch_size: number of patches to process simultaneously
        device: device to run inference on

    Returns:
        full_prediction: segmentation mask for the entire image
    """
    model.eval()

    # Add batch and channel dims if needed
    if flip:
        image = np.flip(image, 0)
    if len(image.shape) == 2:
        image = image[None, None, :, :]
    elif len(image.shape) == 3:
        image = image[None, :, :, :]

    # Convert single channel to 3 channels if needed
    if image.shape[1] == 1:
        image = (
            image.repeat(1, 3, 1, 1)
            if torch.is_tensor(image)
            else np.repeat(image, 3, axis=1)
        )

    _, channels, height, width = image.shape

    # Calculate steps and pad image if necessary
    stride = patch_size - overlap
    h_steps = int(np.ceil((height - patch_size) / stride) + 1)
    w_steps = int(np.ceil((width - patch_size) / stride) + 1)

    pad_h = (h_steps - 1) * stride + patch_size - height
    pad_w = (w_steps - 1) * stride + patch_size - width

    if pad_h > 0 or pad_w > 0:
        image = F.pad(torch.from_numpy(image), (0, pad_w, 0, pad_h), mode="reflect")
    else:
        image = torch.from_numpy(image)

    # Create weight matrix for blending
    weight = np.zeros((patch_size, patch_size))
    for i in range(patch_size):
        for j in range(patch_size):
            weight[i, j] = (1 - abs(i - patch_size / 2) / patch_size) * (
                1 - abs(j - patch_size / 2) / patch_size
            )
    weight = torch.from_numpy(weight).float().to(device)

    # Initialize output arrays
    prediction = torch.zeros((1, 1, height + pad_h, width + pad_w)).to(device)
    weight_sum = torch.zeros((1, 1, height + pad_h, width + pad_w)).to(device)

    patches = []
    positions = []

    # Extract patches
    for i in range(h_steps):
        for j in range(w_steps):
            h_start = i * stride
            w_start = j * stride
            patch = image[
                :, :, h_start : h_start + patch_size, w_start : w_start + patch_size
            ]
            patches.append(patch)
            positions.append((h_start, w_start))

            # Process batch
            if len(patches) == batch_size or (i == h_steps - 1 and j == w_steps - 1):
                batch = torch.cat(patches).float().to(device)
                with torch.no_grad():
                    batch_pred = model(batch)
                    batch_pred = torch.sigmoid(batch_pred)

                # Add predictions to output
                for idx, (h_start, w_start) in enumerate(positions):
                    pred = batch_pred[idx : idx + 1]
                    prediction[
                        :,
                        :,
                        h_start : h_start + patch_size,
                        w_start : w_start + patch_size,
                    ] += (
                        pred * weight
                    )
                    weight_sum[
                        :,
                        :,
                        h_start : h_start + patch_size,
                        w_start : w_start + patch_size,
                    ] += weight

                patches = []
                positions = []

    # Average overlapping regions
    prediction = prediction / (weight_sum + 1e-8)

    # Crop to original size
    prediction = prediction[:, :, :height, :width]

    if flip:
        return np.flip(prediction.cpu().numpy()[0, 0], 0)
    else:
        return np.flip(prediction.cpu().numpy()[0, 0], 0)


def preprocess_image(img, device="cpu", flip=False):
    """
    Preprocess image for model input:
    1. Convert to tensor
    2. Add batch dimension
    3. Convert grayscale to RGB
    4. Ensure correct data type (float32)
    5. Move to correct device

    Args:
        img: Input image (numpy array)
        device: torch device to move tensor to
    Returns:
        Preprocessed tensor ready for model input
    """
    # Convert to tensor and add batch dimension if needed
    if flip:
        img = np.flip(img, 0)
    if not torch.is_tensor(img):
        img = TF.to_tensor(img / 255.0)

    # Ensure we have batch dimension
    if len(img.shape) == 2:
        img = img.unsqueeze(0)
    if len(img.shape) == 3:
        img = img.unsqueeze(0)

    # Convert to RGB if grayscale
    if img.shape[1] == 1:
        img = img.repeat(1, 3, 1, 1)

    # Ensure float32 data type
    img = img.float()

    # Move to device
    img = img.to(device)

    return img


def predict_patch(model, img, device="cpu"):
    """
    Make prediction for a single image patch

    Args:
        model: PyTorch model
        img: Input image (numpy array or tensor)
        device: torch device
    Returns:
        Prediction mask (numpy array)
    """
    model.eval()
    with torch.no_grad():
        # Preprocess image
        x = preprocess_image(img, device)

        # Make prediction
        pred = model(x)
        pred = torch.sigmoid(pred)

        # Convert to numpy array
        pred = pred.cpu().numpy()[0, 0]

    return pred
