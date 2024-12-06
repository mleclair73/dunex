from torchvision.utils import make_grid
import math
import numpy as np
import matplotlib.pyplot as plt
import torch
import random
    
def create_overlay(image, mask, mask_color='red', alpha=0.5):
    """
    Create an overlay of the mask on the image
    
    Args:
        image: Tensor of shape [C, H, W]
        mask: Tensor of shape [1, H, W]
        mask_color: Color of the mask overlay (default: 'red')
        alpha: Opacity of the mask overlay (default: 0.5)
    """
    # Convert image to numpy and proper format
    if image.shape[0] == 3:
        img_np = image.permute(1, 2, 0).numpy()
    else:
        img_np = image.squeeze().numpy()
        img_np = np.stack([img_np]*3, axis=-1)  # Convert to RGB
    
    # Create colored mask
    mask_np = mask.squeeze().numpy()
    colored_mask = np.zeros_like(img_np)
    
    if mask_color == 'red':
        colored_mask[..., 0] = mask_np  # Red channel
    elif mask_color == 'green':
        colored_mask[..., 1] = mask_np  # Green channel
    elif mask_color == 'blue':
        colored_mask[..., 2] = mask_np  # Blue channel
    elif isinstance(mask_color, (list, tuple)) and len(mask_color) == 3:
        for i, color in enumerate(mask_color):
            colored_mask[..., i] = mask_np * color
    
    # Create mask alpha channel
    mask_alpha = np.zeros_like(mask_np)
    mask_alpha[mask_np > 0] = alpha
    
    # Blend image and mask
    overlay = img_np * (1 - mask_alpha[..., None]) + colored_mask * mask_alpha[..., None]
    return overlay


def plot_dataset_grid(dataset, num_images=16, figsize=(15,15), title="Dataset Samples", overlay=True, mask_color='red', indices=None):
    """
    Plot a grid of randomly sampled images and their masks from a dataset

    Args:
        dataset: PyTorch dataset
        num_images: Number of images to show (should be a perfect square)
        figsize: Size of the figure
        title: Title for the plot
        overlay: If True, show mask as overlay on image
        mask_color: Color of the mask overlay
    """
    # Calculate grid size
    grid_size = int(math.sqrt(num_images))
    
    # Generate random indices
    dataset_size = len(dataset)
    random_indices = indices if indices is not None else random.sample(range(dataset_size), num_images)

    if overlay:
        fig, ax = plt.subplots(grid_size, grid_size, figsize=figsize)
        fig.suptitle(title, fontsize=16)
        ax = ax.ravel()
        
        for idx, dataset_idx in enumerate(random_indices):
            image, mask = dataset[dataset_idx]
            overlay_img = create_overlay(image, mask, mask_color=mask_color)
            ax[idx].imshow(overlay_img)
            ax[idx].axis('off')
            
    else:
        # Original two-panel version
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
        fig.suptitle(title, fontsize=16)
        
        # Get random samples from dataset
        images = []
        masks = []
        
        for dataset_idx in random_indices:
            image, mask = dataset[dataset_idx]
            images.append(image)
            masks.append(mask)
        
        # Convert lists to tensors
        images = torch.stack(images)
        masks = torch.stack(masks)
        
        # Plot images
        if images.shape[1] == 3:
            image_grid = make_grid(images, nrow=grid_size, normalize=True)
            ax1.imshow(image_grid.permute(1, 2, 0))
        else:
            image_grid = make_grid(images, nrow=grid_size, normalize=True)
            ax1.imshow(image_grid.permute(1, 2, 0), cmap='gray')
        
        ax1.axis('off')
        ax1.set_title('Images')
        
        # Plot masks
        mask_grid = make_grid(masks, nrow=grid_size, normalize=True, pad_value=1)
        ax2.imshow(mask_grid.permute(1, 2, 0), cmap='gray')
        ax2.axis('off')
        ax2.set_title('Masks')

    plt.tight_layout()
    plt.show()