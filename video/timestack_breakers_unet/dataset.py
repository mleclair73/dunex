import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
from glob import glob
from PIL import Image
from pathlib import Path
from cvat_annotations.cvat_xml import parse_polygons_from_xml
from cvat_annotations.visualize_polygon_masks import create_boolean_mask
import torch
import random
import math

random.seed(1337)
torch.manual_seed(17)


def grayscale_to_rgb(img):
    """Convert a grayscale image to RGB."""
    return img.repeat(3, 1, 1)


# Helper functions for augmentations
def add_gaussian_noise(image, mean=0, std=0.01, p=0.5):
    """Add random gaussian noise to the image"""
    if random.random() > p:
        return image

    # Just grayscale noise
    noise = torch.randn_like(image)[0] * std + mean
    noisy_image = image + noise
    return torch.clamp(noisy_image, 0, 1)


def add_local_intensity_changes(image, p=0.5, max_regions=3):
    """Add random local intensity changes to simulate water reflections/shadows"""
    if random.random() > p:
        return image

    h, w = image.shape[-2:]
    num_regions = random.randint(1, max_regions)

    for _ in range(num_regions):
        # Random region
        size = random.randint(10, 30)
        x = random.randint(0, w - size)
        y = random.randint(0, h - size)

        # Random intensity change
        factor = random.uniform(0.8, 1.2)

        # Apply change with smooth boundaries
        mask = torch.ones_like(image)
        for i in range(size):
            for j in range(size):
                # Create smooth falloff
                dist = ((i - size / 2) ** 2 + (j - size / 2) ** 2) / (size / 2) ** 2
                if dist <= 1:
                    alpha = 0.5 * (1 + math.cos(math.pi * math.sqrt(dist)))
                    mask[:, y + i, x + j] = 1 + (factor - 1) * alpha

        image = image * mask

    return torch.clamp(image, 0, 1)


class DunexDataset(Dataset):
    def __init__(
        self, image_dir, labels_file, samples_per_image=3 * 8, patch_size=(128, 128)
    ):
        self.image_dir = image_dir
        self.labels_file = labels_file
        self.samples_per_image = samples_per_image
        self.patch_size = patch_size

        self.poly_labels = parse_polygons_from_xml(labels_file)
        self.image_files = [
            Path(image_dir) / Path(fname) for fname in self.poly_labels.keys()
        ]
        
    def __len__(self):
        return len(self.image_files) // 3 * self.samples_per_image

    def __getitem__(self, index):
        file_number = index // (3 * self.samples_per_image)
        file = self.image_files[file_number + index % 3]
        image = TF.to_tensor(Image.open(file))
        labels = self.poly_labels[file.name]
        mask = create_boolean_mask(
            labels["polygons"], labels["width"], labels["height"]
        )
        min_y_index = np.min(np.nonzero(mask.sum(-1)))

        # To minimize the class imbalance, crop out most of the blank part of the image
        # In practice this can be done with a timex or similar
        image = image[:, min_y_index - self.patch_size[0] // 2 :, :]
        mask = mask[min_y_index - self.patch_size[0] // 2 :, :]
        i, j, h, w = transforms.RandomCrop.get_params(
            image, output_size=self.patch_size
        )
        crop = lambda image: TF.crop(image, i, j, h, w)
        identity = lambda image: TF.crop(image, 0, 0, 128, 128)

        base_tfs = transforms.Compose(
            [
                transforms.Lambda(crop)
                # transforms.Lambda(identity)
            ]
        )

        augmentations = transforms.Compose(
            [
                # These were cdalculated for the whole dataset
                # transforms.Normalize((123.15394544270833/255,), (50.414253020098215/255,)),
                # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), # imagenet
                # Random gamma adjustment
                transforms.RandomAdjustSharpness(sharpness_factor=1.5, p=0.3),
                # Gaussian noise
                transforms.Lambda(
                    lambda x: add_gaussian_noise(x, mean=0, std=0.05, p=0.3)
                ),
                # Random local changes
                transforms.Lambda(lambda x: add_local_intensity_changes(x, p=0.3)),
            ]
        )

        image = base_tfs(grayscale_to_rgb(image))
        mask = base_tfs(TF.to_tensor(mask))

        if random.random() < 0.5:  # 50% chance to apply augmentations
            image = augmentations(image)

        return image, mask


import numpy as np


def reverse_transform(img):
    mean = 123.15394544270833 / 255
    std = 50.414253020098215 / 255
    return ((img * std + mean) * 255).astype(np.uint8).squeeze(0)




import torch
import matplotlib.pyplot as plt
import math
from torchvision.utils import make_grid
import numpy as np

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

def plot_dataset_grid(dataset, num_images=16, figsize=(15,15), title="Dataset Samples", overlay=True, mask_color='red'):
    """
    Plot a grid of images and their masks from a dataset
    
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
    
    if overlay:
        fig, ax = plt.subplots(grid_size, grid_size, figsize=figsize)
        ax = ax.ravel()
        
        for idx in range(num_images):
            image, mask = dataset[idx]
            overlay_img = create_overlay(image, mask, mask_color=mask_color)
            ax[idx].imshow(overlay_img)
            ax[idx].axis('off')
            
    else:
        # Original two-panel version
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
        fig.suptitle(title, fontsize=16)
        
        # Get samples from dataset
        images = []
        masks = []
        
        for i in range(num_images):
            image, mask = dataset[i]
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

def plot_augmentation_comparison(dataset, num_images=4, figsize=(10,15), overlay=True, mask_color='red'):
    """
    Plot original and augmented versions of the same images
    
    Args:
        dataset: PyTorch dataset
        num_images: Number of images to show
        figsize: Size of the figure
        overlay: If True, show mask as overlay on image
        mask_color: Color of the mask overlay
    """
    if overlay:
        fig, axes = plt.subplots(num_images, 2, figsize=figsize)
        fig.suptitle('Augmentation Comparison', fontsize=16)
        
        for i in range(num_images):
            img1, mask1 = dataset[i*3]
            img2, mask2 = dataset[i*3+1]
            
            # Create overlays
            overlay1 = create_overlay(img1, mask1, mask_color=mask_color)
            overlay2 = create_overlay(img2, mask2, mask_color=mask_color)
            
            axes[i,0].imshow(overlay1)
            axes[i,1].imshow(overlay2)
            
            # Turn off axes
            axes[i,0].axis('off')
            axes[i,1].axis('off')
            
        # Add column labels
        # axes[0,0].set_title('Original Image + Mask')
        # axes[0,1].set_title('Augmented Image + Mask')
        
    else:
        # Original four-panel version
        fig, axes = plt.subplots(num_images, 4, figsize=figsize)
        fig.suptitle('Augmentation Comparison', fontsize=16)
        
        for i in range(num_images):
            img1, mask1 = dataset[i*3]
            img2, mask2 = dataset[i*3+1]
            
            if img1.shape[0] == 3:
                axes[i,0].imshow(img1.permute(1, 2, 0))
                axes[i,1].imshow(mask1.permute(1, 2, 0), cmap='gray')
                axes[i,2].imshow(img2.permute(1, 2, 0))
                axes[i,3].imshow(mask2.permute(1, 2, 0), cmap='gray')
            else:
                axes[i,0].imshow(img1.squeeze(), cmap='gray')
                axes[i,1].imshow(mask1.squeeze(), cmap='gray')
                axes[i,2].imshow(img2.squeeze(), cmap='gray')
                axes[i,3].imshow(mask2.squeeze(), cmap='gray')
            
            for ax in axes[i]:
                ax.axis('off')
        
        axes[0,0].set_title('Original Image')
        axes[0,1].set_title('Mask')
        axes[0,2].set_title('Augmented Image')
        axes[0,3].set_title('Mask')
    
    plt.tight_layout()
    plt.show()

def plot_single_sample(image, mask, figsize=(15,5), overlay=True, mask_color='red'):
    """
    Plot a single image and its mask
    
    Args:
        image: Tensor of shape [C, H, W]
        mask: Tensor of shape [1, H, W]
        figsize: Size of the figure
        overlay: If True, show mask as overlay on image
        mask_color: Color of the mask overlay
    """
    if overlay:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Plot original image
        if image.shape[0] == 3:
            ax1.imshow(image.permute(1, 2, 0))
        else:
            ax1.imshow(image.squeeze(), cmap='gray')
        ax1.axis('off')
        ax1.set_title('Original Image')
        
        # Plot overlay
        overlay = create_overlay(image, mask, mask_color=mask_color)
        ax2.imshow(overlay)
        ax2.axis('off')
        ax2.set_title('Image with Mask Overlay')
        
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        if image.shape[0] == 3:
            ax1.imshow(image.permute(1, 2, 0))
        else:
            ax1.imshow(image.squeeze(), cmap='gray')
        ax1.axis('off')
        ax1.set_title('Image')
        
        ax2.imshow(mask.squeeze(), cmap='gray')
        ax2.axis('off')
        ax2.set_title('Mask')
    
    plt.tight_layout()
    plt.show()