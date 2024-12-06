import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
from PIL import Image
from pathlib import Path
from cvat_annotations.cvat_xml import parse_polygons_from_xml
from cvat_annotations.visualize_polygon_masks import create_boolean_mask
import random
import math
import numpy as np


def grayscale_to_rgb(img):
    """Convert a grayscale image to RGB."""
    return img.repeat(3, 1, 1)


def apply_brightness_contrast(image, p=0.25):
    """Apply brightness/contrast adjustment with fixed seed"""
    if random.random() > p:
        return image
    
    brightness_factor = random.uniform(0.4, 1.6)
    contrast_factor =   random.uniform(0.4, 1.6)
    gamma_factor =      random.uniform(0.6, 1.4)
    
    image = TF.adjust_brightness(image, brightness_factor)
    image = TF.adjust_contrast(image, contrast_factor)
    image = TF.adjust_gamma(image, gamma_factor)
    return image


def apply_vertical_gradient_transform(image, p=0.25, blur_radius=5):
    """Apply brightness/contrast/gamma adjustment above a random y-value with smooth transition"""
    if random.random() > p:
        return image
    
    height, width = image.shape[1:3]
    
    # Generate random y-value
    y_split = random.randint(height // 8, 7 * height // 8)
    
    # Create mask for the area above y_split
    mask = torch.zeros((1, height, width))
    mask[0, :y_split, :] = 1
    
    # Convert to PIL Image for Gaussian blur
    mask_pil = TF.to_pil_image(mask.squeeze())
    mask_blurred = TF.to_tensor(TF.gaussian_blur(mask_pil, blur_radius))
    
    # Apply transformations
    brightness_factor = random.uniform(0.25, 1.75)
    contrast_factor =   random.uniform(0.25, 1.75)
    gamma_factor =      random.uniform(0.8, 1.2)
    
    transformed_image = image.clone()
    transformed_image = TF.adjust_brightness(transformed_image, brightness_factor)
    transformed_image = TF.adjust_contrast(transformed_image, contrast_factor)
    transformed_image = TF.adjust_gamma(transformed_image, gamma_factor)
    
    # Blend original and transformed image using the blurred mask
    mask_blurred = mask_blurred.repeat(3, 1, 1)
    blended_image = image * (1 - mask_blurred) + transformed_image * mask_blurred
    
    return blended_image


def apply_angled_gradient_transform(image, p=0.25, blur_radius=5):
    """Apply gradually changing brightness/contrast/gamma along an angled direction, keeping one half unchanged"""
    if random.random() > p:
        return image
    
    height, width = image.shape[1:3]
    angle = random.uniform(-20, 20)
    
    # Create coordinate grids
    y, x = torch.meshgrid(torch.arange(height), torch.arange(width))
    
    # Calculate rotated coordinates
    angle_rad = math.radians(angle)
    x_rot = x * math.cos(angle_rad) + y * math.sin(angle_rad)
    
    # Normalize to [0, 1] and clip to keep one half unchanged
    x_rot = (x_rot - x_rot.min()) / (x_rot.max() - x_rot.min())
    mask = x_rot.unsqueeze(0)
    mask = torch.where(mask < 0.5, torch.zeros_like(mask), (mask - 0.5) * 2)
    
    # Smooth the transition
    mask_pil = TF.to_pil_image(mask)
    mask_blurred = TF.to_tensor(TF.gaussian_blur(mask_pil, blur_radius))
    
    # Apply transformations only to the changing half
    transformed_image = image.clone()
    transformed_image = TF.adjust_brightness(transformed_image, random.uniform(0.5, 1.5))
    transformed_image = TF.adjust_contrast(transformed_image,   random.uniform(0.5, 1.5))
    transformed_image = TF.adjust_gamma(transformed_image,      random.uniform(0.5, 1.5))
    
    # Blend using the gradient mask
    mask_blurred = mask_blurred.repeat(3, 1, 1)
    blended_image = image * (1 - mask_blurred) + transformed_image * mask_blurred
    
    return blended_image


class DunexDataset(Dataset):
    def __init__(
        self, label_files: list, samples_per_image=32, patch_size=(128, 128), seed=42
    ):
        # Set global seeds
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.label_files = label_files
        self.samples_per_image = samples_per_image
        self.patch_size = patch_size
        self.seed = seed

        all_labels = [parse_polygons_from_xml(labels) for labels in label_files]
        if any(image in seen or seen.add(image) for dataset_labels in all_labels for image, _ in dataset_labels.items() if (seen := set())):
            raise Exception(f'Duplicate Data in {label_files}')

        self.poly_labels = {image: labels for dataset_labels in all_labels for image, labels in dataset_labels.items()}
        self.image_files = [
            Path(fname) for fname in self.poly_labels.keys()
        ]
        
    def __len__(self):
        # Image files are in triplets because reasons
        return (len(self.image_files) // 3) * self.samples_per_image

    def __getitem__(self, index):
        # Set seeds based on index for deterministic behavior per sample
        item_seed = self.seed + index
        random.seed(item_seed)
        torch.manual_seed(item_seed)
        np.random.seed(item_seed)
        
        file = self.image_files[index // (self.samples_per_image)]
        
        image = TF.vflip(TF.to_tensor(Image.open(file)))
        labels = self.poly_labels[str(file)]
        
        mask = TF.vflip(TF.to_tensor(create_boolean_mask(
            labels["polygons"], labels["width"], labels["height"]
        )))
        
        max_y_index = torch.max(mask.squeeze().sum(-1).nonzero())

        # Crop out most of the blank part of the image
        image = image[:, :max_y_index + self.patch_size[0], :]
        mask = mask[:max_y_index + self.patch_size[0], :]
        
        i, j, h, w = transforms.RandomCrop.get_params(
            image, output_size=self.patch_size
        )
        crop = lambda image: TF.crop(image, i, j, h, w)

        base_tfs = transforms.Compose(
            [
                transforms.Lambda(crop)
            ]
        )

        augmentations = transforms.Compose(
            [
                transforms.Lambda(lambda x: apply_brightness_contrast(x, p=0.25)),
                transforms.Lambda(lambda x: apply_vertical_gradient_transform(x, p=0.4)),
                transforms.Lambda(lambda x: apply_angled_gradient_transform(x, p=0.2)),
            ]
        )

        image = base_tfs(grayscale_to_rgb(image))
        mask = base_tfs(mask)

        if random.random() < 0.25:  # 50% chance to apply augmentations
            image = augmentations(image)

        return image, mask


# For use with DataLoader, add this:
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# Usage with DataLoader:
"""
generator = torch.Generator()
generator.manual_seed(42)

dataloader = DataLoader(
    dataset,
    batch_size=32,
    num_workers=4,
    worker_init_fn=seed_worker,
    generator=generator
)
"""