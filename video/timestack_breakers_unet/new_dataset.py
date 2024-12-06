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
    
    brightness_factor = random.uniform(0.8, 1.2)
    contrast_factor = random.uniform(0.8, 1.2)
    gamma_factor = random.uniform(0.8, 1.2)
    
    image = TF.adjust_brightness(image, brightness_factor)
    image = TF.adjust_contrast(image, contrast_factor)
    image = TF.adjust_gamma(image, gamma_factor)
    return image


def apply_angular_brightness_contrast(image, p=0.25, blur_radius=5):
    """Apply brightness/contrast/gamma adjustment above a random angled line with smooth transition"""
    if random.random() > p:
        return image
    
    # Generate random angle between 45 and 90 degrees
    angle = random.uniform(45, 90)
    height, width = image.shape[1:3]
    
    # Create mask for the area above the line
    mask = torch.zeros((1, height, width))
    slope = -math.tan(math.radians(angle))
    
    # Calculate intercept to make line start from bottom of image
    intercept = height
    
    for y in range(height):
        for x in range(width):
            if y < slope * x + intercept:
                mask[0, y, x] = 1
    
    # Convert to PIL Image for Gaussian blur
    mask_pil = TF.to_pil_image(mask.squeeze())
    mask_blurred = TF.to_tensor(TF.gaussian_blur(mask_pil, blur_radius))
    
    # Apply transformations
    brightness_factor = random.uniform(0.8, 1.2)
    contrast_factor = random.uniform(0.8, 1.2)
    gamma_factor = random.uniform(0.8, 1.2)
    
    transformed_image = image.clone()
    transformed_image = TF.adjust_brightness(transformed_image, brightness_factor)
    transformed_image = TF.adjust_contrast(transformed_image, contrast_factor)
    transformed_image = TF.adjust_gamma(transformed_image, gamma_factor)
    
    # Blend original and transformed image using the blurred mask
    mask_blurred = mask_blurred.repeat(3, 1, 1)
    blended_image = image * (1 - mask_blurred) + transformed_image * mask_blurred
    
    return blended_image

from pprint import pprint

class DunexDataset(Dataset):
    def __init__(
        self, image_dir, labels_file, samples_per_image=3 * 16, patch_size=(128, 128), seed=42
    ):
        # Set global seeds
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.labels_file = labels_file
        self.samples_per_image = samples_per_image
        self.patch_size = patch_size
        self.seed = seed

        self.poly_labels = parse_polygons_from_xml(labels_file)
        try:
            self.poly_labels.pop('20211014T120100Z_t1553-2596_a0389.png') # Not sure why this is here but it shouldn't be
        except:
            pass
        self.image_files = [
            Path(image_dir) / Path(fname) for fname in self.poly_labels.keys()
        ]
        

        
    def __len__(self):
        return len(self.image_files) // 3 * self.samples_per_image

    def __getitem__(self, index):
        # Set seeds based on index for deterministic behavior per sample
        item_seed = self.seed + index
        random.seed(item_seed)
        torch.manual_seed(item_seed)
        np.random.seed(item_seed)
        
        file_number = index // (3 * self.samples_per_image)
        file = self.image_files[file_number + index % 3]
        
        image = TF.vflip(TF.to_tensor(Image.open(file)))
        labels = self.poly_labels[file.name]
        
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
                transforms.Lambda(lambda x: apply_angular_brightness_contrast(x, p=0.25))
            ]
        )

        image = base_tfs(grayscale_to_rgb(image))
        mask = base_tfs(mask)

        if random.random() < 0.5:  # 50% chance to apply augmentations
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