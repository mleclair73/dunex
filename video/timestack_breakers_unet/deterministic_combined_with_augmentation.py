import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
from PIL import Image
from pathlib import Path
import random
import math
import numpy as np
from cvat_annotations.cvat_xml import parse_polygons_from_xml
from cvat_annotations.visualize_polygon_masks import create_boolean_mask


class RandomState:
    """Class to manage random state consistently across augmentations"""

    def __init__(self, seed):
        self.rng = np.random.RandomState(seed)

    def random(self):
        return self.rng.random()

    def uniform(self, low, high):
        return self.rng.uniform(low, high)

    def randint(self, low, high):
        return self.rng.randint(low, high)
    
    def normal(self, loc, scale, size):
        return self.rng.normal(loc, scale, size)


def grayscale_to_rgb(img):
    """Convert a grayscale image to RGB."""
    return img.repeat(3, 1, 1)


def apply_brightness_contrast(image, random_state, p=0.25):
    """Apply brightness/contrast adjustment with controlled randomness"""
    if random_state.random() > p:
        return image

    brightness_factor = random_state.uniform(0.4, 1.6)
    contrast_factor =   random_state.uniform(0.4, 1.6)
    gamma_factor =      random_state.uniform(0.7, 1.3)

    image = TF.adjust_brightness(image, brightness_factor)
    image = TF.adjust_contrast(image, contrast_factor)
    image = TF.adjust_gamma(image, gamma_factor)
    return image


def apply_vertical_gradient_transform(image, random_state, p=0.25, blur_radius=5):
    """Apply brightness/contrast/gamma adjustment with controlled randomness"""
    if random_state.random() > p:
        return image

    height, width = image.shape[1:3]
    y_split = random_state.randint(height // 8, 7 * height // 8)

    mask = torch.zeros((1, height, width))
    mask[0, :y_split, :] = 1

    if random_state.random() < 0.5:
        mask = 1 - mask

    mask_pil = TF.to_pil_image(mask.squeeze())
    mask_blurred = TF.to_tensor(TF.gaussian_blur(mask_pil, blur_radius))

    brightness_factor = random_state.uniform(0.5, 1.5)
    contrast_factor =   random_state.uniform(0.5, 1.5)
    # gamma_factor =      random_state.uniform(0.1, 1.9)

    transformed_image = image.clone()
    transformed_image = TF.adjust_brightness(transformed_image, brightness_factor)
    # transformed_image = TF.adjust_contrast(transformed_image, contrast_factor)
    # transformed_image = TF.adjust_gamma(transformed_image, gamma_factor)

    mask_blurred = mask_blurred.repeat(3, 1, 1)
    blended_image = image * (1 - mask_blurred) + transformed_image * mask_blurred

    return blended_image


def apply_angled_gradient_transform(image, random_state, p=0.25, blur_radius=51):
    """Apply hard exposure edge transform with controlled randomness"""
    if random_state.random() > p:
        return image

    height, width = image.shape[1:3]

    point_x = random_state.uniform(width // 4, 3 * width // 4)
    point_y = random_state.uniform(height // 4, 3 * height // 4)
    angle = (1 if random_state.random() < 0.5 else -1) * random_state.uniform(25, 65)

    y, x = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")

    x = x - point_x
    y = y - point_y

    angle_rad = math.radians(angle)
    x_rot = x * math.cos(angle_rad) + y * math.sin(angle_rad)

    mask = x_rot.unsqueeze(0)
    mask = torch.where(
        ((1 if random_state.random() < 0.5 else -1) * mask) < 0,
        torch.zeros_like(mask),
        torch.ones_like(mask),
    )

    mask_pil = TF.to_pil_image(mask)
    mask_blurred = TF.to_tensor(TF.gaussian_blur(mask_pil, blur_radius))

    transformed_image = image.clone()
    transformed_image = TF.adjust_brightness(
        transformed_image, random_state.uniform(0.5, 1.5)
    )
    transformed_image = TF.adjust_contrast(
        transformed_image, random_state.uniform(0.5, 1.5)
    )
    transformed_image = TF.adjust_gamma(
        transformed_image, random_state.uniform(0.5, 1.5)
    )

    mask_blurred = mask_blurred.repeat(3, 1, 1)
    blended_image = image * (1 - mask_blurred) + transformed_image * mask_blurred

    return blended_image


def apply_missing_data_transform(image, random_state, p=0.25):
    """Apply missing data transform with controlled randomness"""
    if random_state.random() > p:
        return image

    height, width = image.shape[1:3]
    y_split = random_state.randint(height // 8, 7 * height // 8)

    upper_mask = torch.ones_like(image)
    lower_mask = torch.ones_like(image)

    overlap = 5
    switch = random_state.random()

    upper_interval = random_state.randint(2, 6)
    lower_interval = random_state.randint(2, 6)

    if random_state.random() > 0.5:
        for x in range(width):
            if switch < 0.33 or switch > 0.67:
                if x % upper_interval == random_state.randint(0, upper_interval - 1):
                    upper_mask[:, y_split - overlap :, x] = 0
            if switch > 0.33:
                if x % lower_interval == random_state.randint(0, upper_interval - 1):
                    lower_mask[:, : y_split + overlap, x] = 0
    else:
        upper_offset = random_state.randint(0, upper_interval - 1)
        lower_offset = random_state.randint(0, lower_interval - 1)
        for x in range(width):
            if switch < 0.33 or switch > 0.67:
                if x % upper_interval == upper_offset:
                    upper_mask[:, y_split - overlap :, x] = 0
            if switch > 0.33:
                if x % lower_interval == lower_offset:
                    lower_mask[:, : y_split + overlap, x] = 0

    lower_mask = TF.gaussian_blur(lower_mask, (1, 11))
    upper_mask = TF.gaussian_blur(upper_mask, (1, 11))

    return image * torch.minimum(upper_mask, lower_mask)


class DunexDataset(Dataset):
    def __init__(
        self,
        label_files: list,
        samples_per_image=32,
        patch_size=(128, 128),
        seed=42,
        use_missing_data=False,
    ):
        self.label_files = label_files
        self.samples_per_image = samples_per_image
        self.patch_size = patch_size
        self.seed = seed
        self.missing_data_augmentation = use_missing_data
        self.p_missing_data = 0.1
        self.p_annotation = 0.5


        # Set initial seeds
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        all_labels = [parse_polygons_from_xml(labels) for labels in label_files]
        if any(
            image in seen or seen.add(image)
            for dataset_labels in all_labels
            for image, _ in dataset_labels.items()
            if (seen := set())
        ):
            raise Exception(f"Duplicate Data in {label_files}")

        self.poly_labels = {
            image: labels
            for dataset_labels in all_labels
            for image, labels in dataset_labels.items()
        }
        self.image_files = [Path(fname) for fname in self.poly_labels.keys()]

    def __len__(self):
        # Images are in triplets fo rwhatever reason
        return len(self.image_files) // 3 * self.samples_per_image

    def __getitem__(self, index):
        # Create a deterministic random state for this item
        random_state = RandomState(self.seed + index)

        file = self.image_files[index // self.samples_per_image]
        image = TF.vflip(TF.to_tensor(Image.open(file)))
        labels = self.poly_labels[str(file)]

        mask = TF.vflip(
            TF.to_tensor(
                create_boolean_mask(
                    labels["polygons"], labels["width"], labels["height"]
                )
            )
        )

        # Crop image to mostly contain the labels
        max_y_index = torch.max(mask.squeeze().sum(-1).nonzero())
        image = image[:, : max_y_index + 7*self.patch_size[0]//8, :]
        mask = mask[: max_y_index + 7*self.patch_size[0]//8, :]

        # Use RandomState for crop parameters       
        h, w = self.patch_size
        th = image.shape[1] - h + 1
        tw = image.shape[2] - w + 1
        i = random_state.randint(0, th)
        j = random_state.randint(0, tw)

        crop = lambda image: TF.crop(image, i, j, h, w)
        base_tfs = transforms.Compose([transforms.Lambda(crop)])

        image = base_tfs(grayscale_to_rgb(image))
        mask = base_tfs(mask)

        if random_state.random() < self.p_annotation:
            # Apply augmentations with controlled random state
            image = apply_brightness_contrast(image, random_state,          p=0.25)
            image = apply_vertical_gradient_transform(image, random_state,  p=0.25)
            image = apply_angled_gradient_transform(image, random_state,    p=0.25)

            if self.missing_data_augmentation:
                image = apply_missing_data_transform(image, random_state, p=self.p_missing_data)

        return image, mask


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
