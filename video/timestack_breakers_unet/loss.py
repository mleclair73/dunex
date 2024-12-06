import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

def boundary_loss(pred, target):
    """Single pixel boundary loss"""
    pred = F.sigmoid(pred)
    
    with torch.no_grad():
        # Convert to numpy for easier pixel operations
        target_np = target.cpu().numpy()
        boundaries = torch.zeros_like(target)
        
        for b in range(target.shape[0]):
            mask = target_np[b].squeeze()
            
            # Convert to boolean array
            mask_bool = mask > 0.5
            
            # Get single pixel boundaries
            from scipy.ndimage import binary_dilation
            
            # Use cross/plus shaped kernel for single pixel boundary
            cross_kernel = np.array([[0,1,0],
                                     [1,1,1],
                                     [0,1,0]], dtype=bool)
            
            dilated = binary_dilation(mask_bool, structure=cross_kernel)
            # Convert both to boolean before XOR
            boundary = np.logical_xor(dilated, mask_bool)
            
            # Convert back to float for PyTorch
            boundaries[b] = torch.from_numpy(boundary.astype(np.float32)).to(target.device)
    
    # Calculate weighted BCE loss focusing on boundary pixels
    boundary_bce = F.binary_cross_entropy(pred, target, weight=boundaries.float(), reduction='mean')
    
    return boundary_bce

def visualize_boundaries(pred, target, phase='train', batch_idx=0):
    if batch_idx % 10 == 0:  # Only visualize every 10 batches
        with torch.no_grad():
            target_np = target[0].squeeze().cpu().numpy()
            
            # Convert to boolean array
            target_bool = target_np > 0.5
            
            # Get single pixel boundaries
            from scipy.ndimage import binary_dilation
            cross_kernel = np.array([[0,1,0],
                                   [1,1,1],
                                   [0,1,0]], dtype=bool)
            
            dilated = binary_dilation(target_bool, structure=cross_kernel)
            boundary = np.logical_xor(dilated, target_bool)

            plt.figure(figsize=(15, 5))
            plt.subplot(131)
            plt.imshow(target_np, cmap='gray')
            plt.title('Target')
            
            plt.subplot(132)
            plt.imshow(F.sigmoid(pred[0]).squeeze().cpu().numpy(), cmap='gray')
            plt.title('Prediction')
            
            plt.subplot(133)
            plt.imshow(boundary, cmap='hot')
            plt.title('Single Pixel Boundaries')
            
            plt.show()

def dice_loss(pred, target, smooth = 1.):
    pred = pred.contiguous()
    target = target.contiguous()    

    intersection = (pred * target).sum(dim=2).sum(dim=2)
    
    loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))
    
    return loss.mean()

def focal_loss(pred, target, smooth=1., alpha=0.25, gamma=2):
    pred = pred.contiguous()
    target = target.contiguous()    

    # Focal loss component
    bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
    pt = torch.exp(-bce)
    focal = alpha * (1-pt)**gamma * bce
    focal = focal.mean()
    return focal

def discriminitive_loss(logits, labels, temperature=0.05):  # Lower temperature for sharper predictions
    # Flatten both tensors
    logits_flat = logits.view(-1)
    labels_flat = labels.view(-1)
    
    # Apply temperature scaling (more aggressive)
    sharp_logits = logits_flat / temperature
    
    # Binary cross entropy with logits
    bce_loss = F.binary_cross_entropy_with_logits(sharp_logits, labels_flat)
    
    # Calculate probabilities
    probabilities = torch.sigmoid(logits_flat)
    
    # Add confidence penalty - encourage predictions to be closer to 0 or 1
    confidence_penalty = -torch.mean(torch.abs(probabilities - 0.5))
    
    # Calculate entropy (modified for binary case)
    prob_dist = torch.stack([1 - probabilities, probabilities], dim=1)
    entropy = -(prob_dist * torch.log(prob_dist + 1e-12)).sum(1).mean()
    
    # Combine losses with higher weight on confidence penalty
    return bce_loss + 0.2 * entropy + 0.3 * confidence_penalty


def discriminitive_loss(logits, labels, temperature=0.05):
    # Flatten both tensors
    logits_flat = logits.view(-1)
    labels_flat = labels.view(-1)
    
    # Get probabilities
    probs = torch.sigmoid(logits_flat)
    
    # Strong penalty for predictions in the middle range
    middle_range_penalty = torch.mean(
        torch.exp(-16 * torch.abs(probs - 0.5))  # Sharper exponential penalty
    )
    
    # Binary cross entropy for basic supervision
    bce_loss = F.binary_cross_entropy_with_logits(logits_flat, labels_flat)
    
    # Add spatial coherence penalty if needed
    spatial_penalty = 0
    if len(logits.shape) == 4:  # If working with images
        # Penalize different predictions for neighboring pixels
        probs_2d = probs.view(*logits.shape)  # Reshape back to [B,1,H,W]
        diff_x = torch.abs(probs_2d[:,:,:-1,:] - probs_2d[:,:,1:,:]).mean()
        diff_y = torch.abs(probs_2d[:,:,:,:-1] - probs_2d[:,:,:,1:]).mean()
        spatial_penalty = -(diff_x + diff_y)  # Encourage sharp transitions
    
    return bce_loss + 0.3 * middle_range_penalty + 0.1 * spatial_penalty


def calc_loss(pred, target, metrics, epoch=0, start_boundary_epoch=50, 
              max_boundary_weight=0.2, start_discriminative_epoch=30,
              max_discriminative_weight=0.2):
    """Loss function with gradient scaling and stability improvements"""
    # Calculate base losses with scaling
    pred_sigmoid = torch.clamp(F.sigmoid(pred), min=1e-7, max=1-1e-7)
    focal = focal_loss(pred, target, gamma=2.0, alpha=0.25)  # Reduced gamma and alpha
    dice = dice_loss(pred_sigmoid, target)
    
    # Initialize weights
    focal_weight = 0.5
    dice_weight = 0.5
    current_boundary_weight = 0.0
    current_discriminative_weight = 0.0
    
    # Gradual ramp-up for all components
    if epoch >= start_discriminative_epoch:
        disc_loss = discriminitive_loss(pred, target, temperature=0.1)  # Increased temperature
        ramp_epochs = 30  # Slower ramp-up
        current_discriminative_weight = max_discriminative_weight * min(1.0,
            (epoch - start_discriminative_epoch) / ramp_epochs)
    else:
        disc_loss = torch.tensor(0.0, device=pred.device)
    
    if epoch >= start_boundary_epoch:
        boundary = boundary_loss(pred, target)
        ramp_epochs = 100  # Much slower ramp-up
        current_boundary_weight = max_boundary_weight * min(1.0,
            (epoch - start_boundary_epoch) / ramp_epochs)
    else:
        boundary = torch.tensor(0.0, device=pred.device)
    
    # Ensure weights sum to 1.0 with smoother transitions
    total_special_weight = current_boundary_weight + current_discriminative_weight
    remaining_weight = 1.0 - total_special_weight
    
    focal_weight = 0.6 * remaining_weight  # Slightly higher weight on focal
    dice_weight = 0.4 * remaining_weight
    
    # Combine losses with gradient scaling
    loss = (focal * focal_weight + 
            dice * dice_weight + 
            boundary * current_boundary_weight +
            disc_loss * current_discriminative_weight)
    
    # Update metrics with scaled values
    metrics['focal'] += focal.data.cpu().numpy() * target.size(0)
    metrics['dice'] += dice.data.cpu().numpy() * target.size(0)
    metrics['loss'] += loss.data.cpu().numpy() * target.size(0)
    
    if epoch >= start_boundary_epoch:
        metrics['boundary'] += boundary.data.cpu().numpy() * target.size(0)
    if epoch >= start_discriminative_epoch:
        metrics['disc'] += disc_loss.data.cpu().numpy() * target.size(0)
    
    return loss
