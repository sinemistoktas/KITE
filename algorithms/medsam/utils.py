"""
MedSAM Utilities Module
Helper functions for MedSAM preprocessing and visualization.
"""

import os
import numpy as np
import torch
from skimage import io, transform
from PIL import Image
import logging

logger = logging.getLogger(__name__)

def get_model_path():
    """Get path to the MedSAM model checkpoint."""
    try:
        from django.conf import settings
        return getattr(settings, 'MEDSAM_MODEL_PATH', os.path.join(
            settings.BASE_DIR, 'models', 'medsam_vit_b.pth'
        ))
    except ImportError:
        # Fallback if Django is not available
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        return os.path.join(base_dir, 'models', 'medsam_vit_b.pth')

def preprocess_image(image_path, device):
    """
    Preprocess an image for MedSAM inference.
    
    Args:
        image_path: Path to the image file
        device: PyTorch device
        
    Returns:
        tuple: (img_1024_tensor, original_img, H, W)
            - img_1024_tensor: Preprocessed tensor ready for model
            - original_img: Original image array
            - H, W: Original image dimensions
    """
    try:
        # Get absolute path if needed
        if not os.path.isabs(image_path):
            try:
                from django.conf import settings
                abs_image_path = os.path.join(settings.MEDIA_ROOT, image_path)
            except ImportError:
                abs_image_path = image_path
        else:
            abs_image_path = image_path
            
        # Load image
        img_np = io.imread(abs_image_path)
        
        # Store original for overlay creation
        original_img = img_np.copy()
        
        # OCT-specific preprocessing - enhance contrast
        p2, p98 = np.percentile(img_np, (2, 98))
        img_contrast = np.clip((img_np - p2) / (p98 - p2), 0, 1)
        
        # Convert to 3-channel if grayscale
        if len(img_contrast.shape) == 2:
            img_3c = np.repeat(img_contrast[:, :, None], 3, axis=-1)
        else:
            img_3c = img_contrast
            
        H, W, _ = img_3c.shape
        
        # Resize to 1024x1024 for MedSAM
        img_1024 = transform.resize(
            img_3c, (1024, 1024), order=3, preserve_range=True, anti_aliasing=True
        ).astype(np.uint8)
        
        # Normalize
        img_1024 = (img_1024 - img_1024.min()) / np.clip(
            img_1024.max() - img_1024.min(), a_min=1e-8, a_max=None
        )
        
        # Convert to tensor
        img_1024_tensor = torch.tensor(img_1024).float().permute(2, 0, 1).unsqueeze(0).to(device)
        
        return img_1024_tensor, original_img, H, W
        
    except Exception as e:
        logger.error(f"Error preprocessing image: {str(e)}")
        raise

def create_overlay(original_img, binary_mask, color=[255, 0, 0], alpha=0.2):
    """
    Create an overlay of the segmentation mask on the original image.
    
    Args:
        original_img: Original image array
        binary_mask: Binary segmentation mask
        color: RGB color for the mask overlay
        alpha: Transparency factor for blending
        
    Returns:
        numpy.ndarray: RGB image with segmentation overlay
    """
    try:
        # Ensure original image is 3-channel
        if len(original_img.shape) == 2:
            img_3c = np.repeat(original_img[:, :, None], 3, axis=-1)
        else:
            img_3c = original_img.copy()
        
        # Ensure values are in [0, 255] range
        if img_3c.max() <= 1.0:
            img_3c = (img_3c * 255).astype(np.uint8)
        else:
            img_3c = img_3c.astype(np.uint8)
        
        # Create colored mask
        mask_color = np.zeros_like(img_3c)
        mask_color[binary_mask != 0] = color
        
        # Blend images
        bg = Image.fromarray(img_3c, "RGB")
        mask_img = Image.fromarray(mask_color.astype(np.uint8), "RGB")
        overlay = Image.blend(bg, mask_img, alpha)
        
        return np.array(overlay)
        
    except Exception as e:
        logger.error(f"Error creating overlay: {str(e)}")
        raise

def validate_box(box, img_height, img_width):
    """
    Validate and clip bounding box coordinates.
    
    Args:
        box: Bounding box [x1, y1, x2, y2]
        img_height: Image height
        img_width: Image width
        
    Returns:
        list: Validated bounding box
    """
    if box is None:
        return None
    
    x1, y1, x2, y2 = box
    
    # Ensure coordinates are within image bounds
    x1 = max(0, min(x1, img_width))
    y1 = max(0, min(y1, img_height))
    x2 = max(x1, min(x2, img_width))
    y2 = max(y1, min(y2, img_height))
    
    # Ensure minimum box size
    min_size = 10
    if x2 - x1 < min_size:
        x2 = min(x1 + min_size, img_width)
    if y2 - y1 < min_size:
        y2 = min(y1 + min_size, img_height)
    
    return [x1, y1, x2, y2]

def save_results(overlay_img, binary_mask, output_dir, base_filename):
    """
    Save segmentation results to files.
    
    Args:
        overlay_img: RGB overlay image
        binary_mask: Binary segmentation mask
        output_dir: Directory to save results
        base_filename: Base filename for output files
        
    Returns:
        tuple: (overlay_path, mask_path)
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        # Save overlay
        overlay_filename = f"medsam_overlay_{base_filename}.png"
        overlay_path = os.path.join(output_dir, overlay_filename)
        Image.fromarray(overlay_img).save(overlay_path)
        
        # Save mask
        mask_filename = f"medsam_mask_{base_filename}.png"
        mask_path = os.path.join(output_dir, mask_filename)
        mask_img = (binary_mask * 255).astype(np.uint8)
        Image.fromarray(mask_img).save(mask_path)
        
        return overlay_path, mask_path
        
    except Exception as e:
        logger.error(f"Error saving results: {str(e)}")
        raise