"""
MedSAM Inference Module
Handles model loading and segmentation inference for MedSAM.
"""

import os
import numpy as np
import torch
from segment_anything import sam_model_registry
from skimage import io, transform
import torch.nn.functional as F
from PIL import Image
import logging
from .utils import get_model_path, preprocess_image, create_overlay

logger = logging.getLogger(__name__)

# Initialize the device
if torch.cuda.is_available():
    device = torch.device("cuda:0")
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = torch.device("cpu")  # Default to CPU for stability
else:
    device = torch.device("cpu")

logger.info(f"MedSAM using device: {device}")

# Global model cache
_medsam_model = None

def load_model():
    """Load the MedSAM model if not already loaded."""
    global _medsam_model
    if _medsam_model is None:
        logger.info("Loading MedSAM model...")
        try:
            model_path = get_model_path()
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"MedSAM model not found at {model_path}")
            
            # Load the checkpoint manually and map it to CPU
            state_dict = torch.load(model_path, map_location='cpu')
            
            # Init model as empty
            _medsam_model = sam_model_registry["vit_b"]()
            
            # Load state dict
            _medsam_model.load_state_dict(state_dict)
            
            # Send to device
            _medsam_model = _medsam_model.to(device)
            _medsam_model.eval()
            logger.info(f"MedSAM model loaded successfully on {device}")
        except Exception as e:
            logger.error(f"Error loading MedSAM model: {str(e)}")
            raise
    return _medsam_model

@torch.no_grad()
def medsam_inference(model, img_embed, box_1024, height, width):
    """Run MedSAM inference with the given embedding and box."""
    box_torch = torch.as_tensor(box_1024, dtype=torch.float, device=img_embed.device)
    if len(box_torch.shape) == 2:
        box_torch = box_torch[:, None, :]  # (B, 1, 4)

    sparse_embeddings, dense_embeddings = model.prompt_encoder(
        points=None,
        boxes=box_torch,
        masks=None,
    )
    low_res_logits, _ = model.mask_decoder(
        image_embeddings=img_embed,
        image_pe=model.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=False,
    )

    low_res_pred = torch.sigmoid(low_res_logits)
    low_res_pred = F.interpolate(
        low_res_pred,
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    )
    low_res_pred = low_res_pred.squeeze().cpu().numpy()
    medsam_seg = (low_res_pred > 0.5).astype(np.uint8)
    return medsam_seg

def segment_with_box(image_path, box=None):
    """
    Segment an image using MedSAM with a bounding box.
    
    Args:
        image_path: Path to the image file
        box: Bounding box [x1, y1, x2, y2]. If None, uses the whole image.
        
    Returns:
        tuple: (overlay_img, binary_mask)
            - overlay_img: RGB image with segmentation overlay
            - binary_mask: Binary segmentation mask
    """
    try:
        # Load model
        model = load_model()
        
        # Preprocess image
        img_1024_tensor, original_img, H, W = preprocess_image(image_path, device)
        
        # Handle bounding box
        if box is None:
            # Use a box slightly smaller than the full image
            margin = min(H, W) // 20  # 5% margin
            box = [margin, margin, W - margin, H - margin]
        
        # Scale bounding box to 1024x1024
        box_np = np.array([box])
        box_1024 = box_np / np.array([W, H, W, H]) * 1024
        
        # Get image embedding
        with torch.no_grad():
            image_embedding = model.image_encoder(img_1024_tensor)
        
        # Run inference
        binary_mask = medsam_inference(model, image_embedding, box_1024, H, W)
        
        # Create overlay
        overlay_img = create_overlay(original_img, binary_mask)
        
        return overlay_img, binary_mask
        
    except Exception as e:
        logger.error(f"Error in segment_with_box: {str(e)}")
        raise