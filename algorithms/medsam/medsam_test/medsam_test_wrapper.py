# algorithms/medsam/medsam_test/medsam_test_wrapper.py
"""
Direct wrapper for original MedSAM functions
Imports only the functions we need, avoiding the argument parsing
"""

import os
import sys
import torch
import numpy as np
from typing import List, Tuple, Union
import logging
from skimage import io, transform
import cv2

# Add parent directory to path to import original wrapper
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from medsam_direct_wrapper import OriginalMedSAMWrapper, validate_box, medsam_inference
from algorithms.medsam.MedSAM_Inference import *  # Absolute import for MedSAM_Inference

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global instance for singleton pattern
_original_medsam_instance = None

# Import only what we need from segment-anything and torch
try:
    from segment_anything import sam_model_registry
    import torch.nn.functional as F
except ImportError as e:
    logger.error(f"Error importing required packages: {e}")
    logger.error("Make sure segment_anything is installed: pip install segment-anything")
    raise

# Import the medsam_inference function directly without executing the script
def import_medsam_inference():
    """
    Import the medsam_inference function from MedSAM_Inference.py
    without executing the main script part
    """
    try:
        # Read the MedSAM_Inference.py file
        current_dir = os.path.dirname(__file__)
        medsam_file = os.path.join(current_dir, 'MedSAM_Inference.py')
        
        if not os.path.exists(medsam_file):
            raise FileNotFoundError(f"MedSAM_Inference.py not found at {medsam_file}")
        
        # Create a namespace to execute the functions
        namespace = {}
        
        # Read and execute only the function definitions
        with open(medsam_file, 'r') as f:
            content = f.read()
        
        # Split content and only execute imports and function definitions
        lines = content.split('\n')
        func_lines = []
        
        in_function = False
        for line in lines:
            # Include imports
            if line.strip().startswith('import ') or line.strip().startswith('from '):
                func_lines.append(line)
            # Include function definitions and their content
            elif line.strip().startswith('def ') or line.strip().startswith('@'):
                in_function = True
                func_lines.append(line)
            elif in_function:
                # Continue collecting function content
                if line.strip() and not line.startswith(' ') and not line.startswith('\t'):
                    # Function ended
                    if not (line.strip().startswith('def ') or line.strip().startswith('@')):
                        in_function = False
                    else:
                        func_lines.append(line)
                else:
                    func_lines.append(line)
        
        # Execute the function definitions
        exec('\n'.join(func_lines), namespace)
        
        return namespace.get('medsam_inference')
        
    except Exception as e:
        logger.error(f"Failed to import medsam_inference: {e}")
        raise

# Try to import the function
try:
    medsam_inference = import_medsam_inference()
    if medsam_inference is None:
        raise ImportError("medsam_inference function not found")
except Exception as e:
    logger.error(f"Could not import medsam_inference: {e}")
    # Provide a fallback implementation
    @torch.no_grad()
    def medsam_inference(medsam_model, img_embed, box_1024, H, W):
        """
        Fallback implementation copied from original MedSAM_Inference.py
        """
        box_torch = torch.as_tensor(box_1024, dtype=torch.float, device=img_embed.device)
        if len(box_torch.shape) == 2:
            box_torch = box_torch[:, None, :]  # (B, 1, 4)

        sparse_embeddings, dense_embeddings = medsam_model.prompt_encoder(
            points=None,
            boxes=box_torch,
            masks=None,
        )
        low_res_logits, _ = medsam_model.mask_decoder(
            image_embeddings=img_embed,  # (B, 256, 64, 64)
            image_pe=medsam_model.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
            multimask_output=False,
        )

        low_res_pred = torch.sigmoid(low_res_logits)  # (1, 1, 256, 256)

        low_res_pred = F.interpolate(
            low_res_pred,
            size=(H, W),
            mode="bilinear",
            align_corners=False,
        )  # (1, 1, gt.shape)
        low_res_pred = low_res_pred.squeeze().cpu().numpy()  # (256, 256)
        medsam_seg = (low_res_pred > 0.5).astype(np.uint8)
        return medsam_seg

class TestMedSAMWrapper(OriginalMedSAMWrapper):
    """
    Test-specific wrapper for MedSAM that adds test functionality
    """
    def __init__(self, checkpoint_path: str = "models/medsam_vit_b.pth", device: str = "auto"):
        super().__init__(checkpoint_path, device)
        
    def segment_image_with_box(self, img_path: str, box: List[int]) -> dict:
        """
        Override the original method to add test-specific functionality
        """
        result = super().segment_image_with_box(img_path, box)
        
        # Add test-specific information
        if result['isSuccess']:
            result['test_info'] = {
                'image_path': img_path,
                'box_used': box,
                'mask_shape': result['segmentation_mask'].shape if result['segmentation_mask'] is not None else None,
                'mask_sum': int(np.sum(result['segmentation_mask'])) if result['segmentation_mask'] is not None else 0
            }
        
        return result

def get_original_medsam_instance(checkpoint_path: str = "models/medsam_vit_b.pth", device: str = "auto") -> TestMedSAMWrapper:
    """Get singleton instance of test MedSAM wrapper"""
    global _original_medsam_instance
    if _original_medsam_instance is None:
        _original_medsam_instance = TestMedSAMWrapper(checkpoint_path, device)
    return _original_medsam_instance

# Drop-in replacement functions for your current interface
def segment_with_box(img_path: str, box: Union[List[int], None] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Drop-in replacement that uses original MedSAM
    
    Returns:
        Tuple of (overlay_image, binary_mask)
    """
    medsam = get_original_medsam_instance()
    
    if box is None:
        # Use whole image
        img = io.imread(img_path)
        H, W = img.shape[:2]
        box = [0, 0, W, H]
    
    result = medsam.segment_image_with_box(img_path, box)
    
    if result['isSuccess']:
        return result['overlay_image'], result['segmentation_mask']
    else:
        raise Exception(f"Segmentation failed: {result['error']}")

def validate_box(box: List[int], H: int, W: int) -> List[int]:
    """Validate bounding box coordinates"""
    x1, y1, x2, y2 = box
    x1 = max(0, min(x1, W-1))
    y1 = max(0, min(y1, H-1))
    x2 = max(x1+1, min(x2, W))
    y2 = max(y1+1, min(y2, H))
    return [x1, y1, x2, y2]

def create_overlay(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Create overlay visualization"""
    wrapper = get_original_medsam_instance()
    return wrapper._create_overlay_like_original(img, mask, [0, 0, img.shape[1], img.shape[0]])

def save_results(overlay_img, mask, results_dir, base_filename):
    """Save results to files"""
    overlay_path = os.path.join(results_dir, f"{base_filename}_overlay.png")
    mask_path = os.path.join(results_dir, f"{base_filename}_mask.png")
    
    io.imsave(overlay_path, overlay_img, check_contrast=False)
    io.imsave(mask_path, mask * 255, check_contrast=False)
    
    return overlay_path, mask_path