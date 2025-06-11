# algorithms/medsam/medsam_direct_wrapper.py
"""
Direct wrapper for original MedSAM functions
Imports only the functions we need, avoiding the argument parsing
"""

import os
import sys
import numpy as np
from skimage import io, transform
import torch
import torch.nn.functional as F
from typing import Tuple, List, Union, Optional
import logging

logger = logging.getLogger(__name__)

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

class OriginalMedSAMWrapper:
    """
    Wrapper that uses the original MedSAM inference logic
    """
    
    def __init__(self, checkpoint_path: str = "models/medsam_vit_b.pth", device: str = "auto"):
        """
        Initialize using the same logic as original MedSAM_Inference.py
        """
        self.checkpoint_path = self._resolve_checkpoint_path(checkpoint_path)
        self.device = self._setup_device(device)
        self.medsam_model = None
        self._load_model()
    
    def _resolve_checkpoint_path(self, checkpoint_path: str) -> str:
        """Resolve the checkpoint path to an absolute path"""
        if os.path.isabs(checkpoint_path):
            return checkpoint_path
        
        # Try relative to current working directory
        if os.path.exists(checkpoint_path):
            return os.path.abspath(checkpoint_path)
        
        # Try relative to project root (go up from algorithms/medsam/)
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        abs_path = os.path.join(project_root, checkpoint_path)
        if os.path.exists(abs_path):
            return abs_path
        
        # Try common locations
        possible_paths = [
            os.path.join(project_root, "models", "medsam_vit_b.pth"),
            os.path.join(project_root, "work_dir", "MedSAM", "medsam_vit_b.pth"),
            os.path.join(os.getcwd(), "models", "medsam_vit_b.pth"),
            "models/medsam_vit_b.pth",
            "./models/medsam_vit_b.pth"
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return os.path.abspath(path)
        
        # Log what we tried
        logger.warning(f"Checkpoint not found. Tried paths:")
        logger.warning(f"  - {checkpoint_path}")
        logger.warning(f"  - {abs_path}")
        for path in possible_paths:
            logger.warning(f"  - {path}")
        
        # Return original path (will fail later with clear error)
        return checkpoint_path
    
    def _setup_device(self, device: str) -> str:
        """Setup device for inference."""
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda:0"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        return device
    
    def _load_model(self):
        """Load model with proper device handling"""
        try:
            if not os.path.exists(self.checkpoint_path):
                raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")
            
            logger.info(f"Loading MedSAM model from {self.checkpoint_path}")
            logger.info(f"Target device: {self.device}")
            
            # Create model without loading checkpoint first
            self.medsam_model = sam_model_registry["vit_b"](checkpoint=None)
            
            # Load checkpoint with proper device mapping
            if self.device == "cpu" or not torch.cuda.is_available():
                # Load with CPU mapping
                checkpoint = torch.load(self.checkpoint_path, map_location='cpu')
            else:
                # Load normally for CUDA/MPS
                checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
            
            # Load the state dict
            self.medsam_model.load_state_dict(checkpoint)
            
            # Move to target device
            self.medsam_model = self.medsam_model.to(self.device)
            self.medsam_model.eval()
            
            logger.info(f"✅ MedSAM model loaded successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"❌ Failed to load MedSAM model: {e}")
            
            # Try alternative loading method
            try:
                logger.info("Trying alternative loading method...")
                
                # Force CPU loading
                self.device = "cpu"
                checkpoint = torch.load(self.checkpoint_path, map_location='cpu')
                
                # Create model and load state
                self.medsam_model = sam_model_registry["vit_b"](checkpoint=None)
                self.medsam_model.load_state_dict(checkpoint)
                self.medsam_model = self.medsam_model.to(self.device)
                self.medsam_model.eval()
                
                logger.info(f"✅ MedSAM model loaded successfully on {self.device} (fallback)")
                
            except Exception as e2:
                logger.error(f"❌ Alternative loading also failed: {e2}")
                raise Exception(f"Could not load MedSAM model: {e2}")
    
    def segment_image_with_box(self, img_path: str, box: List[int]) -> dict:
        """
        Use the original MedSAM inference pipeline exactly as in MedSAM_Inference.py
        
        Args:
            img_path: Path to input image
            box: Bounding box [x1, y1, x2, y2]
            
        Returns:
            Dictionary with results and isSuccess flag
        """
        try:
            # This follows the exact same steps as the original MedSAM_Inference.py
            
            # Load and preprocess image (from original)
            img_np = io.imread(img_path)
            if len(img_np.shape) == 2:
                img_3c = np.repeat(img_np[:, :, None], 3, axis=-1)
            else:
                img_3c = img_np
            H, W, _ = img_3c.shape
            
            # Image preprocessing (exact copy from original)
            img_1024 = transform.resize(
                img_3c, (1024, 1024), order=3, preserve_range=True, anti_aliasing=True
            ).astype(np.uint8)
            img_1024 = (img_1024 - img_1024.min()) / np.clip(
                img_1024.max() - img_1024.min(), a_min=1e-8, a_max=None
            )  # normalize to [0, 1], (H, W, 3)
            
            # Convert the shape to (3, H, W) (exact copy from original)
            img_1024_tensor = (
                torch.tensor(img_1024).float().permute(2, 0, 1).unsqueeze(0).to(self.device)
            )

            # Process box (exact copy from original)
            box_np = np.array([[int(x) for x in box]]) 
            # transfer box_np to 1024x1024 scale (exact copy from original)
            box_1024 = box_np / np.array([W, H, W, H]) * 1024
            
            # Get image embedding (exact copy from original)
            with torch.no_grad():
                image_embedding = self.medsam_model.image_encoder(img_1024_tensor)  # (1, 256, 64, 64)

            # Run inference using the original function
            medsam_seg = medsam_inference(self.medsam_model, image_embedding, box_1024, H, W)
            
            # Create overlay visualization
            overlay_img = self._create_overlay_like_original(img_3c, medsam_seg, box_np[0])
            
            return {
                'isSuccess': True,
                'segmentation_mask': medsam_seg,
                'overlay_image': overlay_img,
                'original_image': img_3c,
                'box_used': box,
                'image_shape': (H, W),
                'mask_pixels': int(np.sum(medsam_seg)),
                'error': None
            }
            
        except Exception as e:
            logger.error(f"Segmentation failed: {e}")
            return {
                'isSuccess': False,
                'segmentation_mask': None,
                'overlay_image': None,
                'original_image': None,
                'box_used': box,
                'image_shape': None,
                'mask_pixels': 0,
                'error': str(e)
            }
    
    def _create_overlay_like_original(self, img_3c: np.ndarray, mask: np.ndarray, box: List[int]) -> np.ndarray:
        """
        Create overlay similar to the original MedSAM visualization
        """
        # Ensure image is in correct format
        if img_3c.max() <= 1.0:
            # Image is normalized [0,1], convert to [0,255]
            img_normalized = (img_3c * 255).astype(np.uint8)
        else:
            img_normalized = img_3c.astype(np.uint8)
        
        # Create overlay similar to original MedSAM
        overlay = img_normalized.copy().astype(np.float32)
        
        # Apply mask overlay (using yellow color like in original)
        color = np.array([251, 252, 30])  # Yellow in 0-255 range
        alpha = 0.4  # Reduce alpha to make it less intrusive
        
        # Only apply color where mask is positive
        mask_indices = mask > 0
        if np.any(mask_indices):
            # Blend: overlay = image * (1-alpha) + color * alpha
            overlay[mask_indices] = (
                overlay[mask_indices] * (1 - alpha) + 
                color * alpha
            )
        
        return overlay.astype(np.uint8)

    def batch_segment(self, img_path: str, boxes: List[List[int]]) -> dict:
        """
        Process multiple boxes using the original inference for each
        """
        try:
            # Load image once
            img_np = io.imread(img_path)
            if len(img_np.shape) == 2:
                img_3c = np.repeat(img_np[:, :, None], 3, axis=-1)
            else:
                img_3c = img_np
            H, W, _ = img_3c.shape
            
            # Preprocess once
            img_1024 = transform.resize(
                img_3c, (1024, 1024), order=3, preserve_range=True, anti_aliasing=True
            ).astype(np.uint8)
            img_1024 = (img_1024 - img_1024.min()) / np.clip(
                img_1024.max() - img_1024.min(), a_min=1e-8, a_max=None
            )
            img_1024_tensor = (
                torch.tensor(img_1024).float().permute(2, 0, 1).unsqueeze(0).to(self.device)
            )
            
            # Get embedding once
            with torch.no_grad():
                image_embedding = self.medsam_model.image_encoder(img_1024_tensor)
            
            # Process each box
            combined_mask = None
            individual_results = []
            
            for i, box in enumerate(boxes):
                try:
                    # Process box exactly like original
                    box_np = np.array([[int(x) for x in box]]) 
                    box_1024 = box_np / np.array([W, H, W, H]) * 1024
                    
                    # Use original inference function
                    mask = medsam_inference(self.medsam_model, image_embedding, box_1024, H, W)
                    
                    # Combine masks
                    if combined_mask is None:
                        combined_mask = mask.copy()
                    else:
                        combined_mask = np.logical_or(combined_mask, mask).astype(np.uint8)
                    
                    individual_results.append({
                        'box': box,
                        'mask_pixels': int(np.sum(mask)),
                        'success': True
                    })
                    
                except Exception as e:
                    logger.error(f"Failed to process box {i}: {e}")
                    individual_results.append({
                        'box': box,
                        'mask_pixels': 0,
                        'success': False,
                        'error': str(e)
                    })
            
            # Create final overlay
            overlay_img = None
            if combined_mask is not None:
                overlay_img = self._create_overlay_like_original(img_3c, combined_mask, boxes[0])
            
            return {
                'isSuccess': True,
                'combined_mask': combined_mask,
                'overlay_image': overlay_img,
                'original_image': img_3c,
                'boxes_processed': len(boxes),
                'total_mask_pixels': int(np.sum(combined_mask)) if combined_mask is not None else 0,
                'individual_results': individual_results,
                'error': None
            }
            
        except Exception as e:
            logger.error(f"Batch segmentation failed: {e}")
            return {
                'isSuccess': False,
                'combined_mask': None,
                'overlay_image': None,
                'original_image': None,
                'boxes_processed': 0,
                'total_mask_pixels': 0,
                'individual_results': None,
                'error': str(e)
            }


# Global instance
_original_medsam_instance = None

def get_original_medsam_instance(checkpoint_path: str = "models/medsam_vit_b.pth", device: str = "auto") -> OriginalMedSAMWrapper:
    """Get singleton instance of original MedSAM wrapper"""
    global _original_medsam_instance
    if _original_medsam_instance is None:
        _original_medsam_instance = OriginalMedSAMWrapper(checkpoint_path, device)
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