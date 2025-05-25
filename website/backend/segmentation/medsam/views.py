# website/backend/segmentation/medsam/views.py
"""
MedSAM API Views - Safe version with fallbacks
Django views for handling MedSAM segmentation requests.
"""

import os
import json
import uuid
import numpy as np
from django.conf import settings
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from skimage import io
import base64
from io import BytesIO
from PIL import Image
import logging

logger = logging.getLogger(__name__)

# Try to import MedSAM functions with fallbacks
MEDSAM_AVAILABLE = False
try:
    from algorithms.medsam import (
        get_original_medsam_instance, 
        validate_box, 
        save_results,
        segment_with_box
    )
    MEDSAM_AVAILABLE = True
    logger.info("✅ MedSAM imported successfully")
except ImportError as e:
    logger.warning(f"⚠️  MedSAM import failed: {e}")
    
    # Provide fallback functions
    def validate_box(box, H, W):
        """Fallback box validation"""
        if box is None:
            return None
        x1, y1, x2, y2 = box
        x1 = max(0, min(x1, W-1))
        y1 = max(0, min(y1, H-1))
        x2 = max(x1+1, min(x2, W))
        y2 = max(y1+1, min(y2, H))
        return [x1, y1, x2, y2]
    
    def save_results(overlay_img, mask, results_dir, base_filename):
        """Fallback save function"""
        overlay_path = os.path.join(results_dir, f"{base_filename}_overlay.png")
        mask_path = os.path.join(results_dir, f"{base_filename}_mask.png")
        
        io.imsave(overlay_path, overlay_img, check_contrast=False)
        io.imsave(mask_path, mask * 255, check_contrast=False)
        
        return overlay_path, mask_path
    
    def segment_with_box(img_path, box):
        """Fallback segmentation - returns dummy results"""
        img = io.imread(img_path)
        if len(img.shape) == 2:
            img = np.repeat(img[:, :, None], 3, axis=-1)
        
        # Create dummy mask
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        if box:
            x1, y1, x2, y2 = box
            mask[y1:y2, x1:x2] = 1
        
        # Create simple overlay
        overlay = img.copy()
        overlay[mask > 0] = [255, 255, 0]  # Yellow
        
        return overlay, mask
    
    def get_original_medsam_instance():
        """Fallback instance"""
        raise ImportError("MedSAM not available")

@csrf_exempt
@require_http_methods(["POST"])
def segment_image(request):
    """
    Main API endpoint for MedSAM segmentation.
    Supports both single and multiple bounding boxes by default.
    """
    try:
        # Parse request data
        data = json.loads(request.body)
        image_path = data.get('image_path')
        
        # Handle both single box and multiple boxes
        single_box = data.get('box')
        multiple_boxes = data.get('boxes', [])
        
        # Determine which format to use
        if single_box and not multiple_boxes:
            boxes_to_process = [single_box]
            is_batch = False
        elif multiple_boxes:
            boxes_to_process = multiple_boxes
            is_batch = True
        else:
            # No boxes provided, use full image
            boxes_to_process = [None]
            is_batch = False
        
        # Validate inputs
        if not image_path:
            return JsonResponse({
                'success': False,
                'isSuccess': False,
                'error': 'Missing image path'
            }, status=400)
        
        # Get absolute image path
        full_image_path = os.path.join(settings.MEDIA_ROOT, image_path)
        
        if not os.path.exists(full_image_path):
            return JsonResponse({
                'success': False,
                'isSuccess': False,
                'error': f'Image not found: {image_path}'
            }, status=404)
        
        # Load image for validation
        original_img = io.imread(full_image_path)
        H, W = original_img.shape[:2]
        
        # Validate all bounding boxes
        validated_boxes = []
        for box in boxes_to_process:
            if box is not None:
                validated_box = validate_box(box, H, W)
                validated_boxes.append(validated_box)
            else:
                validated_boxes.append(None)
        
        logger.info(f"Processing MedSAM segmentation for: {image_path} with {len(validated_boxes)} box(es)")
        
        # Process segmentation based on availability
        if MEDSAM_AVAILABLE:
            # Use original MedSAM
            try:
                if is_batch:
                    # Process multiple boxes
                    medsam = get_original_medsam_instance()
                    result = medsam.batch_segment(
                        full_image_path, 
                        validated_boxes, 
                        combine_masks=True, 
                        return_individual=True
                    )
                    
                    if not result['isSuccess']:
                        return JsonResponse({
                            'success': False,
                            'isSuccess': False,
                            'error': result['error']
                        }, status=500)
                    
                    combined_mask = result['combined_mask']
                    final_overlay = result['overlay_image']
                    individual_results = result['individual_results']
                    
                else:
                    # Single box processing
                    medsam = get_original_medsam_instance()
                    result = medsam.segment_image_with_box(
                        full_image_path, 
                        validated_boxes[0] if validated_boxes[0] is not None else [0, 0, W, H]
                    )
                    
                    if not result['isSuccess']:
                        return JsonResponse({
                            'success': False,
                            'isSuccess': False,
                            'error': result['error']
                        }, status=500)
                    
                    combined_mask = result['segmentation_mask']
                    final_overlay = result['overlay_image']
                    individual_results = None
                    
            except Exception as e:
                logger.error(f"Original MedSAM failed: {e}")
                return JsonResponse({
                    'success': False,
                    'isSuccess': False,
                    'error': f'MedSAM processing failed: {str(e)}'
                }, status=500)
        else:
            # Use fallback implementation
            logger.warning("Using fallback segmentation")
            try:
                individual_results = []
                combined_mask = None
                
                for i, box in enumerate(validated_boxes):
                    overlay_img, binary_mask = segment_with_box(full_image_path, box)
                    
                    # Combine masks for multiple boxes
                    if combined_mask is None:
                        combined_mask = binary_mask.copy()
                        final_overlay = overlay_img.copy()
                    else:
                        combined_mask = np.logical_or(combined_mask, binary_mask).astype(np.uint8)
                        final_overlay = overlay_img  # Simple approach for fallback
                    
                    # Store individual results for batch processing
                    if is_batch:
                        individual_results.append({
                            'box': box,
                            'mask_pixels': int(np.sum(binary_mask)),
                            'box_area': (box[2] - box[0]) * (box[3] - box[1]) if box else H * W,
                            'success': True
                        })
                        
            except Exception as e:
                logger.error(f"Fallback segmentation failed: {e}")
                return JsonResponse({
                    'success': False,
                    'isSuccess': False,
                    'error': f'Segmentation failed: {str(e)}'
                }, status=500)
        
        # Create results directory
        results_dir = os.path.join(settings.MEDIA_ROOT, 'medsam_results')
        os.makedirs(results_dir, exist_ok=True)
        
        # Generate unique filenames
        unique_id = uuid.uuid4().hex[:8]
        prefix = "batch" if is_batch else "single"
        implementation = "original" if MEDSAM_AVAILABLE else "fallback"
        base_filename = f"{prefix}_{implementation}_{unique_id}_{os.path.splitext(os.path.basename(image_path))[0]}"
        
        # Save results
        overlay_path, mask_path = save_results(
            final_overlay, combined_mask, results_dir, base_filename
        )
        
        # Convert overlay to base64 for immediate display
        buffered = BytesIO()
        Image.fromarray(final_overlay).save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        # Generate relative paths for URLs
        overlay_rel_path = os.path.relpath(overlay_path, settings.MEDIA_ROOT)
        mask_rel_path = os.path.relpath(mask_path, settings.MEDIA_ROOT)
        
        logger.info(f"MedSAM segmentation completed successfully for: {image_path}")
        
        # Prepare response
        response_data = {
            'success': True,
            'isSuccess': True,
            'overlay_path': os.path.join(settings.MEDIA_URL, overlay_rel_path),
            'mask_path': os.path.join(settings.MEDIA_URL, mask_rel_path),
            'overlay_data': f"data:image/png;base64,{img_str}",
            'boxes_processed': len(validated_boxes),
            'total_mask_pixels': int(np.sum(combined_mask)),
            'is_batch': is_batch,
            'error': None,
            'using_original_medsam': MEDSAM_AVAILABLE,
            'implementation': "original" if MEDSAM_AVAILABLE else "fallback"
        }
        
        # Add individual results for batch processing
        if is_batch and individual_results:
            response_data['individual_results'] = individual_results
        
        # Add box info for single box
        if not is_batch and validated_boxes[0] is not None:
            response_data['box_used'] = validated_boxes[0]
        
        return JsonResponse(response_data)
        
    except json.JSONDecodeError:
        return JsonResponse({
            'success': False,
            'isSuccess': False,
            'error': 'Invalid JSON data'
        }, status=400)
        
    except Exception as e:
        logger.error(f"Error in MedSAM segmentation: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return JsonResponse({
            'success': False,
            'isSuccess': False,
            'error': f'Segmentation failed: {str(e)}'
        }, status=500)

def health_check(request):
    """Simple health check endpoint."""
    try:
        if MEDSAM_AVAILABLE:
            # Test if MedSAM can be loaded
            try:
                medsam = get_original_medsam_instance()
                model_status = "original_medsam_loaded"
            except Exception as e:
                model_status = f"original_medsam_error: {str(e)}"
        else:
            model_status = "fallback_implementation"
    except Exception as e:
        model_status = f"error: {str(e)}"
    
    return JsonResponse({
        'status': 'healthy',
        'service': 'MedSAM Backend',
        'version': '1.0.0',
        'features': ['single_box', 'batch_boxes', 'fallback_support'],
        'model_status': model_status,
        'using_original_medsam': MEDSAM_AVAILABLE,
        'medsam_available': MEDSAM_AVAILABLE
    })