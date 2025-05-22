# website/backend/medsam/views.py
"""
MedSAM API Views
Django views for handling MedSAM segmentation requests.
Default: Batch segmentation support
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

# Import MedSAM algorithm
from algorithms.medsam.inference import segment_with_box
from algorithms.medsam.utils import validate_box, save_results, create_overlay

logger = logging.getLogger(__name__)

@csrf_exempt
@require_http_methods(["POST"])
def segment_image(request):
    """
    Main API endpoint for MedSAM segmentation.
    Supports both single and multiple bounding boxes by default.
    
    Expected POST data:
    {
        "image_path": "relative/path/to/image.jpg",
        "box": [x1, y1, x2, y2]  // Single box
        // OR
        "boxes": [[x1, y1, x2, y2], [x1, y1, x2, y2], ...]  // Multiple boxes
    }
    
    Returns:
    {
        "success": true,
        "overlay_path": "/media/medsam_results/overlay_xxx.png",
        "mask_path": "/media/medsam_results/mask_xxx.png",
        "overlay_data": "data:image/png;base64,...",
        "boxes_processed": 2,
        "individual_results": [...] // Only for multiple boxes
    }
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
                'error': 'Missing image path'
            }, status=400)
        
        # Get absolute image path
        full_image_path = os.path.join(settings.MEDIA_ROOT, image_path)
        
        if not os.path.exists(full_image_path):
            return JsonResponse({
                'success': False, 
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
        
        # Process segmentation for each box
        individual_results = []
        combined_mask = None
        
        for i, box in enumerate(validated_boxes):
            logger.info(f"Processing box {i+1}/{len(validated_boxes)}: {box}")
            
            # Run segmentation for this box
            overlay_img, binary_mask = segment_with_box(full_image_path, box)
            
            # Combine masks for multiple boxes
            if combined_mask is None:
                combined_mask = binary_mask.copy()
                final_overlay = overlay_img.copy()
            else:
                # Combine masks (union of all segmentations)
                combined_mask = np.logical_or(combined_mask, binary_mask).astype(np.uint8)
                # Update overlay with combined mask
                final_overlay = create_overlay(original_img, combined_mask)
            
            # Store individual results for batch processing
            if is_batch:
                individual_results.append({
                    'box': box,
                    'mask_pixels': int(np.sum(binary_mask)),
                    'box_area': (box[2] - box[0]) * (box[3] - box[1]) if box else H * W
                })
        
        # Create results directory
        results_dir = os.path.join(settings.MEDIA_ROOT, 'medsam_results')
        os.makedirs(results_dir, exist_ok=True)
        
        # Generate unique filenames
        unique_id = uuid.uuid4().hex[:8]
        prefix = "batch" if is_batch else "single"
        base_filename = f"{prefix}_{unique_id}_{os.path.splitext(os.path.basename(image_path))[0]}"
        
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
            'overlay_path': os.path.join(settings.MEDIA_URL, overlay_rel_path),
            'mask_path': os.path.join(settings.MEDIA_URL, mask_rel_path),
            'overlay_data': f"data:image/png;base64,{img_str}",
            'boxes_processed': len(validated_boxes),
            'total_mask_pixels': int(np.sum(combined_mask)),
            'is_batch': is_batch
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
            'error': 'Invalid JSON data'
        }, status=400)
        
    except Exception as e:
        logger.error(f"Error in MedSAM segmentation: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return JsonResponse({
            'success': False,
            'error': f'Segmentation failed: {str(e)}'
        }, status=500)

def health_check(request):
    """Simple health check endpoint."""
    return JsonResponse({
        'status': 'healthy',
        'service': 'MedSAM Backend',
        'version': '1.0.0',
        'features': ['single_box', 'batch_boxes', 'auto_detection']
    })