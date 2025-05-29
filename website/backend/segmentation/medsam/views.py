# website/backend/segmentation/medsam/views.py
"""
MedSAM API Views - Fixed version with proper state management
Django views for handling MedSAM segmentation requests with cumulative results.
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

# Global storage for session-based cumulative results
session_results = {}

def get_session_id(request):
    """Get or create session ID for tracking cumulative results"""
    session_id = request.session.get('medsam_session_id')
    if not session_id:
        session_id = uuid.uuid4().hex
        request.session['medsam_session_id'] = session_id
    return session_id

def create_combined_overlay(original_img, combined_mask, individual_masks=None, colors=None):
    """
    Create overlay with different colors for different regions
    """
    if len(original_img.shape) == 2:
        img_3c = np.repeat(original_img[:, :, None], 3, axis=-1)
    else:
        img_3c = original_img.copy()
    
    # Ensure image is in [0, 255] range
    if img_3c.max() <= 1.0:
        img_3c = (img_3c * 255).astype(np.uint8)
    else:
        img_3c = img_3c.astype(np.uint8)
    
    overlay = img_3c.astype(np.float32)
    alpha = 0.4
    
    if individual_masks and colors:
        # Apply different colors for each mask
        for mask, color in zip(individual_masks, colors):
            mask_indices = mask > 0
            if np.any(mask_indices):
                overlay[mask_indices] = (
                    overlay[mask_indices] * (1 - alpha) + 
                    np.array(color) * alpha
                )
    else:
        # Single color for combined mask
        color = np.array([251, 252, 30])  # Yellow
        mask_indices = combined_mask > 0
        if np.any(mask_indices):
            overlay[mask_indices] = (
                overlay[mask_indices] * (1 - alpha) + 
                color * alpha
            )
    
    return overlay.astype(np.uint8)

@csrf_exempt
@require_http_methods(["POST"])
def segment_image(request):
    """
    Main API endpoint for MedSAM segmentation with cumulative support.
    """
    try:
        # Parse request data
        data = json.loads(request.body)
        image_path = data.get('image_path')
        
        # Get session ID for tracking cumulative results
        session_id = get_session_id(request)
        
        # Handle different modes
        mode = data.get('mode', 'add')  # 'add', 'replace', 'clear'
        single_box = data.get('box')
        multiple_boxes = data.get('boxes', [])
        
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
        
        # Initialize session results if not exists or if new image
        if (session_id not in session_results or 
            session_results[session_id].get('image_path') != image_path or
            mode == 'clear'):
            session_results[session_id] = {
                'image_path': image_path,
                'boxes': [],
                'masks': [],
                'combined_mask': None,
                'colors': []
            }
            logger.info(f"Initialized new session results for {session_id}")
        
        session_data = session_results[session_id]
        
        # Handle clear mode
        if mode == 'clear':
            return JsonResponse({
                'success': True,
                'isSuccess': True,
                'message': 'Session cleared',
                'boxes_count': 0,
                'total_mask_pixels': 0
            })
        
        # Determine boxes to process
        if single_box and not multiple_boxes:
            new_boxes = [single_box]
        elif multiple_boxes:
            new_boxes = multiple_boxes
        else:
            return JsonResponse({
                'success': False,
                'isSuccess': False,
                'error': 'No boxes provided'
            }, status=400)
        
        # Validate new boxes
        validated_new_boxes = []
        for box in new_boxes:
            if box is not None:
                validated_box = validate_box(box, H, W)
                validated_new_boxes.append(validated_box)
        
        if not validated_new_boxes:
            return JsonResponse({
                'success': False,
                'isSuccess': False,
                'error': 'No valid boxes provided'
            }, status=400)
        
        logger.info(f"Processing {len(validated_new_boxes)} new box(es) in {mode} mode")
        
        # Handle replace mode
        if mode == 'replace':
            session_data['boxes'] = []
            session_data['masks'] = []
            session_data['combined_mask'] = None
            session_data['colors'] = []
        
        # Generate colors for new boxes
        color_palette = [
            [251, 252, 30],   # Yellow
            [255, 0, 0],      # Red
            [0, 255, 0],      # Green
            [0, 0, 255],      # Blue
            [255, 165, 0],    # Orange
            [128, 0, 128],    # Purple
            [255, 192, 203],  # Pink
            [0, 255, 255],    # Cyan
        ]
        
        # Process new boxes
        new_masks = []
        new_colors = []
        
        for i, box in enumerate(validated_new_boxes):
            try:
                if MEDSAM_AVAILABLE:
                    # Use original MedSAM
                    medsam = get_original_medsam_instance()
                    result = medsam.segment_image_with_box(full_image_path, box)
                    
                    if not result['isSuccess']:
                        logger.error(f"MedSAM failed for box {i}: {result['error']}")
                        continue
                    
                    binary_mask = result['segmentation_mask']
                else:
                    # Use fallback
                    _, binary_mask = segment_with_box(full_image_path, box)
                
                new_masks.append(binary_mask)
                
                # Assign color
                color_index = (len(session_data['boxes']) + i) % len(color_palette)
                new_colors.append(color_palette[color_index])
                
                logger.info(f"Successfully processed box {i}: {np.sum(binary_mask)} pixels")
                
            except Exception as e:
                logger.error(f"Failed to process box {i}: {e}")
                continue
        
        if not new_masks:
            return JsonResponse({
                'success': False,
                'isSuccess': False,
                'error': 'Failed to process any boxes'
            }, status=500)
        
        # Add new results to session
        session_data['boxes'].extend(validated_new_boxes)
        session_data['masks'].extend(new_masks)
        session_data['colors'].extend(new_colors)
        
        # Create combined mask
        if session_data['masks']:
            combined_mask = np.zeros_like(session_data['masks'][0], dtype=np.uint8)
            for mask in session_data['masks']:
                combined_mask = np.logical_or(combined_mask, mask).astype(np.uint8)
            session_data['combined_mask'] = combined_mask
        else:
            combined_mask = np.zeros((H, W), dtype=np.uint8)
            session_data['combined_mask'] = combined_mask
        
        # Create combined overlay with different colors
        final_overlay = create_combined_overlay(
            original_img, 
            session_data['combined_mask'],
            session_data['masks'],
            session_data['colors']
        )
        
        # Save results
        results_dir = os.path.join(settings.MEDIA_ROOT, 'medsam_results')
        os.makedirs(results_dir, exist_ok=True)
        
        # Generate filename
        unique_id = uuid.uuid4().hex[:8]
        implementation = "original" if MEDSAM_AVAILABLE else "fallback"
        base_filename = f"cumulative_{implementation}_{unique_id}_{os.path.splitext(os.path.basename(image_path))[0]}"
        
        # Save combined results
        overlay_path, mask_path = save_results(
            final_overlay, session_data['combined_mask'], results_dir, base_filename
        )
        
        # Convert overlay to base64
        buffered = BytesIO()
        Image.fromarray(final_overlay).save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        # Generate relative paths
        overlay_rel_path = os.path.relpath(overlay_path, settings.MEDIA_ROOT)
        mask_rel_path = os.path.relpath(mask_path, settings.MEDIA_ROOT)
        
        # Prepare individual results
        individual_results = []
        for i, (box, mask, color) in enumerate(zip(session_data['boxes'], session_data['masks'], session_data['colors'])):
            individual_results.append({
                'box_id': i,
                'box': box,
                'mask_pixels': int(np.sum(mask)),
                'color': color,
                'box_area': (box[2] - box[0]) * (box[3] - box[1]),
                'coverage_ratio': float(np.sum(mask)) / ((box[2] - box[0]) * (box[3] - box[1]))
            })
        
        logger.info(f"Cumulative segmentation completed: {len(session_data['boxes'])} total boxes")
        
        return JsonResponse({
            'success': True,
            'isSuccess': True,
            'overlay_path': os.path.join(settings.MEDIA_URL, overlay_rel_path),
            'mask_path': os.path.join(settings.MEDIA_URL, mask_rel_path),
            'overlay_data': f"data:image/png;base64,{img_str}",
            'boxes_processed': len(validated_new_boxes),
            'total_boxes': len(session_data['boxes']),
            'total_mask_pixels': int(np.sum(session_data['combined_mask'])),
            'new_mask_pixels': sum(int(np.sum(mask)) for mask in new_masks),
            'individual_results': individual_results,
            'mode': mode,
            'session_id': session_id,
            'using_original_medsam': MEDSAM_AVAILABLE,
            'implementation': "original" if MEDSAM_AVAILABLE else "fallback",
            'error': None
        })
        
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

@csrf_exempt
@require_http_methods(["POST"])
def clear_session(request):
    """Clear session results for a fresh start"""
    try:
        session_id = get_session_id(request)
        if session_id in session_results:
            del session_results[session_id]
        
        return JsonResponse({
            'success': True,
            'message': 'Session cleared successfully'
        })
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=500)

@csrf_exempt
@require_http_methods(["GET"])
def get_session_status(request):
    """Get current session status"""
    try:
        session_id = get_session_id(request)
        
        if session_id not in session_results:
            return JsonResponse({
                'success': True,
                'session_id': session_id,
                'boxes_count': 0,
                'total_mask_pixels': 0,
                'image_path': None
            })
        
        session_data = session_results[session_id]
        
        return JsonResponse({
            'success': True,
            'session_id': session_id,
            'boxes_count': len(session_data['boxes']),
            'total_mask_pixels': int(np.sum(session_data['combined_mask'])) if session_data['combined_mask'] is not None else 0,
            'image_path': session_data['image_path'],
            'boxes': session_data['boxes'],
            'colors': session_data['colors']
        })
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': str(e)
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
        'version': '2.0.0',
        'features': ['cumulative_segmentation', 'session_management', 'multi_color_overlay', 'fallback_support'],
        'model_status': model_status,
        'using_original_medsam': MEDSAM_AVAILABLE,
        'medsam_available': MEDSAM_AVAILABLE,
        'active_sessions': len(session_results)
    })