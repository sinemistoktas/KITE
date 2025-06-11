import os
import numpy as np
import torch
import cv2
from os import listdir
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add parent directory to path to import MedSAM
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from algorithms.medsam.medsam_test.medsam_test_wrapper import get_original_medsam_instance, validate_box

def calculate_metrics(pred_mask, gt_mask):
    """
    Calculate Dice Score and IoU metrics
    
    Args:
        pred_mask: Predicted binary mask
        gt_mask: Ground truth binary mask
        
    Returns:
        dict: Dictionary containing metrics
    """
    # Ensure masks are binary
    pred_mask = (pred_mask > 0.5).astype(np.uint8)
    gt_mask = (gt_mask > 0.5).astype(np.uint8)
    
    # Calculate intersection and union
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    
    # Calculate metrics
    iou = intersection / (union + 1e-8)
    dice = (2 * intersection) / (pred_mask.sum() + gt_mask.sum() + 1e-8)
    
    return {
        'iou': float(iou),
        'dice': float(dice)
    }

def visualize_results(image, pred_mask, gt_mask, save_path=None):
    """
    Visualize segmentation results with ground truth comparison
    
    Args:
        image: Original image
        pred_mask: Predicted mask
        gt_mask: Ground truth mask
        save_path: Path to save visualization
    """
    plt.figure(figsize=(15, 5))
    
    # Original image
    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(image, cmap='gray')
    
    # Predicted mask
    plt.subplot(1, 3, 2)
    plt.title("MedSAM Prediction")
    plt.imshow(pred_mask, cmap='gray')
    
    # Ground truth
    plt.subplot(1, 3, 3)
    plt.title("Ground Truth")
    plt.imshow(gt_mask, cmap='gray')
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def main():
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, 
                      default=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))), 'data/duke_original'),
                      help='Path to dataset directory')
    parser.add_argument('--list_dir', type=str, 
                      default=os.path.join(script_dir, 'contains_lesion'),
                      help='Directory containing test lists')
    parser.add_argument('--fold', type=int, default=1,
                      help='Fold number for cross-validation')
    parser.add_argument('--save_dir', type=str, 
                      default=os.path.join(script_dir, 'results'),
                      help='Directory to save results')
    args = parser.parse_args()

    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, 'visualizations'), exist_ok=True)
    
    # Initialize MedSAM
    checkpoint_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))), 'models/medsam_vit_b.pth')
    logger.info(f"Loading MedSAM model from: {checkpoint_path}")
    
    # Force a new instance by setting the global instance to None
    import algorithms.medsam.medsam_test.medsam_test_wrapper as medsam_test_wrapper
    medsam_test_wrapper._original_medsam_instance = None
    
    # Create a new instance directly
    medsam = medsam_test_wrapper.TestMedSAMWrapper(checkpoint_path=checkpoint_path, device="cpu")
    
    # Load test image list
    test_list_path = os.path.join(args.list_dir, f'fold{args.fold}', 'test.txt')
    logger.info(f"Loading test list from: {test_list_path}")
    
    with open(test_list_path, 'r') as f:
        test_images = [line.strip() for line in f.readlines()]
    
    # Initialize metrics
    all_metrics = []
    
    # Process each test image
    for img_name in tqdm(test_images, desc="Processing test images"):
        # Construct paths
        img_path = os.path.join(args.dataset, 'image', img_name)
        gt_path = os.path.join(args.dataset, 'lesion', img_name)
        
        # Check if files exist
        if not os.path.exists(img_path):
            logger.error(f"Image file not found: {img_path}")
            continue
        if not os.path.exists(gt_path):
            logger.error(f"Ground truth file not found: {gt_path}")
            continue
            
        logger.info(f"Processing image: {img_path}")
        
        try:
            # Load image and ground truth
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            gt_mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
            
            if img is None:
                logger.error(f"Failed to load image: {img_path}")
                continue
            if gt_mask is None:
                logger.error(f"Failed to load ground truth: {gt_path}")
                continue
            
            # Get image dimensions
            H, W = img.shape
            
            # Create a box that covers the entire image
            box = [0, 0, W, H]
            
            # Run segmentation
            result = medsam.segment_image_with_box(img_path, box)
            
            if not result['isSuccess']:
                logger.error(f"Segmentation failed for {img_name}: {result.get('error', 'Unknown error')}")
                continue
            
            # Get prediction mask
            pred_mask = result['segmentation_mask']
            
            # Calculate metrics
            metrics = calculate_metrics(pred_mask, gt_mask)
            all_metrics.append(metrics)
            
            # Save visualization
            vis_path = os.path.join(args.save_dir, 'visualizations', f'{os.path.splitext(img_name)[0]}_result.png')
            visualize_results(img, pred_mask, gt_mask, vis_path)
            
            # Log test info
            logger.info(f"Test info for {img_name}:")
            logger.info(f"  - Image path: {result['test_info']['image_path']}")
            logger.info(f"  - Box used: {result['test_info']['box_used']}")
            logger.info(f"  - Mask shape: {result['test_info']['mask_shape']}")
            logger.info(f"  - Mask sum: {result['test_info']['mask_sum']}")
            logger.info(f"  - Metrics: {metrics}")
            
        except Exception as e:
            logger.error(f"Error processing {img_name}: {str(e)}")
            continue
    
    # Calculate and save average metrics
    if all_metrics:
        avg_metrics = {
            'iou': np.mean([m['iou'] for m in all_metrics]),
            'dice': np.mean([m['dice'] for m in all_metrics])
        }
        
        logger.info("\nAverage Metrics:")
        logger.info(f"IoU: {avg_metrics['iou']:.4f}")
        logger.info(f"Dice: {avg_metrics['dice']:.4f}")
        
        # Save metrics to file
        metrics_path = os.path.join(args.save_dir, f'metrics_fold{args.fold}.txt')
        with open(metrics_path, 'w') as f:
            f.write(f"IoU: {avg_metrics['iou']:.4f}\n")
            f.write(f"Dice: {avg_metrics['dice']:.4f}\n")
    else:
        logger.error("No metrics were calculated - check if any images were processed successfully")

if __name__ == "__main__":
    main() 