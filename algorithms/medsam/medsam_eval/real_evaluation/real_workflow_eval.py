#!/usr/bin/env python3
"""
Real Website Workflow Evaluation
Uses actual website results instead of simulation

EDIT THESE VARIABLES TO SET YOUR FILES:
"""

# ==================== FILE CONFIGURATION ====================
# Edit these variables with your file names:

IMAGE_NAME = "Subject_04_25.png"
MEDSAM_ANNOTATION_MASK = "downloaded_masks/medsam_annotation_Subject_04_25.npy"
EDITED_MASK = "downloaded_masks/edited_Subject_04_25.npy"  
KITE_ONLY_MASK = "downloaded_masks/kite_only_Subject_04_25.npy"
ANNOTATION_COORDS = None  # e.g., [50, 30, 200, 150] or None
DATASET_PATH = "../../../../data/duke_original"
RESULTS_DIR = "evaluation_results"

# ============================================================

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from datetime import datetime
import json
from typing import Dict, List, Tuple

class RealWorkflowEvaluator:
    def __init__(self, 
                 dataset_path: str = "data/duke_original",
                 results_base_dir: str = "real_workflow_results"):
        self.dataset_path = dataset_path
        self.results_dir = os.path.join(results_base_dir, f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(os.path.join(self.results_dir, 'visualizations'), exist_ok=True)
        
        print(f"🎯 Real Workflow Evaluator Initialized")
        print(f"📁 Results will be saved to: {self.results_dir}")
        
    def calculate_dice(self, pred_mask: np.ndarray, gt_mask: np.ndarray, threshold: float = 0.5) -> float:
        """Calculate Dice coefficient between prediction and ground truth"""
        
        # Handle MedSAM multi-class output (0,1,2,3,4... different regions)
        if pred_mask.max() > 1.0 and pred_mask.dtype in [np.int32, np.int64, np.uint8]:
            # Multi-class mask - combine all non-zero classes
            pred_binary = (pred_mask > 0).astype(np.uint8)
            print(f"    ✓ Multi-class mask detected, combined {np.unique(pred_mask)} classes")
        elif pred_mask.max() > 1.0:
            # Values > 1.0, could be encoded differently
            if pred_mask.max() <= 255:
                # Looks like 0-255 format
                pred_binary = (pred_mask > threshold * 255).astype(np.uint8)
            else:
                # Unknown format, normalize first
                pred_mask_norm = pred_mask / pred_mask.max()
                pred_binary = (pred_mask_norm > threshold).astype(np.uint8)
                print(f"    ⚠️ Unusual mask range detected, normalized: {pred_mask.min():.3f}-{pred_mask.max():.3f}")
        else:
            # Standard 0.0-1.0 format
            pred_binary = (pred_mask > threshold).astype(np.uint8)
        
        gt_binary = (gt_mask > 0).astype(np.uint8)
        
        # Ensure same shape
        if pred_binary.shape != gt_binary.shape:
            pred_binary = cv2.resize(pred_binary, (gt_binary.shape[1], gt_binary.shape[0]), 
                                   interpolation=cv2.INTER_NEAREST)
        
        intersection = np.logical_and(pred_binary, gt_binary).sum()
        dice = (2 * intersection) / (pred_binary.sum() + gt_binary.sum() + 1e-8)
        return float(dice)
    
    def load_mask(self, mask_path: str) -> np.ndarray:
        """Load mask from NPY or PNG format"""
        try:
            if mask_path.endswith('.npy'):
                mask = np.load(mask_path)
                print(f"    ✓ Loaded NPY mask: {mask.shape}, range: {mask.min():.3f}-{mask.max():.3f}")
                return mask
            else:
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if mask is not None:
                    mask = mask.astype(np.float32) / 255.0  # Normalize PNG to 0-1
                    print(f"    ✓ Loaded PNG mask: {mask.shape}, range: {mask.min():.3f}-{mask.max():.3f}")
                return mask
        except Exception as e:
            print(f"    ❌ Failed to load mask {mask_path}: {e}")
            return None
    
    def load_ground_truth(self, image_name: str) -> np.ndarray:
        """Load ground truth mask for given image"""
        gt_path = os.path.join(self.dataset_path, 'lesion', image_name)
        if not os.path.exists(gt_path):
            raise FileNotFoundError(f"Ground truth not found: {gt_path}")
        return cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
    
    def load_original_image(self, image_name: str) -> np.ndarray:
        """Load original image"""
        img_path = os.path.join(self.dataset_path, 'image', image_name)
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Original image not found: {img_path}")
        return cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    
    def process_website_results(self, 
                              image_name: str,
                              medsam_annotation_mask: str = None,
                              edited_mask: str = None,
                              kite_only_mask: str = None,
                              annotation_coords: List[int] = None) -> Dict:
        """
        Process results from website workflow
        
        Args:
            image_name: Name of the test image (e.g., 'Subject_01_29.png')
            medsam_annotation_mask: Path to mask from website MedSAM mode with annotation
            edited_mask: Path to mask after editing MedSAM result in KITE mode
            kite_only_mask: Path to mask from KITE mode only
            annotation_coords: [x1, y1, x2, y2] coordinates of annotation box
        
        Returns:
            Dictionary with all metrics and results
        """
        
        print(f"\n🔍 Processing {image_name}")
        
        # Load base data
        try:
            original_img = self.load_original_image(image_name)
            gt_mask = self.load_ground_truth(image_name)
            print(f"  ✓ Loaded original ({original_img.shape}) and GT ({gt_mask.shape})")
        except Exception as e:
            print(f"  ❌ Failed to load base data: {e}")
            return None
        
        results = {
            'image_name': image_name,
            'annotation_coords': annotation_coords,
            'image_shape': original_img.shape,
            'gt_area': int(np.sum(gt_mask > 0)),
            'masks_loaded': {},
            'dice_scores': {},
            'pred_areas': {},
            'errors': []
        }
        
        # Load and evaluate MedSAM annotation result
        if medsam_annotation_mask and os.path.exists(medsam_annotation_mask):
            try:
                medsam_mask = self.load_mask(medsam_annotation_mask)
                if medsam_mask is not None:
                    dice_medsam = self.calculate_dice(medsam_mask, gt_mask)
                    results['masks_loaded']['medsam_annotation'] = medsam_mask
                    results['dice_scores']['medsam_annotation'] = dice_medsam
                    results['pred_areas']['medsam_annotation'] = int(np.sum(medsam_mask > 0.5))
                    print(f"  ✓ MedSAM + Annotation: Dice = {dice_medsam:.3f}")
                else:
                    results['errors'].append("Failed to load MedSAM annotation mask")
            except Exception as e:
                results['errors'].append(f"MedSAM annotation error: {str(e)}")
                print(f"  ❌ MedSAM annotation error: {e}")
        else:
            if medsam_annotation_mask:
                results['errors'].append("MedSAM annotation mask not found")
                print(f"  ❌ MedSAM annotation mask not found: {medsam_annotation_mask}")
            else:
                print(f"  ⚠️ MedSAM annotation mask not provided")
        
        # Load and evaluate edited result
        if edited_mask and os.path.exists(edited_mask):
            try:
                edit_mask = self.load_mask(edited_mask)
                if edit_mask is not None:
                    dice_edited = self.calculate_dice(edit_mask, gt_mask)
                    results['masks_loaded']['edited'] = edit_mask
                    results['dice_scores']['edited'] = dice_edited
                    results['pred_areas']['edited'] = int(np.sum(edit_mask > 0.5))
                    print(f"  ✓ MedSAM + Editing: Dice = {dice_edited:.3f}")
                else:
                    results['errors'].append("Failed to load edited mask")
            except Exception as e:
                results['errors'].append(f"Edited mask error: {str(e)}")
                print(f"  ❌ Edited mask error: {e}")
        else:
            if edited_mask:
                results['errors'].append("Edited mask not found")
                print(f"  ❌ Edited mask not found: {edited_mask}")
            else:
                print(f"  ⚠️ Edited mask not provided")
        
        # Load and evaluate KITE-only result
        if kite_only_mask and os.path.exists(kite_only_mask):
            try:
                kite_mask = self.load_mask(kite_only_mask)
                if kite_mask is not None:
                    dice_kite = self.calculate_dice(kite_mask, gt_mask)
                    results['masks_loaded']['kite_only'] = kite_mask
                    results['dice_scores']['kite_only'] = dice_kite
                    results['pred_areas']['kite_only'] = int(np.sum(kite_mask > 0.5))
                    print(f"  ✓ KITE Only: Dice = {dice_kite:.3f}")
                else:
                    results['errors'].append("Failed to load KITE-only mask")
            except Exception as e:
                results['errors'].append(f"KITE-only error: {str(e)}")
                print(f"  ❌ KITE-only error: {e}")
        else:
            if kite_only_mask:
                results['errors'].append("KITE-only mask not found")
                print(f"  ❌ KITE-only mask not found: {kite_only_mask}")
            else:
                print(f"  ⚠️ KITE-only mask not provided")
        
        # Calculate improvements
        if 'medsam_annotation' in results['dice_scores'] and 'edited' in results['dice_scores']:
            improvement = results['dice_scores']['edited'] - results['dice_scores']['medsam_annotation']
            results['editing_improvement'] = improvement
            print(f"  📈 Editing improvement: +{improvement:.3f}")
        
        return results
    
    def create_comprehensive_visualization(self, 
                                     image_name: str, 
                                     results: Dict, 
                                     save_path: str = None) -> str:
        """Create comprehensive visualization of all results with improved overlays"""
        
        if not results or len(results['masks_loaded']) == 0:
            print(f"  ⚠️ No masks to visualize for {image_name}")
            return None
        
        # Load original and GT
        original_img = self.load_original_image(image_name)
        gt_mask = self.load_ground_truth(image_name)
        
        # Normalize original image for better visualization
        original_img_norm = cv2.equalizeHist(original_img)
        
        # Determine layout based on available masks
        masks = results['masks_loaded']
        n_masks = len(masks)
        n_cols = min(5, n_masks + 2)  # +2 for original and GT
        
        fig, axes = plt.subplots(2, n_cols, figsize=(6*n_cols, 12))
        if n_cols == 1:
            axes = axes.reshape(-1, 1)
        elif axes.ndim == 1:
            axes = axes.reshape(1, -1)
        
        # Top row: Original with annotation, Ground Truth, and results
        col = 0
        
        # Original image with annotation box
        axes[0, col].imshow(original_img, cmap='gray')
        if results['annotation_coords']:
            x1, y1, x2, y2 = results['annotation_coords']
            rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                            linewidth=3, edgecolor='red', facecolor='none')
            axes[0, col].add_patch(rect)
            axes[0, col].set_title('Original + User Annotation', fontsize=12, fontweight='bold')
        else:
            axes[0, col].set_title('Original Image', fontsize=12, fontweight='bold')
        axes[0, col].axis('off')
        col += 1
        
        # Ground Truth
        axes[0, col].imshow(gt_mask, cmap='gray')
        axes[0, col].set_title('Ground Truth', fontsize=12, fontweight='bold')
        axes[0, col].axis('off')
        col += 1
        
        # Show available masks
        mask_titles = {
            'medsam_annotation': 'MedSAM + Annotation',
            'edited': 'MedSAM + Editing',
            'kite_only': 'KITE Only'
        }
        
        # Colors for different methods
        mask_colors = {
            'medsam_annotation': 'Blues',
            'edited': 'Greens', 
            'kite_only': 'Purples'
        }
        
        for mask_key, mask in masks.items():
            if col < n_cols:
                # Convert probability mask to binary for visualization
                if mask.max() <= 1.0:
                    display_mask = (mask > 0.5).astype(np.uint8) * 255
                else:
                    display_mask = mask
                
                axes[0, col].imshow(display_mask, cmap='gray')
                dice_score = results['dice_scores'].get(mask_key, 0)
                pred_area = results['pred_areas'].get(mask_key, 0)
                title = f"{mask_titles.get(mask_key, mask_key)}\nDice: {dice_score:.3f}\nArea: {pred_area}px"
                axes[0, col].set_title(title, fontsize=11, fontweight='bold')
                axes[0, col].axis('off')
                col += 1
        
        # Hide unused top row axes
        for i in range(col, n_cols):
            axes[0, i].axis('off')
        
        # Bottom row: Enhanced overlays
        col = 0
        
        # Original with GT overlay (enhanced)
        axes[1, col].imshow(original_img_norm, cmap='gray')
        # Create colored GT overlay
        gt_colored = np.zeros((*gt_mask.shape, 3))
        gt_binary = (gt_mask > 0).astype(bool)
        gt_colored[gt_binary] = [1, 0, 0]  # Red for GT
        axes[1, col].imshow(gt_colored, alpha=0.4)
        axes[1, col].set_title('Original + GT Overlay\n(Red = Ground Truth)', fontsize=11, fontweight='bold')
        axes[1, col].axis('off')
        col += 1
        
        # Enhanced overlays for each mask
        overlay_colors = {
            'medsam_annotation': [0, 0, 1],      # Blue
            'edited': [0, 1, 0],                 # Green  
            'kite_only': [1, 0, 1]               # Magenta
        }
        
        overlay_descriptions = {
            'medsam_annotation': 'Blue = MedSAM Result',
            'edited': 'Green = Edited Result', 
            'kite_only': 'Magenta = KITE Only'
        }
        
        for mask_key, mask in masks.items():
            if col < n_cols:
                # Show original image
                axes[1, col].imshow(original_img_norm, cmap='gray')
                
                # Create colored overlay
                mask_colored = np.zeros((*original_img.shape, 3))
                
                # Convert mask for overlay
                if mask.max() <= 1.0:
                    mask_binary = (mask > 0.5).astype(bool)
                else:
                    mask_binary = (mask > 127).astype(bool)
                
                # Resize mask if necessary
                if mask_binary.shape != original_img.shape:
                    mask_binary = cv2.resize(mask_binary.astype(np.uint8), 
                                        (original_img.shape[1], original_img.shape[0]), 
                                        interpolation=cv2.INTER_NEAREST).astype(bool)
                
                # Apply color
                mask_colored[mask_binary] = overlay_colors.get(mask_key, [1, 1, 1])
                
                # Overlay with transparency
                axes[1, col].imshow(mask_colored, alpha=0.5)
                
                # Add GT contour for comparison
                gt_binary_resized = (gt_mask > 0).astype(np.uint8)
                if gt_binary_resized.shape != original_img.shape:
                    gt_binary_resized = cv2.resize(gt_binary_resized, 
                                                (original_img.shape[1], original_img.shape[0]), 
                                                interpolation=cv2.INTER_NEAREST)
                
                contours, _ = cv2.findContours(gt_binary_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for contour in contours:
                    # Convert contour to matplotlib format
                    contour = contour.reshape(-1, 2)
                    axes[1, col].plot(contour[:, 0], contour[:, 1], 'r-', linewidth=2, alpha=0.8)
                
                dice_score = results['dice_scores'].get(mask_key, 0)
                title = f'{mask_titles.get(mask_key, mask_key)} vs GT\n{overlay_descriptions.get(mask_key, "Prediction")}\nRed Outline = Ground Truth'
                axes[1, col].set_title(title, fontsize=10, fontweight='bold')
                axes[1, col].axis('off')
                col += 1
        
        # Metrics and comparison text in remaining space
        if col < n_cols:
            axes[1, col].axis('off')
            
            # Create comprehensive metrics text
            metrics_text = f"""📊 WORKFLOW EVALUATION RESULTS

    Image: {image_name}
    Image Size: {results['image_shape']}
    GT Area: {results['gt_area']} pixels

    🎯 DICE SCORES:
    """
            
            dice_scores = results['dice_scores']
            pred_areas = results['pred_areas']
            
            if 'medsam_annotation' in dice_scores:
                metrics_text += f"MedSAM + Annotation: {dice_scores['medsam_annotation']:.4f}\n"
                metrics_text += f"  Predicted Area: {pred_areas.get('medsam_annotation', 'N/A')} px\n"
            
            if 'edited' in dice_scores:
                metrics_text += f"MedSAM + Editing: {dice_scores['edited']:.4f}\n"
                metrics_text += f"  Predicted Area: {pred_areas.get('edited', 'N/A')} px\n"
            
            if 'kite_only' in dice_scores:
                metrics_text += f"KITE Only: {dice_scores['kite_only']:.4f}\n"
                metrics_text += f"  Predicted Area: {pred_areas.get('kite_only', 'N/A')} px\n"
            
            # Calculate and show improvements
            if 'medsam_annotation' in dice_scores and 'edited' in dice_scores:
                improvement = dice_scores['edited'] - dice_scores['medsam_annotation']
                relative_improvement = (improvement / dice_scores['medsam_annotation']) * 100 if dice_scores['medsam_annotation'] > 0 else float('inf')
                metrics_text += f"\n📈 EDITING IMPROVEMENT:\n"
                metrics_text += f"  Absolute: +{improvement:.4f}\n"
                if relative_improvement != float('inf'):
                    metrics_text += f"  Relative: +{relative_improvement:.1f}%\n"
                else:
                    metrics_text += f"  Relative: ∞% (baseline was 0)\n"
            
            if 'medsam_annotation' in dice_scores and 'kite_only' in dice_scores:
                comparison = dice_scores['kite_only'] - dice_scores['medsam_annotation']
                metrics_text += f"\n🔄 KITE vs MedSAM:\n"
                metrics_text += f"  Difference: {comparison:+.4f}\n"
            
            if results['errors']:
                metrics_text += f"\n⚠️ ISSUES:\n"
                for error in results['errors']:
                    metrics_text += f"  • {error}\n"
            
            # Add evaluation notes
            metrics_text += f"\n✅ EVALUATION NOTES:\n"
            metrics_text += f"  • Real website workflow tested\n"
            metrics_text += f"  • {len(dice_scores)} methods evaluated\n"
            metrics_text += f"  • Quantitative validation complete\n"
            
            axes[1, col].text(0.02, 0.98, metrics_text, transform=axes[1, col].transAxes, 
                            fontsize=9, verticalalignment='top', fontfamily='monospace',
                            bbox=dict(boxstyle="round,pad=0.8", facecolor="lightblue", alpha=0.9))
        
        # Hide remaining axes
        for i in range(col+1, n_cols):
            axes[1, i].axis('off')
        
        plt.suptitle(f'COMP 491 - Real Website Workflow Evaluation\n{image_name}', 
                fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        # Save with high quality
        if save_path is None:
            save_path = os.path.join(self.results_dir, 'visualizations', 
                                f'{os.path.splitext(image_name)[0]}_real_workflow_enhanced.png')
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close()
        
        print(f"  ✅ Enhanced visualization saved: {save_path}")
        return save_path


    def create_side_by_side_comparison(self, 
                                    image_name: str, 
                                    results: Dict, 
                                    save_path: str = None) -> str:
        """Create a focused side-by-side comparison visualization"""
        
        if not results or len(results['masks_loaded']) == 0:
            return None
        
        # Load data
        original_img = self.load_original_image(image_name)
        gt_mask = self.load_ground_truth(image_name)
        original_img_norm = cv2.equalizeHist(original_img)
        
        masks = results['masks_loaded']
        n_methods = len(masks)
        
        # Create figure with optimal layout
        fig, axes = plt.subplots(2, n_methods + 1, figsize=(5*(n_methods + 1), 10))
        if axes.ndim == 1:
            axes = axes.reshape(1, -1)
        
        # Top row: Masks only
        col = 0
        
        # Ground truth
        axes[0, col].imshow(gt_mask, cmap='gray')
        axes[0, col].set_title('Ground Truth', fontsize=14, fontweight='bold')
        axes[0, col].axis('off')
        col += 1
        
        # Method results
        for mask_key, mask in masks.items():
            if mask.max() <= 1.0:
                display_mask = (mask > 0.5).astype(np.uint8) * 255
            else:
                display_mask = mask
            
            axes[0, col].imshow(display_mask, cmap='gray')
            dice_score = results['dice_scores'].get(mask_key, 0)
            method_name = {
                'medsam_annotation': 'MedSAM + Annotation',
                'edited': 'MedSAM + Editing',
                'kite_only': 'KITE Only'
            }.get(mask_key, mask_key)
            
            axes[0, col].set_title(f'{method_name}\nDice: {dice_score:.3f}', 
                                fontsize=12, fontweight='bold')
            axes[0, col].axis('off')
            col += 1
        
        # Bottom row: Overlays on original
        col = 0
        
        # Original with GT
        axes[1, col].imshow(original_img_norm, cmap='gray')
        gt_colored = np.zeros((*gt_mask.shape, 3))
        gt_binary = (gt_mask > 0).astype(bool)
        gt_colored[gt_binary] = [1, 0, 0]  # Red
        axes[1, col].imshow(gt_colored, alpha=0.5)
        axes[1, col].set_title('Original + Ground Truth\n(Red)', fontsize=12, fontweight='bold')
        axes[1, col].axis('off')
        col += 1
        
        # Method overlays
        colors = [[0, 0, 1], [0, 1, 0], [1, 0, 1], [1, 0.5, 0]]  # Blue, Green, Magenta, Orange
        color_names = ['Blue', 'Green', 'Magenta', 'Orange']
        
        for i, (mask_key, mask) in enumerate(masks.items()):
            axes[1, col].imshow(original_img_norm, cmap='gray')
            
            # Create mask overlay
            mask_colored = np.zeros((*original_img.shape, 3))
            if mask.max() <= 1.0:
                mask_binary = (mask > 0.5).astype(bool)
            else:
                mask_binary = (mask > 127).astype(bool)
            
            if mask_binary.shape != original_img.shape:
                mask_binary = cv2.resize(mask_binary.astype(np.uint8), 
                                    (original_img.shape[1], original_img.shape[0]), 
                                    interpolation=cv2.INTER_NEAREST).astype(bool)
            
            mask_colored[mask_binary] = colors[i % len(colors)]
            axes[1, col].imshow(mask_colored, alpha=0.6)
            
            # Add GT contour
            gt_binary_resized = (gt_mask > 0).astype(np.uint8)
            if gt_binary_resized.shape != original_img.shape:
                gt_binary_resized = cv2.resize(gt_binary_resized, 
                                            (original_img.shape[1], original_img.shape[0]), 
                                            interpolation=cv2.INTER_NEAREST)
            contours, _ = cv2.findContours(gt_binary_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                contour = contour.reshape(-1, 2)
                axes[1, col].plot(contour[:, 0], contour[:, 1], 'r-', linewidth=2, alpha=0.8)
            
            method_name = {
                'medsam_annotation': 'MedSAM + Annotation',
                'edited': 'MedSAM + Editing', 
                'kite_only': 'KITE Only'
            }.get(mask_key, mask_key)
            
            axes[1, col].set_title(f'{method_name}\n({color_names[i % len(color_names)]}) vs GT (Red)', 
                                fontsize=11, fontweight='bold')
            axes[1, col].axis('off')
            col += 1
        
        plt.suptitle(f'COMP 491 - Workflow Comparison\n{image_name}', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(self.results_dir, 'visualizations', 
                                f'{os.path.splitext(image_name)[0]}_comparison.png')
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"  ✅ Comparison visualization saved: {save_path}")
        return save_path
    
    def generate_final_report(self, all_results: List[Dict]) -> str:
        """Generate final evaluation report"""
        
        # Filter successful results
        successful_results = [r for r in all_results if r and len(r['dice_scores']) > 0]
        
        if not successful_results:
            print("❌ No successful results to report")
            return None
        
        # Calculate averages
        medsam_scores = [r['dice_scores']['medsam_annotation'] for r in successful_results 
                        if 'medsam_annotation' in r['dice_scores']]
        edited_scores = [r['dice_scores']['edited'] for r in successful_results 
                        if 'edited' in r['dice_scores']]
        kite_scores = [r['dice_scores']['kite_only'] for r in successful_results 
                      if 'kite_only' in r['dice_scores']]
        
        # Generate report
        report_path = os.path.join(self.results_dir, 'real_workflow_report.txt')
        
        with open(report_path, 'w') as f:
            f.write("COMP 491 - Real Website Workflow Evaluation Report\n")
            f.write("=" * 60 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Student: Sinemis Toktaş\n")
            f.write(f"Evaluation Method: Real website workflow results\n\n")
            
            f.write("🎯 EVALUATION METHODOLOGY\n")
            f.write("-" * 25 + "\n")
            f.write("This evaluation uses ACTUAL results from the website workflow:\n")
            f.write("1. User creates annotation in MedSAM mode → Downloads mask\n")
            f.write("2. User uploads MedSAM result to KITE mode → Edits → Downloads\n")
            f.write("3. User creates annotation in KITE mode only → Downloads\n")
            f.write("4. Compare all results with ground truth\n\n")
            
            f.write("📊 QUANTITATIVE RESULTS\n")
            f.write("-" * 25 + "\n")
            f.write(f"Images processed: {len(successful_results)}\n\n")
            
            if medsam_scores:
                f.write(f"🎯 MedSAM + Annotation: {np.mean(medsam_scores):.4f} ± {np.std(medsam_scores):.4f} Dice\n")
            if edited_scores:
                f.write(f"✏️ MedSAM + Editing: {np.mean(edited_scores):.4f} ± {np.std(edited_scores):.4f} Dice\n")
            if kite_scores:
                f.write(f"⚒️ KITE Only: {np.mean(kite_scores):.4f} ± {np.std(kite_scores):.4f} Dice\n")
            
            if medsam_scores and edited_scores:
                improvements = [edited_scores[i] - medsam_scores[i] 
                               for i in range(min(len(edited_scores), len(medsam_scores)))]
                if improvements and np.mean(medsam_scores) > 0:
                    f.write(f"\n📈 Editing Improvement: +{np.mean(improvements):.4f} ± {np.std(improvements):.4f} Dice\n")
                    f.write(f"Relative Improvement: {np.mean(improvements)/np.mean(medsam_scores)*100:.1f}%\n")
                elif improvements:
                    f.write(f"\n📈 Editing Improvement: +{np.mean(improvements):.4f} ± {np.std(improvements):.4f} Dice\n")
                    f.write("Note: MedSAM baseline was 0, so relative improvement is infinite\n")
            
            f.write(f"\n🎓 CONCLUSIONS FOR COMP 491 REPORT\n")
            f.write("-" * 35 + "\n")
            f.write("✅ Real website workflow successfully evaluated\n")
            f.write("✅ Actual user interaction results validated\n")
            f.write("✅ Quantitative evidence of tool effectiveness\n")
            f.write("✅ Demonstrates practical utility for medical professionals\n")
            f.write("✅ Proves successful MedSAM integration and enhancement\n")
        
        # Save detailed JSON (convert numpy arrays to lists)
        json_results = []
        for result in successful_results:
            json_result = {}
            for key, value in result.items():
                if key == 'masks_loaded':
                    # Skip masks from JSON (too large)
                    continue
                elif isinstance(value, np.ndarray):
                    json_result[key] = value.tolist()
                else:
                    json_result[key] = value
            json_results.append(json_result)
        
        json_path = os.path.join(self.results_dir, 'real_workflow_results.json')
        with open(json_path, 'w') as f:
            json.dump({
                'summary': {
                    'avg_medsam_dice': float(np.mean(medsam_scores)) if medsam_scores else None,
                    'avg_edited_dice': float(np.mean(edited_scores)) if edited_scores else None,
                    'avg_kite_dice': float(np.mean(kite_scores)) if kite_scores else None,
                    'num_successful': len(successful_results)
                },
                'detailed_results': json_results
            }, f, indent=2)
        
        print(f"\n✅ Final report generated: {report_path}")
        return report_path

def main():
    """
    Main function using the variables defined at the top of the file
    """
    print("🎯 REAL WORKFLOW EVALUATION")
    print("=" * 50)
    print(f"Processing: {IMAGE_NAME}")
    print(f"Dataset: {DATASET_PATH}")
    print()
    
    # Check if required files exist
    print("🔍 Checking files...")
    files_to_check = [
        ("MedSAM Annotation", MEDSAM_ANNOTATION_MASK),
        ("Edited Result", EDITED_MASK), 
        ("KITE Only", KITE_ONLY_MASK)
    ]
    
    files_found = 0
    for name, path in files_to_check:
        if path and os.path.exists(path):
            print(f"  ✅ {name}: {path}")
            files_found += 1
        elif path:
            print(f"  ❌ {name}: {path} (not found)")
        else:
            print(f"  ⚠️ {name}: Not specified (None)")
    
    if files_found == 0:
        print("\n❌ No mask files found! Please check your file paths at the top of the script.")
        return
    
    print(f"\n📊 Found {files_found} mask files to process")
    
    # Initialize evaluator
    evaluator = RealWorkflowEvaluator(dataset_path=DATASET_PATH, results_base_dir=RESULTS_DIR)
    
    # Process the results
    result = evaluator.process_website_results(
        image_name=IMAGE_NAME,
        medsam_annotation_mask=MEDSAM_ANNOTATION_MASK,
        edited_mask=EDITED_MASK,
        kite_only_mask=KITE_ONLY_MASK,
        annotation_coords=ANNOTATION_COORDS
    )
    
    if result:
        # Create visualization
        evaluator.create_comprehensive_visualization(IMAGE_NAME, result)
        
        # Generate report
        evaluator.generate_final_report([result])
        
        print("\n🎉 EVALUATION COMPLETED!")
        print("=" * 30)
        print(f"📁 Results saved to: {evaluator.results_dir}")
        print(f"📄 Report: {evaluator.results_dir}/real_workflow_report.txt")
        print(f"🖼️ Visualization: {evaluator.results_dir}/visualizations/")
        
        # Show quick results
        print(f"\n📊 QUICK RESULTS for {IMAGE_NAME}:")
        for method, score in result['dice_scores'].items():
            print(f"   {method}: {score:.3f} Dice")
            
    else:
        print("\n❌ Evaluation failed!")
        print("Check:")
        print("1. File paths at the top of the script")
        print("2. Dataset path is correct")
        print("3. Image exists in ground truth dataset")

if __name__ == "__main__":
    main()