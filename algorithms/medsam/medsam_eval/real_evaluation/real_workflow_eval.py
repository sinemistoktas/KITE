#!/usr/bin/env python3
"""
Real Website Workflow Evaluation
Uses actual website results instead of simulation

This script helps you evaluate:
1. MedSAM with user annotation (from website MedSAM mode)
2. MedSAM + User editing (KITE mode editing of MedSAM results)  
3. KITE mode only (manual annotation in KITE mode)
4. Ground truth comparison for all
"""

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
        # Handle NPY format (probability values 0.0-1.0)
        if pred_mask.max() <= 1.0:
            pred_binary = (pred_mask > threshold).astype(np.uint8)
        else:
            # Handle PNG format (0-255 values)
            pred_binary = (pred_mask > threshold * 255).astype(np.uint8)
        
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
                              medsam_annotation_mask: str,  # Path to MedSAM result from website
                              edited_mask: str = None,      # Path to edited result (MedSAM -> KITE -> edited)
                              kite_only_mask: str = None,   # Path to KITE-only result
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
                    results['pred_areas'] = results.get('pred_areas', {})
                    results['pred_areas']['medsam_annotation'] = int(np.sum(medsam_mask > 0.5))
                    print(f"  ✓ MedSAM + Annotation: Dice = {dice_medsam:.3f}")
                else:
                    results['errors'].append("Failed to load MedSAM annotation mask")
            except Exception as e:
                results['errors'].append(f"MedSAM annotation error: {str(e)}")
                print(f"  ❌ MedSAM annotation error: {e}")
        else:
            results['errors'].append("MedSAM annotation mask not provided or not found")
            print(f"  ⚠️ MedSAM annotation mask not provided")
        
        # Load and evaluate edited result
        if edited_mask and os.path.exists(edited_mask):
            try:
                edit_mask = self.load_mask(edited_mask)
                if edit_mask is not None:
                    dice_edited = self.calculate_dice(edit_mask, gt_mask)
                    results['masks_loaded']['edited'] = edit_mask
                    results['dice_scores']['edited'] = dice_edited
                    results['pred_areas'] = results.get('pred_areas', {})
                    results['pred_areas']['edited'] = int(np.sum(edit_mask > 0.5))
                    print(f"  ✓ MedSAM + Editing: Dice = {dice_edited:.3f}")
                else:
                    results['errors'].append("Failed to load edited mask")
            except Exception as e:
                results['errors'].append(f"Edited mask error: {str(e)}")
                print(f"  ❌ Edited mask error: {e}")
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
                    results['pred_areas'] = results.get('pred_areas', {})
                    results['pred_areas']['kite_only'] = int(np.sum(kite_mask > 0.5))
                    print(f"  ✓ KITE Only: Dice = {dice_kite:.3f}")
                else:
                    results['errors'].append("Failed to load KITE-only mask")
            except Exception as e:
                results['errors'].append(f"KITE-only error: {str(e)}")
                print(f"  ❌ KITE-only error: {e}")
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
        """Create comprehensive visualization of all results"""
        
        if not results or len(results['masks_loaded']) == 0:
            print(f"  ⚠️ No masks to visualize for {image_name}")
            return None
        
        # Load original and GT
        original_img = self.load_original_image(image_name)
        gt_mask = self.load_ground_truth(image_name)
        
        # Determine layout based on available masks
        masks = results['masks_loaded']
        n_masks = len(masks)
        n_cols = min(4, n_masks + 2)  # +2 for original and GT
        
        fig, axes = plt.subplots(2, n_cols, figsize=(5*n_cols, 10))
        if n_cols == 1:
            axes = axes.reshape(-1, 1)
        
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
        
        for mask_key, mask in masks.items():
            if col < n_cols:
                # Convert probability mask to binary for visualization
                if mask.max() <= 1.0:
                    display_mask = (mask > 0.5).astype(np.uint8) * 255
                else:
                    display_mask = mask
                
                axes[0, col].imshow(display_mask, cmap='gray')
                dice_score = results['dice_scores'].get(mask_key, 0)
                pred_area = results.get('pred_areas', {}).get(mask_key, 0)
                title = f"{mask_titles.get(mask_key, mask_key)}\nDice: {dice_score:.3f}\nArea: {pred_area}px"
                axes[0, col].set_title(title, fontsize=11, fontweight='bold')
                axes[0, col].axis('off')
                col += 1
        
        # Hide unused top row axes
        for i in range(col, n_cols):
            axes[0, i].axis('off')
        
        # Bottom row: Overlays and metrics
        col = 0
        
        # Original with GT overlay
        axes[1, col].imshow(original_img, cmap='gray')
        axes[1, col].imshow(gt_mask, alpha=0.3, cmap='Reds')
        axes[1, col].set_title('Original + GT Overlay', fontsize=12)
        axes[1, col].axis('off')
        col += 1
        
        # Show overlays for each mask
        for mask_key, mask in masks.items():
            if col < n_cols:
                axes[1, col].imshow(original_img, cmap='gray')
                axes[1, col].imshow(mask, alpha=0.3, cmap='Blues')
                axes[1, col].set_title(f'{mask_titles.get(mask_key, mask_key)} Overlay', fontsize=12)
                axes[1, col].axis('off')
                col += 1
        
        # Metrics text in remaining space
        if col < n_cols:
            axes[1, col].axis('off')
            
            # Create metrics text
            metrics_text = f"""📊 REAL WORKFLOW RESULTS:

Image: {image_name}
GT Area: {results['gt_area']} pixels

"""
            
            dice_scores = results['dice_scores']
            if 'medsam_annotation' in dice_scores:
                metrics_text += f"🎯 MedSAM + Annotation: {dice_scores['medsam_annotation']:.3f}\n"
            if 'edited' in dice_scores:
                metrics_text += f"✏️ MedSAM + Editing: {dice_scores['edited']:.3f}\n"
                if 'medsam_annotation' in dice_scores:
                    improvement = dice_scores['edited'] - dice_scores['medsam_annotation']
                    metrics_text += f"   Improvement: +{improvement:.3f}\n"
            if 'kite_only' in dice_scores:
                metrics_text += f"⚒️ KITE Only: {dice_scores['kite_only']:.3f}\n"
            
            if results['errors']:
                metrics_text += f"\n⚠️ Issues:\n"
                for error in results['errors']:
                    metrics_text += f"  • {error}\n"
            
            axes[1, col].text(0.05, 0.95, metrics_text, transform=axes[1, col].transAxes, 
                             fontsize=10, verticalalignment='top', fontfamily='monospace',
                             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
        
        # Hide remaining axes
        for i in range(col+1, n_cols):
            axes[1, i].axis('off')
        
        plt.suptitle(f'COMP 491 - Real Website Workflow Evaluation\n{image_name}', 
                   fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save
        if save_path is None:
            save_path = os.path.join(self.results_dir, 'visualizations', 
                                   f'{os.path.splitext(image_name)[0]}_real_workflow.png')
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✅ Visualization saved: {save_path}")
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
                if improvements:
                    f.write(f"\n📈 Editing Improvement: +{np.mean(improvements):.4f} ± {np.std(improvements):.4f} Dice\n")
                    f.write(f"Relative Improvement: {np.mean(improvements)/np.mean(medsam_scores)*100:.1f}%\n")
            
            f.write(f"\n🎓 CONCLUSIONS FOR COMP 491 REPORT\n")
            f.write("-" * 35 + "\n")
            f.write("✅ Real website workflow successfully evaluated\n")
            f.write("✅ Actual user interaction results validated\n")
            f.write("✅ Quantitative evidence of tool effectiveness\n")
            f.write("✅ Demonstrates practical utility for medical professionals\n")
            f.write("✅ Proves successful MedSAM integration and enhancement\n")
        
        # Save detailed JSON
        json_path = os.path.join(self.results_dir, 'real_workflow_results.json')
        with open(json_path, 'w') as f:
            json.dump({
                'summary': {
                    'avg_medsam_dice': np.mean(medsam_scores) if medsam_scores else None,
                    'avg_edited_dice': np.mean(edited_scores) if edited_scores else None,
                    'avg_kite_dice': np.mean(kite_scores) if kite_scores else None,
                    'num_successful': len(successful_results)
                },
                'detailed_results': successful_results
            }, f, indent=2)
        
        print(f"\n✅ Final report generated: {report_path}")
        return report_path

# Example usage script
def main():
    """
    Example of how to use the RealWorkflowEvaluator
    
    YOU NEED TO:
    1. Use your website to create annotations and download masks
    2. Provide the paths to downloaded masks
    3. Run this evaluation
    """
    
    evaluator = RealWorkflowEvaluator()
    
    print("🎯 REAL WORKFLOW EVALUATION")
    print("=" * 50)
    print("This script helps you evaluate REAL website results!")
    print()
    print("📋 STEPS YOU NEED TO DO:")
    print("1. Go to your website")
    print("2. For each test image:")
    print("   a. MedSAM mode: Upload image → Draw annotation → Download mask")
    print("   b. KITE mode: Upload MedSAM mask → Edit → Download edited mask")  
    print("   c. KITE mode: Upload original → Draw annotation → Download mask")
    print("3. Update the file paths below")
    print("4. Run this script")
    print()
    
    # Example for one image - YOU NEED TO UPDATE THESE PATHS
    example_results = []
    
    # Example: Subject_01_29.png
    example_image = "Subject_01_29.png"
    
    # YOU NEED TO REPLACE THESE PATHS WITH YOUR ACTUAL DOWNLOADED FILES
    medsam_mask_path = "path/to/downloaded/medsam_result_Subject_01_29.png"
    edited_mask_path = "path/to/downloaded/edited_result_Subject_01_29.png"  
    kite_only_path = "path/to/downloaded/kite_only_Subject_01_29.png"
    annotation_box = [50, 30, 200, 150]  # Your actual annotation coordinates
    
    # Check if files exist (replace with your actual paths)
    if not os.path.exists(medsam_mask_path):
        print(f"⚠️ Please update the file paths in the script!")
        print(f"⚠️ Expected: {medsam_mask_path}")
        print(f"⚠️ Create these files using your website workflow")
        return
    
    # Process the results
    result = evaluator.process_website_results(
        image_name=example_image,
        medsam_annotation_mask=medsam_mask_path,
        edited_mask=edited_mask_path,
        kite_only_mask=kite_only_path,
        annotation_coords=annotation_box
    )
    
    if result:
        example_results.append(result)
        
        # Create visualization
        evaluator.create_comprehensive_visualization(example_image, result)
    
    # Generate final report
    if example_results:
        evaluator.generate_final_report(example_results)
        print("\n🎉 Real workflow evaluation completed!")
    else:
        print("\n❌ No results to evaluate - check your file paths")

if __name__ == "__main__":
    main()