# algorithms/medsam/__init__.py
"""
MedSAM package initialization
Using original MedSAM implementation
"""

# Import from the wrapper that uses original MedSAM
try:
    from .medsam_direct_wrapper import (
        get_original_medsam_instance,
        segment_with_box,
        validate_box,
        create_overlay,
        save_results
    )
    
    # Backward compatibility - provide the old interface
    def load_model():
        """Backward compatibility function"""
        return get_original_medsam_instance()
    
    __all__ = [
        'get_original_medsam_instance',
        'segment_with_box', 
        'validate_box',
        'create_overlay',
        'save_results',
        'load_model'  # For backward compatibility
    ]
    
    print("MedSAM package initialized with original MedSAM wrapper")
    
except ImportError as e:
    print(f"❌ Warning: Could not import MedSAM wrapper: {e}")
    
    # Try to fall back to your existing utils.py if it exists
    try:
        from .utils import validate_box, save_results, create_overlay
        
        # Create dummy functions for missing components
        def segment_with_box(img_path, box=None):
            """Fallback function - requires original MedSAM setup"""
            raise ImportError("Original MedSAM not properly configured. Please check setup.")
        
        def get_original_medsam_instance():
            """Fallback function"""
            raise ImportError("Original MedSAM not properly configured. Please check setup.")
        
        def load_model():
            """Fallback function"""
            return get_original_medsam_instance()
        
        __all__ = [
            'segment_with_box', 
            'load_model',
            'validate_box',
            'save_results',
            'create_overlay',
            'get_original_medsam_instance'
        ]
        
        print("⚠️  MedSAM package initialized with fallback functions")
        
    except ImportError as e2:
        print(f"❌ Error: Could not import any MedSAM implementation: {e2}")
        print("Please check:")
        print("1. segment-anything is installed: pip install segment-anything")
        print("2. MedSAM checkpoint is available")
        print("3. Original MedSAM files are in the correct location")
        
        # Provide minimal dummy functions to prevent import errors
        def segment_with_box(img_path, box=None):
            raise ImportError("MedSAM not properly configured")
        
        def get_original_medsam_instance():
            raise ImportError("MedSAM not properly configured")
        
        def load_model():
            raise ImportError("MedSAM not properly configured")
        
        def validate_box(box, H, W):
            return box
        
        def save_results(overlay_img, mask, results_dir, base_filename):
            raise ImportError("MedSAM not properly configured")
        
        def create_overlay(img, mask):
            raise ImportError("MedSAM not properly configured")
        
        __all__ = [
            'segment_with_box', 
            'load_model',
            'validate_box',
            'save_results',
            'create_overlay',
            'get_original_medsam_instance'
        ]