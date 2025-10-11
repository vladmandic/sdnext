"""
SeedVR2 Video Upscaler - Modular Architecture
Refactored from monolithic seedvr2.py for better maintainability

Author: Refactored codebase
Version: 2.0.0 - Modular

Available Modules:
- utils: Download and path utilities
- optimization: Memory, performance, and compatibility optimizations  
- core: Model management and generation pipeline (NEW)
- processing: Video and tensor processing (coming next)
- interfaces: ComfyUI integration
"""
'''
# Track which modules are available for progressive migration
MODULES_AVAILABLE = {
    'downloads': True,          # ✅ Module 1 - Downloads and model management
    'memory_manager': True,     # ✅ Module 2 - Memory optimization
    'performance': True,        # ✅ Module 3 - Performance optimizations  
    'compatibility': True,      # ✅ Module 4 - FP8/FP16 compatibility
    'model_manager': True,      # ✅ Module 5 - Model configuration and loading
    'generation': True,         # ✅ Module 6 - Generation loop and inference
    'video_transforms': True,   # ✅ Module 7 - Video processing and transforms
    'comfyui_node': True,       # ✅ Module 8 - ComfyUI node interface (COMPLETE!)
    'infer': True,              # ✅ Module 9 - Infer
}
'''
# Core imports (always available)
import os
import sys

# Add current directory to path for fallback imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
