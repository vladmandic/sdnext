"""
# Core imports (always available)
import os
import sys

# Add current directory to path for fallback imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
"""