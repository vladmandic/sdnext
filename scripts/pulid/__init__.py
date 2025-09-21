"""
Credit and original implementation: <https://github.com/ToTheBeginning/PuLID>
"""

import os
import sys
from modules.errors import log
sys.path.append(os.path.dirname(__file__))
try:
    from pulid_sdxl import StableDiffusionXLPuLIDPipeline, StableDiffusionXLPuLIDPipelineImage, StableDiffusionXLPuLIDPipelineInpaint
    from pulid_flux import apply_flux, unapply_flux
    from pulid_utils import resize_numpy_image_long as resize
    import attention_processor as attention
    import pulid_sampling as sampling
except Exception as e:
    import traceback
    log.error(f'PuLID import error: {e}')
    print(traceback.format_exc())
    print(sys.exc_info()[0])
    raise ImportError(f'PuLID import error: {e}') from e
