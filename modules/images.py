from modules.image.metadata import image_data, read_info_from_image
from modules.image.save import save_image, sanitize_filename_part
from modules.image.resize import resize_image
from modules.image.namegen import FilenameGenerator, get_next_sequence_number
from modules.image.grid import Grid, image_grid, check_grid_size, get_grid_size, draw_grid_annotations, draw_prompt_matrix, combine_grid, get_font
from modules.image.util import draw_text, flatten

__all__ = [
    'check_grid_size',
    'combine_grid',
    'draw_grid_annotations',
    'draw_prompt_matrix',
    'FilenameGenerator',
    'get_grid_size',
    'Grid',
    'image_data',
    'image_grid',
    'read_info_from_image',
    'resize_image',
    'sanitize_filename_part',
    'save_image',
    'get_font',
    'get_next_sequence_number',
    'draw_text',
    'flatten',
]

def register_heif():
    from installer import install
    install('pillow-heif', quiet=True)
    try:
        import pillow_heif
        pillow_heif.register_heif_opener()
    except Exception:
        pass

register_heif()
