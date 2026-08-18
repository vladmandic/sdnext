from .models import detailer_models
from .helper import detailer_opt, DetailerResult, list_models, assign_prompts, parse_prompt_lines
from .detailer import Detailer


def initialize():
    from modules import shared
    shared.detailer = Detailer()
