from .models import detailer_models
from .helper import DetailerResult, detailer_opt, list_models, get_mask
from .detailer import Detailer


def initialize():
    from modules import shared
    shared.detailer = Detailer()
