"""Populate external model choices before UI construction, never at import time."""

from modules import script_callbacks
from modules.llm.catalog import initialize


script_callbacks.on_before_ui(initialize)
