"""Canonical selection persistence and token defaults for existing LLM controls."""

from modules.llm.catalog import is_external
from modules.llm.openai import get_default_max_tokens


def bind_display_names(component, to_display, to_selection):
    """Persist canonical IDs while Gradio's string-only choices show UI aliases."""
    component.value_from_config = to_display
    component.value_to_config = to_selection


def bind_token_limit(model_component, limit_component, explicit_component, local_default, model_id=lambda value: value, active_model=None, is_active=None):
    """Use UiLoadsave's restoration hooks, not browser change/input events.

    Saved model values are applied before saved slider values. A saved limit is
    explicit even when it equals an old provider default; never guess from its
    numeric value. Gradio State carries that provenance into each UI session.
    """
    def restore_model(value):
        if is_active is not None and not is_active():
            return
        if active_model is not None:
            active_model.value = value
        if not explicit_component.value:
            limit_component.value = get_default_max_tokens() if is_external(model_id(value)) else local_default

    def restore_limit(_value):
        explicit_component.value = True

    model_component.init_field = restore_model
    limit_component.init_field = restore_limit
