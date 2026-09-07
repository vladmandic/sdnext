"""Helpers for filtering display-only UI choice lists."""


def filter_ui_choices(choices: list[str], preferences: list[str] | None = None, selected: str | None = None) -> tuple[list[str], bool]:
    """Return choices selected for display without changing the underlying catalog.

    Empty or stale preferences leave the complete list visible. A current value is
    always retained so loading a saved workflow cannot silently replace it.
    """
    available = list(dict.fromkeys(choices))
    preferred = set(preferences or [])
    filtered = [choice for choice in available if choice in preferred]
    if not filtered:
        return available, False
    if selected in available and selected not in filtered:
        filtered.append(selected)
    return filtered, len(filtered) < len(available)
