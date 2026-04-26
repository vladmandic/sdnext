"""Always-on script providing tag autocomplete dictionary management UI."""

import json
import threading
import gradio as gr
from modules import shared, scripts_manager
from modules.api import autocomplete as ac_api
from modules.ui_components import ToolButton
import modules.ui_symbols as symbols
from modules.logger import log


def get_all_names():
    """Merge local file names with cached remote manifest names."""
    local = ac_api.local_names()
    remote = set()
    cached = ac_api.manifest_cache.get('data')
    if cached:
        remote = {e['name'] for e in cached if 'name' in e}
    return sorted(local | remote)


def get_config_json():
    """Serialize autocomplete opts for the JS config bridge."""
    enabled = [n for n in shared.opts.data.get('autocomplete_enabled', []) if not n.endswith('.translations')]
    return json.dumps({
        "autocomplete_active": bool(shared.opts.data.get('autocomplete_active', False)),
        "autocomplete_enabled": enabled,
        "autocomplete_min_chars": shared.opts.data.get('autocomplete_min_chars', 3),
        "autocomplete_replace_underscores": shared.opts.data.get('autocomplete_replace_underscores', True),
        "autocomplete_append_comma": shared.opts.data.get('autocomplete_append_comma', True),
        "autocomplete_at_prefix_artist": shared.opts.data.get('autocomplete_at_prefix_artist', False),
        "autocomplete_translations": bool(shared.opts.data.get('autocomplete_translations', False)),
    })


def on_active_change(value):
    shared.opts.data['autocomplete_active'] = bool(value)
    shared.opts.save(silent=True)
    return get_config_json(), ""


def on_enabled_change(selected):
    shared.opts.data['autocomplete_enabled'] = list(selected)
    shared.opts.save(silent=True)
    return get_config_json(), ""


def on_min_chars_change(value):
    shared.opts.data['autocomplete_min_chars'] = int(value)
    shared.opts.save(silent=True)
    return get_config_json()


def on_replace_underscores_change(value):
    shared.opts.data['autocomplete_replace_underscores'] = bool(value)
    shared.opts.save(silent=True)
    return get_config_json()


def on_append_comma_change(value):
    shared.opts.data['autocomplete_append_comma'] = bool(value)
    shared.opts.save(silent=True)
    return get_config_json()


def on_at_prefix_artist_change(value):
    shared.opts.data['autocomplete_at_prefix_artist'] = bool(value)
    shared.opts.save(silent=True)
    return get_config_json()


def on_translations_change(value):
    enabled = bool(value)
    shared.opts.data['autocomplete_translations'] = enabled
    shared.opts.save(silent=True)
    # Drop cached entries so the new setting takes effect on the next get_content call.
    ac_api.cache.clear()
    # Reset the missing-companion warning set so the next request re-evaluates and re-logs if needed.
    ac_api.translations_warned.clear()
    # On enable, sync any missing companion files in the background; downloads land
    # while the user works and become visible on the next dict request.
    if enabled:
        threading.Thread(target=ac_api.sync_translations_for_enabled, daemon=True).start()
    return get_config_json()


def format_status(local, remote_entries, fetch_ok):
    """Build status HTML showing available dictionaries."""
    lines = []
    remote_names = set()
    for e in remote_entries:
        name = e.get('name', '')
        remote_names.add(name)
        dl_status = '' if name in local else symbols.save
        desc = e.get('description', '')
        # size = e.get('size_mb', 0)
        tags = e.get('tag_count', 0)
        lines.append(f"<b>{name}</b> | {desc} | {tags:,} tags {dl_status}")
    for name in sorted(local - remote_names):
        lines.append(f"<b>{name}</b>")
    if not fetch_ok:
        lines.insert(0, "<i>Remote fetch failed; showing local files only</i>")
    elif not lines:
        lines.append("No dictionaries found")
    return "<br>".join(lines)


def on_refresh():
    """Fetch remote manifest and update dropdown choices."""
    try:
        ac_api.manifest_cache.pop('fetched_at', None)  # force re-fetch by expiring cache
        ac_api.fetch_manifest_sync()
        fetch_ok = bool(ac_api.manifest_cache.get('fetched_at'))
        names = get_all_names()
        current = list(shared.opts.data.get('autocomplete_enabled', []))
        local = ac_api.local_names()
        remote_entries = ac_api.manifest_cache.get('data', [])
        msg = format_status(local, remote_entries, fetch_ok)
        return gr.update(choices=names, value=current), msg
    except Exception as e:
        log.warning(f"Autocomplete refresh: {e}")
        return gr.update(), f"Refresh failed: {e}"


def on_update(selected):
    """Re-download enabled dictionaries if remote version is newer."""
    if not selected:
        return "No dictionaries enabled"
    try:
        entries = ac_api.fetch_manifest_sync()
    except Exception as e:
        return f"Failed to fetch manifest: {e}"
    updated = []
    for name in selected:
        remote_entry = next((e for e in entries if e.get('name') == name), None)
        if not remote_entry:
            continue
        remote_ver = remote_entry.get('version', '')
        local_ver = ac_api.local_version(name)
        if not local_ver or (remote_ver and local_ver != remote_ver):
            try:
                ac_api.download_sync(name)
                updated.append(name)
            except Exception as e:
                log.warning(f"Autocomplete update {name}: {e}")
    if updated:
        return f"Updated: {', '.join(updated)}"
    return "All dictionaries are up to date"


class AutocompleteScript(scripts_manager.Script):

    def show(self, is_img2img):
        return scripts_manager.AlwaysVisible

    def title(self):
        return "Tag Autocomplete"

    def ui(self, is_img2img):
        initial_names = get_all_names()
        initial_enabled = list(shared.opts.data.get('autocomplete_enabled', []))

        with gr.Accordion('Tag Autocomplete', open=False, elem_id='autocomplete_settings'):
            with gr.Row():
                active_cb = gr.Checkbox(
                    label="Enable Autocomplete",
                    value=bool(shared.opts.data.get('autocomplete_active', False)),
                    elem_id=self.elem_id("active"),
                )
            with gr.Row():
                enabled_dd = gr.Dropdown(
                    label="Active dictionaries",
                    multiselect=True,
                    choices=initial_names,
                    value=initial_enabled,
                    interactive=True,
                    elem_id=self.elem_id("enabled"),
                )
                refresh_btn = ToolButton(value=symbols.refresh, elem_id=self.elem_id("refresh"))
                update_btn = ToolButton(value=symbols.save, elem_id=self.elem_id("update"))
            with gr.Row():
                replace_underscores = gr.Checkbox(
                    label="Replace underscores",
                    value=shared.opts.data.get('autocomplete_replace_underscores', True),
                    elem_id=self.elem_id("replace_underscores"),
                )
                append_comma = gr.Checkbox(
                    label="Comma separator",
                    value=shared.opts.data.get('autocomplete_append_comma', True),
                    elem_id=self.elem_id("append_comma"),
                )
                at_prefix_artist = gr.Checkbox(
                    label="Keep @ on artist insert",
                    value=shared.opts.data.get('autocomplete_at_prefix_artist', False),
                    elem_id=self.elem_id("at_prefix_artist"),
                )
                translations_cb = gr.Checkbox(
                    label="Foreign-term translations",
                    value=shared.opts.data.get('autocomplete_translations', False),
                    elem_id=self.elem_id("translations"),
                )
                min_chars = gr.Slider(
                    label="Min characters",
                    minimum=2, maximum=6, step=1,
                    value=shared.opts.data.get('autocomplete_min_chars', 3),
                    elem_id=self.elem_id("min_chars"),
                )
            with gr.Row():
                status = gr.HTML(value="", elem_id=self.elem_id("status"))
            config_json = gr.Textbox(
                value=get_config_json,
                visible=False,
                elem_id=self.elem_id("config_json"),
            )

        active_cb.change(fn=on_active_change, inputs=[active_cb], outputs=[config_json, status])
        enabled_dd.change(fn=on_enabled_change, inputs=[enabled_dd], outputs=[config_json, status])
        min_chars.change(fn=on_min_chars_change, inputs=[min_chars], outputs=[config_json])
        replace_underscores.change(fn=on_replace_underscores_change, inputs=[replace_underscores], outputs=[config_json])
        append_comma.change(fn=on_append_comma_change, inputs=[append_comma], outputs=[config_json])
        at_prefix_artist.change(fn=on_at_prefix_artist_change, inputs=[at_prefix_artist], outputs=[config_json])
        translations_cb.change(fn=on_translations_change, inputs=[translations_cb], outputs=[config_json])
        refresh_btn.click(fn=on_refresh, inputs=[], outputs=[enabled_dd, status])
        update_btn.click(fn=on_update, inputs=[enabled_dd], outputs=[status])

        for comp in [enabled_dd, min_chars, replace_underscores, append_comma, at_prefix_artist, translations_cb, config_json, status]:
            comp.do_not_save_to_config = True

        return [active_cb, enabled_dd, min_chars, replace_underscores, append_comma, at_prefix_artist, translations_cb, config_json]
