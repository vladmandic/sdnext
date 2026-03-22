import gradio as gr
import installer
from modules import scripts_manager


def _is_rocm() -> bool:
    try:
        from scripts.rocm import rocm_mgr  # pylint: disable=no-name-in-module
        return rocm_mgr.is_rocm
    except Exception:
        return False


class Script(scripts_manager.Script):
    def title(self):
        return "ROCm: Advanced Config"

    def show(self, _is_img2img):
        if _is_rocm():
            return scripts_manager.AlwaysVisible
        return False

    def ui(self, _is_img2img):
        from scripts.rocm import rocm_mgr, rocm_vars  # pylint: disable=no-name-in-module

        if not rocm_mgr.is_rocm:
            with gr.Accordion('ROCm: Advanced Config', open=False, elem_id='rocm_config'):
                gr.HTML("<p><b>ROCm is not installed.</b> This section is disabled.</p>")
            return []

        config = rocm_mgr.load_config()
        var_names = []
        components = []

        def _make_component(name, meta, cfg):
            val = cfg.get(name, meta["default"])
            if meta["widget"] == "checkbox":
                return gr.Checkbox(label=meta["desc"], value=(val == "1"), elem_id=f"rocm_var_{name.lower()}")
            elif meta["widget"] == "dropdown":
                choices = rocm_mgr._dropdown_choices(meta["options"])
                display = rocm_mgr._dropdown_display(val, meta["options"])
                if display not in choices:  # corrupted stored value — fall back to default
                    display = rocm_mgr._dropdown_display(meta["default"], meta["options"])
                return gr.Dropdown(label=meta["desc"], choices=choices, value=display, elem_id=f"rocm_var_{name.lower()}")
            else:  # textbox
                return gr.Textbox(label=meta["desc"], value=rocm_mgr._expand_venv(val), lines=1)

        def _info_html():
            d = rocm_mgr.info()
            ts = "<style>#rocm_info_table table{width:100%;border-collapse:collapse;font-size:12px} #rocm_info_table td,#rocm_info_table th{padding:2px 8px;border-bottom:1px solid var(--sd-panel-border-color)} #rocm_info_table th{text-align:left;color:var(--sd-muted-color);font-weight:normal} #rocm_info_table td:first-child{color:var(--sd-muted-color);width:38%}</style>"
            rows = []
            def section(title):
                rows.append(f"<tr><th colspan='2' style='padding-top:6px;color:var(--highlight-color)'>{title}</th></tr>")
            def row(k, v):
                rows.append(f"<tr><td>{k}</td><td>{v}</td></tr>")
            section("ROCm / HIP")
            for k, v in d.get("rocm", {}).items():
                row(k, v)
            section("System DB")
            sdb = d.get("system_db", {})
            row("path", sdb.get("path", ""))
            for sub in ("solver_db", "find_db", "kernel_db"):
                for fname, sz in sdb.get(sub, {}).items():
                    row(sub.replace("_", " "), f"{fname} &nbsp; {sz}")
            section("User DB (~/.miopen/db)")
            udb = d.get("user_db", {})
            row("path", udb.get("path", ""))
            for fname, finfo in udb.get("files", {}).items():
                row(fname, finfo)
            return ts + f"<table>{''.join(rows)}</table>"

        with gr.Accordion('ROCm: Advanced Config', open=False, elem_id='rocm_config'):
            gr.HTML("""<style>
                #rocm_btn_apply button { background: var(--button-primary-background-fill) !important; color: var(--button-primary-text-color) !important; border-color: var(--button-primary-border-color) !important; }
                #rocm_btn_apply button:hover { background: var(--button-primary-background-fill-hover) !important; border-color: var(--button-primary-border-color) !important; }
                #rocm_btn_delete button { color: var(--color-error) !important; border-color: var(--color-error) !important; background: transparent !important; }
                #rocm_btn_delete button:hover { background: color-mix(in srgb, var(--color-error) 12%, transparent) !important; }

                /* Strip all block/group backgrounds and borders inside rocm_config */
                #rocm_config .block,
                #rocm_config .form,
                #rocm_config .gradio-group {
                    background: transparent !important;
                    box-shadow: none !important;
                    border: none !important;
                    padding: 0 !important;
                    margin: 0 !important;
                    gap: 0 !important;
                }
                #rocm_config fieldset { border: none !important; padding: 0 !important; margin: 0 !important; box-shadow: none !important; background: transparent !important; }
            </style>""")
            with gr.Row():
                gr.HTML("<p>Advanced configuration for ROCm users.</p><br><p>Set Your database and solver selections based on GPU profile or individually.</p><br><p>Enable cuDNN in Backend Settings to activate MIOpen.</p>")
            with gr.Row():
                btn_info   = gr.Button("Refresh Info", variant="primary", elem_id="rocm_btn_info")
                btn_apply  = gr.Button("Apply", elem_id="rocm_btn_apply")
                btn_reset  = gr.Button("Defaults")
                btn_clear  = gr.Button("Clear Runtime")
                btn_delete = gr.Button("Delete", elem_id="rocm_btn_delete")
            with gr.Row():
                btn_rdna2 = gr.Button("RDNA2 (RX 6000)", elem_id="rocm_btn_rdna2")
                btn_rdna3 = gr.Button("RDNA3 (RX 7000)", elem_id="rocm_btn_rdna3")
                btn_rdna4 = gr.Button("RDNA4 (RX 9000)", elem_id="rocm_btn_rdna4")
            style_out = gr.HTML("")
            info_out = gr.HTML(value=_info_html, elem_id="rocm_info_table")
            
            # General vars (dropdowns, textboxes, checkboxes)
            with gr.Group():
                gr.HTML("<h3>MIOpen Settings</h3><hr>")
                for name, meta in rocm_vars.GENERAL_VARS.items():
                    comp = _make_component(name, meta, config)
                    var_names.append(name)
                    components.append(comp)

            # Solver groups (all checkboxes, grouped by section)
            for group_name, varlist in rocm_vars.SOLVER_GROUPS:
                with gr.Group():
                    gr.HTML(f"<h3>{group_name}</h3><hr>")
                    for name in varlist:
                        meta = rocm_vars.ROCM_ENV_VARS[name]
                        comp = _make_component(name, meta, config)
                        var_names.append(name)
                        components.append(comp)
            gr.HTML("<br><center><div style='margin:0 Auto'><a href='https://rocm.docs.amd.com/projects/MIOpen/en/develop/reference/env_variables.html' target='_blank'>&#128196; MIOpen Environment Variables Reference</a></div></center><br>")
           
        def _autosave_dropdown(name, value):
            meta = rocm_vars.ROCM_ENV_VARS[name]
            if meta["widget"] == "dropdown":
                stored = rocm_mgr._dropdown_stored(str(value), meta["options"])
                config = rocm_mgr.load_config()
                config[name] = stored
                rocm_mgr.save_config(config)
                rocm_mgr.apply_env(config)

        for name, comp in zip(var_names, components):
            meta = rocm_vars.ROCM_ENV_VARS[name]
            if meta["widget"] == "dropdown":
                comp.change(fn=lambda v, n=name: _autosave_dropdown(n, v), inputs=[comp], outputs=[])

        def apply_fn(*values):
            rocm_mgr.apply_all(var_names, list(values))
            saved = rocm_mgr.load_config()
            result = [gr.update(value="")]
            for name in var_names:
                meta = rocm_vars.ROCM_ENV_VARS[name]
                val = saved.get(name, meta["default"])
                if meta["widget"] == "checkbox":
                    result.append(gr.update(value=(val == "1")))
                elif meta["widget"] == "dropdown":
                    result.append(gr.update(value=rocm_mgr._dropdown_display(val, meta["options"])))
                else:
                    result.append(gr.update(value=rocm_mgr._expand_venv(val)))
            return result

        def _build_style(unavailable):
            if not unavailable:
                return ""
            rules = " ".join(
                f"#rocm_var_{v.lower()} label {{ text-decoration: line-through; opacity: 0.5; }}"
                for v in unavailable
            )
            return f"<style>{rules}</style>"

        def reset_fn():
            rocm_mgr.reset_defaults()
            updated = rocm_mgr.load_config()
            result = [gr.update(value="")]
            for name in var_names:
                meta = rocm_vars.ROCM_ENV_VARS[name]
                val = updated.get(name, meta["default"])
                if meta["widget"] == "checkbox":
                    result.append(gr.update(value=(val == "1")))
                elif meta["widget"] == "radio":
                    result.append(gr.update(value=rocm_mgr._dropdown_display(val, meta["options"])))
                else:
                    result.append(gr.update(value=rocm_mgr._expand_venv(val)))
            return result

        def clear_fn():
            rocm_mgr.clear_env()
            result = [gr.update(value="")]
            for name in var_names:
                meta = rocm_vars.ROCM_ENV_VARS[name]
                if meta["widget"] == "checkbox":
                    result.append(gr.update(value=False))
                elif meta["widget"] == "radio":
                    choices = rocm_mgr._dropdown_choices(meta["options"])
                    result.append(gr.update(value=choices[0] if choices else None))
                else:
                    result.append(gr.update(value=""))
            return result

        def delete_fn():
            rocm_mgr.delete_config()
            result = [gr.update(value="")]
            for name in var_names:
                meta = rocm_vars.ROCM_ENV_VARS[name]
                if meta["widget"] == "checkbox":
                    result.append(gr.update(value=False))
                elif meta["widget"] == "radio":
                    choices = rocm_mgr._dropdown_choices(meta["options"])
                    result.append(gr.update(value=choices[0] if choices else None))
                else:
                    result.append(gr.update(value=""))
            return result

        def profile_fn(arch):
            from scripts.rocm import rocm_profiles  # pylint: disable=no-name-in-module
            rocm_mgr.apply_profile(arch)
            updated = rocm_mgr.load_config()
            unavailable = rocm_profiles.UNAVAILABLE.get(arch, set())
            result = [gr.update(value=_build_style(unavailable))]
            for pname in var_names:
                meta = rocm_vars.ROCM_ENV_VARS[pname]
                val = updated.get(pname, meta["default"])
                if meta["widget"] == "checkbox":
                    result.append(gr.update(value=(val == "1")))
                elif meta["widget"] == "radio":
                    result.append(gr.update(value=rocm_mgr._dropdown_display(val, meta["options"])))
                else:
                    result.append(gr.update(value=rocm_mgr._expand_venv(val)))
            return result

        btn_info.click(fn=_info_html, inputs=[], outputs=[info_out])
        btn_apply.click(fn=apply_fn, inputs=components, outputs=[style_out] + components)
        btn_reset.click(fn=reset_fn, inputs=[], outputs=[style_out] + components)
        btn_clear.click(fn=clear_fn, inputs=[], outputs=[style_out] + components)
        btn_delete.click(fn=delete_fn, inputs=[], outputs=[style_out] + components)
        btn_rdna2.click(fn=lambda: profile_fn("RDNA2"), inputs=[], outputs=[style_out] + components)
        btn_rdna3.click(fn=lambda: profile_fn("RDNA3"), inputs=[], outputs=[style_out] + components)
        btn_rdna4.click(fn=lambda: profile_fn("RDNA4"), inputs=[], outputs=[style_out] + components)

        return components
