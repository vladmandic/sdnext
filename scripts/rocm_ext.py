import gradio as gr
import installer
from modules import scripts_manager, shared

# rocm_mgr exposes package-internal helpers (prefixed _) that are intentionally called here
# pylint: disable=protected-access


class ROCmScript(scripts_manager.Script):
    def title(self):
        return "ROCm: Advanced Config"

    def show(self, _is_img2img):
        if shared.cmd_opts.use_rocm or installer.torch_info.get('type') == 'rocm':
            return scripts_manager.AlwaysVisible  # script should be visible only if rocm is detected or forced
        return False

    def ui(self, _is_img2img):
        if not shared.cmd_opts.use_rocm and not installer.torch_info.get('type') == 'rocm':  # skip ui creation if not rocm
            return []

        from scripts.rocm import rocm_mgr, rocm_vars  # pylint: disable=no-name-in-module

        config = rocm_mgr.load_config()
        var_names = []
        components = []

        def _make_component(name, meta, cfg):
            val = cfg.get(name, meta["default"])
            widget = meta["widget"]
            if widget == "checkbox":
                dtype_tag = meta.get("dtype")
                label = f"[{dtype_tag}] {meta['desc']}" if dtype_tag else meta["desc"]
                return gr.Checkbox(label=label, value=(val == "1"), elem_id=f"rocm_var_{name.lower()}")
            if widget == "dropdown":
                choices = rocm_mgr._dropdown_choices(meta["options"])
                display = rocm_mgr._dropdown_display(val, meta["options"])
                return gr.Dropdown(label=meta["desc"], choices=choices, value=display, elem_id=f"rocm_var_{name.lower()}")
            return gr.Textbox(label=meta["desc"], value=rocm_mgr._expand_venv(val), lines=1)

        def _info_html():
            d = rocm_mgr.info()
            rows = []
            def section(title):
                rows.append(f"<tr><th colspan='2' style='padding-top:6px;text-align:left;color:var(--sd-main-accent-color)'>{title}</th></tr>")
            def row(k, v):
                rows.append(f"<tr><td style='color:var(--sd-muted-color);width:38%;padding:2px 8px;border-bottom:1px solid var(--sd-panel-border-color)'>{k}</td><td style='color:var(--sd-label-color);padding:2px 8px;border-bottom:1px solid var(--sd-panel-border-color)'>{v}</td></tr>")
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
            return f"<table style='width:100%;border-collapse:collapse'>{''.join(rows)}</table>"

        with gr.Accordion('ROCm: Advanced Config', open=False, elem_id='rocm_config'):
            with gr.Row():
                gr.HTML("<p>Advanced configuration for ROCm users.</p><br><p>Set your database and solver selections based on GPU profile or individually.</p><br><p>Enable cuDNN in Backend Settings to activate MIOpen.</p>")
            with gr.Row():
                btn_info   = gr.Button("Refresh Info",   variant="primary", elem_id="rocm_btn_info",   size="sm")
                btn_apply  = gr.Button("Apply",          variant="primary", elem_id="rocm_btn_apply",  size="sm")
                btn_reset  = gr.Button("Defaults",       elem_id="rocm_btn_reset",  size="sm")
                btn_clear  = gr.Button("Clear Run Vars", elem_id="rocm_btn_clear",  size="sm")
                btn_delete = gr.Button("Delete UserDb",  variant="stop",    elem_id="rocm_btn_delete", size="sm")
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

        def _autosave_field(name, value):
            meta = rocm_vars.ROCM_ENV_VARS[name]
            stored = rocm_mgr._dropdown_stored(str(value), meta["options"])
            cfg = rocm_mgr.load_config()
            cfg[name] = stored
            rocm_mgr.save_config(cfg)
            rocm_mgr.apply_env(cfg)

        for name, comp in zip(var_names, components):
            meta = rocm_vars.ROCM_ENV_VARS[name]
            if meta["widget"] == "dropdown":
                comp.change(fn=lambda v, n=name: _autosave_field(n, v), inputs=[comp], outputs=[], show_progress='hidden')

        def apply_fn(*values):
            rocm_mgr.apply_all(var_names, list(values))
            saved = rocm_mgr.load_config()
            result = [gr.update(value="")]
            for name in var_names:
                meta = rocm_vars.ROCM_ENV_VARS[name]
                val = saved.get(name, meta["default"])
                if meta["widget"] == "checkbox":
                    result.append(gr.update(value=val == "1"))
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
                    result.append(gr.update(value=val == "1"))
                elif meta["widget"] == "dropdown":
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
                elif meta["widget"] == "dropdown":
                    result.append(gr.update(value=rocm_mgr._dropdown_display(meta["default"], meta["options"])))
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
                elif meta["widget"] == "dropdown":
                    result.append(gr.update(value=rocm_mgr._dropdown_display(meta["default"], meta["options"])))
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
                    result.append(gr.update(value=val == "1"))
                elif meta["widget"] == "dropdown":
                    result.append(gr.update(value=rocm_mgr._dropdown_display(val, meta["options"])))
                else:
                    result.append(gr.update(value=rocm_mgr._expand_venv(val)))
            return result

        btn_info.click(fn=_info_html, inputs=[], outputs=[info_out], show_progress='hidden')
        btn_apply.click(fn=apply_fn, inputs=components, outputs=[style_out] + components, show_progress='hidden')
        btn_reset.click(fn=reset_fn, inputs=[], outputs=[style_out] + components, show_progress='hidden')
        btn_clear.click(fn=clear_fn, inputs=[], outputs=[style_out] + components, show_progress='hidden')
        btn_delete.click(fn=delete_fn, inputs=[], outputs=[style_out] + components, show_progress='hidden')
        btn_rdna2.click(fn=lambda: profile_fn("RDNA2"), inputs=[], outputs=[style_out] + components, show_progress='hidden')
        btn_rdna3.click(fn=lambda: profile_fn("RDNA3"), inputs=[], outputs=[style_out] + components, show_progress='hidden')
        btn_rdna4.click(fn=lambda: profile_fn("RDNA4"), inputs=[], outputs=[style_out] + components, show_progress='hidden')

        return components
