import gradio as gr
import installer
from modules import scripts_manager, shared

# rocm_mgr exposes package-internal helpers (prefixed _) that are intentionally called here
# pylint: disable=protected-access

class ROCmScript(scripts_manager.Script):
    def title(self):
        return "Windows ROCm: Advanced Config"

    def show(self, _is_img2img):
        if shared.cmd_opts.use_rocm or installer.torch_info.get('type') == 'rocm':
            return scripts_manager.AlwaysVisible  # script should be visible only if rocm is detected or forced
        return False

    def ui(self, _is_img2img):
        if not shared.cmd_opts.use_rocm and not installer.torch_info.get('type') == 'rocm':  # skip ui creation if not rocm
            return []

        from scripts.rocm import rocm_mgr, rocm_vars, rocm_profiles  # pylint: disable=no-name-in-module

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
            section("User cache (~/.miopen/cache)")
            ucache = d.get("user_cache", {})
            row("path", ucache.get("path", ""))
            for fname, sz in ucache.get("files", {}).items():
                row(fname, sz)
            return f"<table style='width:100%;border-collapse:collapse'>{''.join(rows)}</table>"

        def _build_style(unavailable, hipblaslt_disabled=False):
            rules = []
            for v in (unavailable or []):
                rules.append(f"#rocm_var_{v.lower()} label {{ text-decoration: line-through; opacity: 0.5; }}")
            if hipblaslt_disabled:
                for v in rocm_vars.HIPBLASLT_VARS:
                    rules.append(f"#rocm_var_{v.lower()} {{ opacity: 0.45; pointer-events: none; }}")
            return f"<style>{' '.join(rules)}</style>" if rules else ""

        with gr.Accordion('ROCm: Advanced Config', open=False, elem_id='rocm_config'):
            with gr.Row():
                gr.HTML("<p><u>Advanced configuration for ROCm users.</u></p><br><p>This script aims to take the guesswork out of configuring MIOpen and rocBLAS on Windows ROCm, but also to expose the functioning switches of MIOpen for advanced configurations.</p><br><p>For best performance ensure that cuDNN and PyTorch tunable ops are set to <b><i>default</i></b> in Backend Settings.</p><br><p>This script was written with the intent to support ROCm Windows users, it should however, function identically for Linux users.</p><br>")
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
            _init_gemm = config.get("MIOPEN_GEMM_ENFORCE_BACKEND", "1")
            _init_arch = config.get(rocm_mgr._ARCH_KEY, "")
            _init_unavailable = rocm_profiles.UNAVAILABLE.get(_init_arch, set()) if _init_arch else set()
            style_out = gr.HTML(_build_style(_init_unavailable, _init_gemm == "1"))
            info_out = gr.HTML(value=_info_html, elem_id="rocm_info_table")

            # General vars (dropdowns, textboxes, checkboxes)
            with gr.Group():
                gr.HTML("<br><h3>MIOpen Settings</h3><hr>")
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
            if meta["widget"] == "dropdown" and name != "MIOPEN_GEMM_ENFORCE_BACKEND":
                comp.change(fn=lambda v, n=name: _autosave_field(n, v), inputs=[comp], outputs=[], show_progress='hidden')

        _GEMM_COMPANIONS = {
            "PYTORCH_ROCM_USE_ROCBLAS":           {"1": "1", "5": "0"},
            "PYTORCH_HIPBLASLT_DISABLE":           {"1": "1", "5": "0"},
            "ROCBLAS_USE_HIPBLASLT":               {"1": "0", "5": "1"},
            "PYTORCH_TUNABLEOP_HIPBLASLT_ENABLED": {"1": "0", "5": "1"},
        }

        def gemm_changed(gemm_display_val):
            stored = rocm_mgr._dropdown_stored(str(gemm_display_val), rocm_vars.ROCM_ENV_VARS["MIOPEN_GEMM_ENFORCE_BACKEND"]["options"])
            cfg = rocm_mgr.load_config().copy()
            cfg["MIOPEN_GEMM_ENFORCE_BACKEND"] = stored
            for var, vals in _GEMM_COMPANIONS.items():
                cfg[var] = vals.get(stored, cfg.get(var, ""))
            rocm_mgr.save_config(cfg)
            rocm_mgr.apply_env(cfg)
            arch = cfg.get(rocm_mgr._ARCH_KEY, "")
            unavailable = rocm_profiles.UNAVAILABLE.get(arch, set())
            result = [gr.update(value=_build_style(unavailable, stored == "1"))]
            for pname in var_names:
                if pname in _GEMM_COMPANIONS:
                    meta = rocm_vars.ROCM_ENV_VARS[pname]
                    val = _GEMM_COMPANIONS[pname].get(stored, cfg.get(pname, ""))
                    result.append(gr.update(value=rocm_mgr._dropdown_display(val, meta["options"])))
                else:
                    result.append(gr.update())
            return result

        gemm_comp = components[var_names.index("MIOPEN_GEMM_ENFORCE_BACKEND")]
        gemm_comp.change(fn=gemm_changed, inputs=[gemm_comp], outputs=[style_out] + components, show_progress='hidden')

        def apply_fn(*values):
            rocm_mgr.apply_all(var_names, list(values))
            saved = rocm_mgr.load_config()
            arch = saved.get(rocm_mgr._ARCH_KEY, "")
            unavailable = rocm_profiles.UNAVAILABLE.get(arch, set())
            gemm_val = saved.get("MIOPEN_GEMM_ENFORCE_BACKEND", "1")
            result = [gr.update(value=_build_style(unavailable, gemm_val == "1"))]
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

        def reset_fn():
            rocm_mgr.reset_defaults()
            updated = rocm_mgr.load_config()
            arch = updated.get(rocm_mgr._ARCH_KEY, "")
            unavailable = rocm_profiles.UNAVAILABLE.get(arch, set())
            gemm_val = updated.get("MIOPEN_GEMM_ENFORCE_BACKEND", "1")
            result = [gr.update(value=_build_style(unavailable, gemm_val == "1"))]
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
            cfg = rocm_mgr.load_config()
            gemm_val = cfg.get("MIOPEN_GEMM_ENFORCE_BACKEND", "1")
            result = [gr.update(value=_build_style(None, gemm_val == "1"))]
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
            gemm_default = rocm_vars.ROCM_ENV_VARS.get("MIOPEN_GEMM_ENFORCE_BACKEND", {}).get("default", "1")
            result = [gr.update(value=_build_style(None, gemm_default == "1"))]
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
            rocm_mgr.apply_profile(arch)
            updated = rocm_mgr.load_config()
            unavailable = rocm_profiles.UNAVAILABLE.get(arch, set())
            gemm_val = updated.get("MIOPEN_GEMM_ENFORCE_BACKEND", "1")
            result = [gr.update(value=_build_style(unavailable, gemm_val == "1"))]
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
