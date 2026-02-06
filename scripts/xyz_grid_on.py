# xyz grid that shows up as alwayson script
import os
import time
import random
from collections import namedtuple
from copy import copy
from itertools import permutations
from PIL import Image
import numpy as np
import gradio as gr
from scripts.xyz.xyz_grid_shared import str_permutations, list_to_csv_string, restore_comma, re_range, re_plain_comma # pylint: disable=no-name-in-module
from scripts.xyz.xyz_grid_classes import axis_options, AxisOption, SharedSettingsStackHelper # pylint: disable=no-name-in-module
from scripts.xyz.xyz_grid_draw import draw_xyz_grid # pylint: disable=no-name-in-module
from modules import shared, errors, scripts_manager, images, processing
from modules.ui_components import ToolButton
from modules.ui_sections import create_video_inputs
import modules.ui_symbols as symbols


active = False
xyz_results_cache = None
debug = shared.log.trace if os.environ.get('SD_XYZ_DEBUG', None) is not None else lambda *args, **kwargs: None


class Script(scripts_manager.Script):
    current_axis_options = []

    def show(self, is_img2img):
        return scripts_manager.AlwaysVisible

    def title(self):
        return "XYZ Grid"

    def ui(self, is_img2img):
        self.current_axis_options = [x for x in axis_options if type(x) == AxisOption or x.is_img2img == is_img2img]

        with gr.Accordion('XYZ Grid', open = False, elem_id='xyz_grid'):
            with gr.Row():
                enabled = gr.Checkbox(label = 'Enabled', value = False)

            with gr.Row():
                with gr.Column():
                    with gr.Row(variant='compact'):
                        x_type = gr.Dropdown(label="X type", container=True, choices=[x.label for x in self.current_axis_options], value=self.current_axis_options[0].label, type="index", elem_id=self.elem_id("x_type"))
                        x_values = gr.Textbox(label="X values", container=True, lines=1, elem_id=self.elem_id("x_values"))
                        x_values_dropdown = gr.Dropdown(label="X values", container=True, visible=False, multiselect=True, interactive=True)
                        fill_x_button = ToolButton(value=symbols.fill, elem_id="xyz_gridon_x_list", visible=False)
                    with gr.Row(variant='compact'):
                        y_type = gr.Dropdown(label="Y type", container=True, choices=[x.label for x in self.current_axis_options], value=self.current_axis_options[0].label, type="index", elem_id=self.elem_id("y_type"))
                        y_values = gr.Textbox(label="Y values", container=True, lines=1, elem_id=self.elem_id("y_values"))
                        y_values_dropdown = gr.Dropdown(label="Y values", container=True, visible=False, multiselect=True, interactive=True)
                        fill_y_button = ToolButton(value=symbols.fill, elem_id="xyz_gridon_y_list", visible=False)
                    with gr.Row(variant='compact'):
                        z_type = gr.Dropdown(label="Z type", container=True, choices=[x.label for x in self.current_axis_options], value=self.current_axis_options[0].label, type="index", elem_id=self.elem_id("z_type"))
                        z_values = gr.Textbox(label="Z values", container=True, lines=1, elem_id=self.elem_id("z_values"))
                        z_values_dropdown = gr.Dropdown(label="Z values", container=True, visible=False, multiselect=True, interactive=True)
                        fill_z_button = ToolButton(value=symbols.fill, elem_id="xyz_gridon_z_list", visible=False)

            with gr.Row():
                with gr.Column():
                    draw_legend = gr.Checkbox(label='Draw legend', value=True, elem_id=self.elem_id("draw_legend"), container=False)
                    csv_mode = gr.Checkbox(label='Use text inputs', value=False, elem_id=self.elem_id("csv_mode"), container=False)
                    no_fixed_seeds = gr.Checkbox(label='Use random seeds', value=False, elem_id=self.elem_id("no_fixed_seeds"), container=False)
                    include_time = gr.Checkbox(label='Add time info', value=False, elem_id=self.elem_id("include_time"), container=False)
                    include_text = gr.Checkbox(label='Add text info', value=False, elem_id=self.elem_id("include_text"), container=False)
                with gr.Column():
                    include_grid = gr.Checkbox(label='Include main grid', value=True, elem_id=self.elem_id("no_xyz_grid"), container=False)
                    include_subgrids = gr.Checkbox(label='Include sub grids', value=False, elem_id=self.elem_id("include_sub_grids"), container=False)
                    include_images = gr.Checkbox(label='Include images', value=False, elem_id=self.elem_id("include_lone_images"), container=False)
                    create_video = gr.Checkbox(label='Create video', value=False, elem_id=self.elem_id("xyz_create_video"), container=False)

            with gr.Row(visible=False) as ui_video:
                video_type, video_duration, video_loop, video_pad, video_interpolate = create_video_inputs(tab='img2img' if is_img2img else 'txt2img')
                create_video.change(fn=lambda x: gr.update(visible=x), inputs=[create_video], outputs=[ui_video])

            with gr.Row():
                margin_size = gr.Slider(label="Grid margins", minimum=0, maximum=500, value=0, step=2, elem_id=self.elem_id("margin_size"))

            with gr.Row():
                swap_xy_axes_button = gr.Button(value="Swap X/Y", elem_id="xy_grid_swap_axes_button", variant="secondary")
                swap_yz_axes_button = gr.Button(value="Swap Y/Z", elem_id="yz_grid_swap_axes_button", variant="secondary")
                swap_xz_axes_button = gr.Button(value="Swap X/Z", elem_id="xz_grid_swap_axes_button", variant="secondary")

        def swap_axes(axis1_type, axis1_values, axis1_values_dropdown, axis2_type, axis2_values, axis2_values_dropdown):
            return self.current_axis_options[axis2_type].label, axis2_values, axis2_values_dropdown, self.current_axis_options[axis1_type].label, axis1_values, axis1_values_dropdown

        xy_swap_args = [x_type, x_values, x_values_dropdown, y_type, y_values, y_values_dropdown]
        swap_xy_axes_button.click(swap_axes, inputs=xy_swap_args, outputs=xy_swap_args)
        yz_swap_args = [y_type, y_values, y_values_dropdown, z_type, z_values, z_values_dropdown]
        swap_yz_axes_button.click(swap_axes, inputs=yz_swap_args, outputs=yz_swap_args)
        xz_swap_args = [x_type, x_values, x_values_dropdown, z_type, z_values, z_values_dropdown]
        swap_xz_axes_button.click(swap_axes, inputs=xz_swap_args, outputs=xz_swap_args)

        def fill(axis_type, csv_mode):
            axis = self.current_axis_options[axis_type]
            if axis.choices:
                if csv_mode:
                    return list_to_csv_string(axis.choices()), gr.update()
                else:
                    return gr.update(), axis.choices()
            else:
                return gr.update(), gr.update()

        fill_x_button.click(fn=fill, inputs=[x_type, csv_mode], outputs=[x_values, x_values_dropdown])
        fill_y_button.click(fn=fill, inputs=[y_type, csv_mode], outputs=[y_values, y_values_dropdown])
        fill_z_button.click(fn=fill, inputs=[z_type, csv_mode], outputs=[z_values, z_values_dropdown])

        def select_axis(axis_type, axis_values, axis_values_dropdown, csv_mode):
            choices = self.current_axis_options[axis_type].choices
            has_choices = choices is not None
            current_values = axis_values
            current_dropdown_values = axis_values_dropdown
            if has_choices:
                choices = choices()
                if csv_mode:
                    current_dropdown_values = list(filter(lambda x: x in choices, current_dropdown_values))
                    current_values = list_to_csv_string(current_dropdown_values)
                else:
                    current_dropdown_values = [restore_comma(x.strip()) for x in re_plain_comma.split(axis_values) if x]
                    current_dropdown_values = list(filter(lambda x: x in choices, current_dropdown_values))

            return (gr.Button.update(visible=has_choices), gr.Textbox.update(visible=not has_choices or csv_mode, value=current_values),
                    gr.update(choices=choices if has_choices else None, visible=has_choices and not csv_mode, value=current_dropdown_values))

        x_type.change(fn=select_axis, inputs=[x_type, x_values, x_values_dropdown, csv_mode], outputs=[fill_x_button, x_values, x_values_dropdown])
        y_type.change(fn=select_axis, inputs=[y_type, y_values, y_values_dropdown, csv_mode], outputs=[fill_y_button, y_values, y_values_dropdown])
        z_type.change(fn=select_axis, inputs=[z_type, z_values, z_values_dropdown, csv_mode], outputs=[fill_z_button, z_values, z_values_dropdown])

        def change_choice_mode(csv_mode, x_type, x_values, x_values_dropdown, y_type, y_values, y_values_dropdown, z_type, z_values, z_values_dropdown):
            _fill_x_button, _x_values, _x_values_dropdown = select_axis(x_type, x_values, x_values_dropdown, csv_mode)
            _fill_y_button, _y_values, _y_values_dropdown = select_axis(y_type, y_values, y_values_dropdown, csv_mode)
            _fill_z_button, _z_values, _z_values_dropdown = select_axis(z_type, z_values, z_values_dropdown, csv_mode)
            return _fill_x_button, _x_values, _x_values_dropdown, _fill_y_button, _y_values, _y_values_dropdown, _fill_z_button, _z_values, _z_values_dropdown

        csv_mode.change(fn=change_choice_mode, inputs=[csv_mode, x_type, x_values, x_values_dropdown, y_type, y_values, y_values_dropdown, z_type, z_values, z_values_dropdown], outputs=[fill_x_button, x_values, x_values_dropdown, fill_y_button, y_values, y_values_dropdown, fill_z_button, z_values, z_values_dropdown])

        def get_dropdown_update_from_params(axis,params):
            val_key = f"{axis} Values"
            vals = params.get(val_key,"")
            valslist = [restore_comma(x.strip()) for x in re_plain_comma.split(vals) if x]
            return gr.update(value = valslist)

        self.infotext_fields = (
            (enabled, "XYZ Grid Enabled"),
            (x_type, "X Type"),
            (x_values, "X Values"),
            (x_values_dropdown, lambda params:get_dropdown_update_from_params("X",params)),
            (y_type, "Y Type"),
            (y_values, "Y Values"),
            (y_values_dropdown, lambda params:get_dropdown_update_from_params("Y",params)),
            (z_type, "Z Type"),
            (z_values, "Z Values"),
            (z_values_dropdown, lambda params:get_dropdown_update_from_params("Z",params)),
        )

        return [
            enabled,
            x_type, x_values, x_values_dropdown,
            y_type, y_values, y_values_dropdown,
            z_type, z_values, z_values_dropdown,
            csv_mode, draw_legend, no_fixed_seeds,
            include_grid, include_subgrids, include_images,
            include_time, include_text, margin_size,
            create_video, video_type, video_duration, video_loop, video_pad, video_interpolate,
        ]

    def process(self, p,
                enabled,
                x_type, x_values, x_values_dropdown,
                y_type, y_values, y_values_dropdown,
                z_type, z_values, z_values_dropdown,
                csv_mode, draw_legend, no_fixed_seeds,
                include_grid, include_subgrids, include_images,
                include_time, include_text, margin_size,
                create_video, video_type, video_duration, video_loop, video_pad, video_interpolate,
               ): # pylint: disable=W0221
        global active, xyz_results_cache # pylint: disable=W0603
        xyz_results_cache = None
        if not enabled or active:
            return None
        active = True
        if not no_fixed_seeds:
            processing.fix_seed(p)
        if not shared.opts.return_grid:
            p.batch_size = 1
        jobid = shared.state.begin('XYZ Grid')

        def process_axis(opt, vals, vals_dropdown):
            if opt.label == 'Nothing':
                return [0]
            if opt.choices is not None and not csv_mode:
                valslist = vals_dropdown
            else:
                valslist = [restore_comma(x.strip()) for x in re_plain_comma.split(vals) if x]
            if opt.type == int:
                valslist_ext = []
                for val in valslist:
                    try:
                        m = re_range.fullmatch(val)
                        if m is not None:
                            start_val = int(m.group(1)) if m.group(1) is not None else val
                            end_val = int(m.group(2)) if m.group(2) is not None else val
                            num = int(m.group(3)) if m.group(3) is not None else int(end_val-start_val)
                            valslist_ext += [int(x) for x in np.linspace(start=start_val, stop=end_val, num=max(2, num)).tolist()]
                            shared.log.debug(f'XYZ grid range: start={start_val} end={end_val} num={max(2, num)} list={valslist}')
                        else:
                            valslist_ext.append(int(val))
                    except Exception as e:
                        shared.log.error(f"XYZ grid: value={val} {e}")
                valslist.clear()
                valslist = [x for x in valslist_ext if x not in valslist]
            elif opt.type == float:
                valslist_ext = []
                for val in valslist:
                    try:
                        m = re_range.fullmatch(val)
                        if m is not None:
                            start_val = float(m.group(1)) if m.group(1) is not None else val
                            end_val = float(m.group(2)) if m.group(2) is not None else val
                            num = int(m.group(3)) if m.group(3) is not None else int(end_val-start_val)
                            valslist_ext += [round(float(x), 2) for x in np.linspace(start=start_val, stop=end_val, num=max(2, num)).tolist()]
                            shared.log.debug(f'XYZ grid range: start={start_val} end={end_val} num={max(2, num)} list={valslist}')
                        else:
                            valslist_ext.append(float(val))
                    except Exception as e:
                        shared.log.error(f"XYZ grid: value={val} {e}")
                valslist.clear()
                valslist = [x for x in valslist_ext if x not in valslist]
            elif opt.type == str_permutations: # pylint: disable=comparison-with-callable
                valslist = list(permutations(valslist))
            valslist = [opt.type(x) for x in valslist]
            # Confirm options are valid before starting
            if opt.confirm:
                opt.confirm(p, valslist)
            return valslist

        def parse_axis(x_type, x_values, x_values_dropdown):
            x_opt = None
            if isinstance(x_type, str):
                x_opt = [o for o in self.current_axis_options if o.label.lower() == x_type.lower()]
                if len(x_opt) == 0:
                    x_opt = [o for o in self.current_axis_options if x_type.lower() in o.label.lower()]
                if len(x_opt) > 0:
                    x_opt = x_opt[0]
            else:
                x_opt = self.current_axis_options[x_type]
            if x_opt:
                if x_opt.choices is not None and not csv_mode:
                    x_values = list_to_csv_string(x_values_dropdown)
                xs = process_axis(x_opt, x_values, x_values_dropdown)
            else:
                xs = []
            return x_opt, xs

        try:
            x_opt, xs = parse_axis(x_type, x_values, x_values_dropdown)
            y_opt, ys = parse_axis(y_type, y_values, y_values_dropdown)
            z_opt, zs = parse_axis(z_type, z_values, z_values_dropdown)
        except Exception as e:
            shared.log.error(f"XYZ grid: invalid axis values {e}")
            errors.display(e, 'xyz')
            return None

        Image.MAX_IMAGE_PIXELS = None # disable check in Pillow and rely on check below to allow large custom image sizes

        def fix_axis_seeds(axis_opt, axis_list):
            if axis_opt.label in ['[Param] Seed', '[Param] Variation seed']:
                return [int(random.randrange(4294967294)) if val is None or val == '' or val == -1 else val for val in axis_list]
            else:
                return axis_list

        if not no_fixed_seeds:
            xs = fix_axis_seeds(x_opt, xs)
            ys = fix_axis_seeds(y_opt, ys)
            zs = fix_axis_seeds(z_opt, zs)

        total_jobs = len(xs) * len(ys) * len(zs)
        if x_opt.label == 'Steps':
            total_steps = sum(xs) * len(ys) * len(zs)
        elif y_opt.label == 'Steps':
            total_steps = sum(ys) * len(xs) * len(zs)
        elif z_opt.label == 'Steps':
            total_steps = sum(zs) * len(xs) * len(ys)
        else:
            total_steps = p.steps * total_jobs
        if p.enable_hr:
            if x_opt.label == "Hires steps":
                total_steps += sum(xs) * len(ys) * len(zs)
            elif y_opt.label == "Hires steps":
                total_steps += sum(ys) * len(xs) * len(zs)
            elif z_opt.label == "Hires steps":
                total_steps += sum(zs) * len(xs) * len(ys)
            elif p.hr_second_pass_steps:
                total_steps += p.hr_second_pass_steps * total_jobs
            else:
                total_steps *= 2
        if p.detailer_enabled:
            total_steps += p.detailer_steps * total_jobs

        total_steps *= p.n_iter
        total_jobs *= p.n_iter
        shared.state.update('Grid', total_steps, total_jobs)

        image_cell_count = p.n_iter * p.batch_size
        shared.log.info(f"XYZ grid start: images={len(xs)*len(ys)*len(zs)*image_cell_count} grid={len(zs)} shape={len(xs)}x{len(ys)} cells={len(zs)} steps={total_steps} csv={csv_mode} legend={draw_legend} grid={include_grid} subgrid={include_subgrids} images={include_images} time={include_time} text={include_text}")
        AxisInfo = namedtuple('AxisInfo', ['axis', 'values'])
        shared.state.xyz_plot_x = AxisInfo(x_opt, xs)
        shared.state.xyz_plot_y = AxisInfo(y_opt, ys)
        shared.state.xyz_plot_z = AxisInfo(z_opt, zs)
        # If one of the axes is very slow to change between (like SD model checkpoint), then make sure it is in the outer iteration of the nested `for` loop.
        first_axes_processed = 'z'
        second_axes_processed = 'y'
        if x_opt.cost > y_opt.cost and x_opt.cost > z_opt.cost:
            first_axes_processed = 'x'
            if y_opt.cost > z_opt.cost:
                second_axes_processed = 'y'
            else:
                second_axes_processed = 'z'
        elif y_opt.cost > x_opt.cost and y_opt.cost > z_opt.cost:
            first_axes_processed = 'y'
            if x_opt.cost > z_opt.cost:
                second_axes_processed = 'x'
            else:
                second_axes_processed = 'z'
        elif z_opt.cost > x_opt.cost and z_opt.cost > y_opt.cost:
            first_axes_processed = 'z'
            if x_opt.cost > y_opt.cost:
                second_axes_processed = 'x'
            else:
                second_axes_processed = 'y'
        grid_infotext = []

        def cell(x, y, z, ix, iy, iz):
            if shared.state.interrupted:
                shared.log.warning('XYZ grid: Interrupted')
                return processing.Processed(p, [], p.seed, ""), 0
            p.xyz = True
            pc = copy(p)
            pc.override_settings_restore_afterwards = False
            pc.styles = pc.styles[:]
            if no_fixed_seeds:
                pc.seed = -1
                processing.fix_seed(pc)
                pc.all_seeds = None
                pc.all_subseeds = None
            x_opt.apply(pc, x, xs)
            y_opt.apply(pc, y, ys)
            z_opt.apply(pc, z, zs)

            t0 = time.time()
            try:
                processed = processing.process_images(pc)
            except Exception as e:
                shared.log.error(f"XYZ grid: Failed to process image: {e}")
                errors.display(e, 'XYZ grid')
                processed = None

            if ix == 0 and iy == 0: # create subgrid info text
                pc.extra_generation_params = copy(pc.extra_generation_params)
                pc.extra_generation_params['Script'] = self.title()
                pc.extra_generation_params['XYZ Grid Enabled'] = enabled
                if x_opt.label != 'Nothing':
                    pc.extra_generation_params["X Type"] = x_opt.label
                    pc.extra_generation_params["X Values"] = x_values
                    if x_opt.label in ["[Param] Seed", "[Param] Variation seed"] and not no_fixed_seeds:
                        pc.extra_generation_params["Fixed X Values"] = ", ".join([str(x) for x in xs])
                if y_opt.label != 'Nothing':
                    pc.extra_generation_params["Y Type"] = y_opt.label
                    pc.extra_generation_params["Y Values"] = y_values
                    if y_opt.label in ["[Param] Seed", "[Param] Variation seed"] and not no_fixed_seeds:
                        pc.extra_generation_params["Fixed Y Values"] = ", ".join([str(y) for y in ys])
                info = processing.create_infotext(pc, pc.all_prompts, pc.all_seeds, pc.all_subseeds, grid=f'{len(xs)}x{len(ys)}')
                grid_infotext.append(info)
            if ix == 0 and iy == 0 and iz == 0 and len(zs) > 1: # create main grid info text
                pc.extra_generation_params = copy(pc.extra_generation_params)
                if z_opt.label != 'Nothing':
                    pc.extra_generation_params["Z Type"] = z_opt.label
                    pc.extra_generation_params["Z Values"] = z_values
                    if z_opt.label in ["[Param] Seed", "[Param] Variation seed"] and not no_fixed_seeds:
                        pc.extra_generation_params["Fixed Z Values"] = ", ".join([str(z) for z in zs])
                info = processing.create_infotext(pc, pc.all_prompts, pc.all_seeds, pc.all_subseeds, grid=f'{len(zs)}x{len(xs)}x{len(ys)}')
                grid_infotext.insert(0, info)
            t1 = time.time()
            return processed, t1-t0

        with SharedSettingsStackHelper():
            processed: processing.Processed = draw_xyz_grid(
                p,
                xs=xs,
                ys=ys,
                zs=zs,
                x_labels=[x_opt.format_value(p, x_opt, x) for x in xs],
                y_labels=[y_opt.format_value(p, y_opt, y) for y in ys],
                z_labels=[z_opt.format_value(p, z_opt, z) for z in zs],
                cell=cell,
                draw_legend=draw_legend,
                include_lone_images=include_images,
                include_sub_grids=include_subgrids,
                first_axes_processed=first_axes_processed,
                second_axes_processed=second_axes_processed,
                margin_size=margin_size,
                no_grid=not include_grid,
                include_time=include_time,
                include_text=include_text,
            )

        if hasattr(shared.sd_model, 'restore_pipeline') and (shared.sd_model.restore_pipeline is not None):
            shared.sd_model.restore_pipeline()

        if not processed.images:
            active = False
            return processed # something broke, no further handling needed.

        have_grid = 1 if include_grid else 0
        have_subgrids = len(zs) if len(zs) > 1 and include_subgrids else 0
        have_images = processed.images[have_grid+have_subgrids:]
        processed.infotexts[:have_grid+have_subgrids] = grid_infotext[:have_grid+have_subgrids] # update infotexts with grid and subgrid info
        shared.log.debug(f'XYZ grid: grid={have_grid} subgrids={have_subgrids} images={len(have_images)} total={len(processed.images)}')

        if not include_images: # dont need images anymore, drop from list:
            processed.images = processed.images[:have_grid+have_subgrids]
            debug(f'XYZ grid remove images: total={processed.images}')

        if shared.opts.grid_save and not shared.state.interrupted: # auto-save main and sub-grids:
            for g in range(have_grid + have_subgrids):
                adj_g = g-1 if g > 0 else g
                info = processed.infotexts[g]
                prompt = processed.all_prompts[adj_g]
                seed = processed.all_seeds[adj_g]
                debug(f'XYZ grid save grid: i={g+1}')
                images.save_image(processed.images[g], p.outpath_grids, "grid", info=info, extension=shared.opts.grid_format, prompt=prompt, seed=seed, grid=True, p=processed)

        if not include_subgrids and have_subgrids > 0: # done with sub-grids, drop all related information:
            for _sg in range(have_subgrids):
                del processed.images[1]
                del processed.all_prompts[1]
                del processed.all_seeds[1]
                del processed.infotexts[1]
            debug(f'XYZ grid remove subgrids: total={processed.images}')

        if create_video and video_type != 'None' and not shared.state.interrupted:
            images.save_video(p, filename=None, images=have_images, video_type=video_type, duration=video_duration, loop=video_loop, pad=video_pad, interpolate=video_interpolate)

        p.do_not_save_grid = True
        p.do_not_save_samples = True
        p.disable_extra_networks = True
        active = False
        xyz_results_cache = processed
        shared.state.end(jobid)
        return processed


    def process_images(self, p, *args): # pylint: disable=W0221, W0613
        if xyz_results_cache is not None and len(xyz_results_cache.images) > 0:
            p.restore_faces = False
            p.detailer_enabled = False
            p.color_corrections = None
            # p.scripts = None
            return xyz_results_cache
        return None
