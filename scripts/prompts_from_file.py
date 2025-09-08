import copy
import random
import shlex
import gradio as gr
from modules import sd_samplers, errors, scripts_manager
from modules.processing import Processed, process_images
from modules.shared import state, log


def process_string_tag(tag):
    return tag


def process_int_tag(tag):
    return int(tag)


def process_float_tag(tag):
    return float(tag)


def process_boolean_tag(tag):
    return True if (tag == "true") else False


prompt_tags = {
    "sd_model": None,
    "outpath_samples": process_string_tag,
    "outpath_grids": process_string_tag,
    "prompt_for_display": process_string_tag,
    "prompt": process_string_tag,
    "negative_prompt": process_string_tag,
    "styles": process_string_tag,
    "seed": process_int_tag,
    "subseed_strength": process_float_tag,
    "subseed": process_int_tag,
    "seed_resize_from_h": process_int_tag,
    "seed_resize_from_w": process_int_tag,
    "sampler_index": process_int_tag,
    "sampler_name": process_string_tag,
    "batch_size": process_int_tag,
    "n_iter": process_int_tag,
    "steps": process_int_tag,
    "cfg_scale": process_float_tag,
    "width": process_int_tag,
    "height": process_int_tag,
    "detailer": process_boolean_tag,
    "tiling": process_boolean_tag,
    "do_not_save_samples": process_boolean_tag,
    "do_not_save_grid": process_boolean_tag
}


def cmdargs(line):
    args = shlex.split(line)
    pos = 0
    res = {}
    while pos < len(args):
        arg = args[pos]
        assert arg.startswith("--"), f'must start with "--": {arg}'
        assert pos+1 < len(args), f'missing argument for command line option {arg}'
        tag = arg[2:]
        if tag == "prompt" or tag == "negative_prompt":
            pos += 1
            prompt = args[pos]
            pos += 1
            while pos < len(args) and not args[pos].startswith("--"):
                prompt += " "
                prompt += args[pos]
                pos += 1
            res[tag] = prompt
            continue

        func = prompt_tags.get(tag, None)
        assert func, f'unknown commandline option: {arg}'
        val = args[pos+1]
        if tag == "sampler_name":
            val = sd_samplers.samplers_map.get(val.lower(), None)
        res[tag] = func(val)
        pos += 2
    return res


def load_prompt_file(file):
    if file is None:
        return None, gr.update(), gr.update(lines=7)
    else:
        try:
            lines = [x.strip() for x in file.decode('utf8', errors='ignore').split("\n")]
        except Exception as e:
            log.error(f"Prompt file: {e}")
            lines = ''
        return None, "\n".join(lines), gr.update(lines=7)


class Script(scripts_manager.Script):
    def title(self):
        return "Prompts from file"

    def ui(self, is_img2img):
        with gr.Row():
            gr.HTML('<span">&nbsp Prompt from file</span><br>')
        with gr.Row():
            checkbox_iterate = gr.Checkbox(label="Iterate seed per line", value=False, elem_id=self.elem_id("checkbox_iterate"))
            checkbox_iterate_batch = gr.Checkbox(label="Use same seed", value=False, elem_id=self.elem_id("checkbox_iterate_batch"))
        prompt_txt = gr.Textbox(label="Prompts", lines=2, elem_id=self.elem_id("prompt_txt"), value='')
        file = gr.File(label="Upload prompts", type='binary', elem_id=self.elem_id("file"))
        file.change(fn=load_prompt_file, inputs=[file], outputs=[file, prompt_txt, prompt_txt], show_progress=False)
        prompt_txt.change(lambda tb: gr.update(lines=7) if ("\n" in tb) else gr.update(lines=2), inputs=[prompt_txt], outputs=[prompt_txt], show_progress=False)
        return [checkbox_iterate, checkbox_iterate_batch, prompt_txt]

    def run(self, p, checkbox_iterate, checkbox_iterate_batch, prompt_txt: str): # pylint: disable=arguments-differ
        lines = [x.strip() for x in prompt_txt.splitlines()]
        lines = [x for x in lines if len(x) > 0]
        job_count = 0
        jobs = []
        for line in lines:
            if "--" in line:
                try:
                    args = cmdargs(line)
                except Exception as e:
                    errors.display(e, f'parsing prompts: {line}')
                    args = {"prompt": line}
            else:
                args = {"prompt": line}
            job_count += args.get("n_iter", p.n_iter)
            jobs.append(args)

        log.info(f"Prompts-from-file: lines={len(lines)} jobs={job_count}")
        if (checkbox_iterate or checkbox_iterate_batch) and p.seed == -1:
            p.seed = int(random.randrange(4294967294))
        state.job_count = job_count
        images = []
        all_prompts = []
        all_seeds = []
        all_negative = []
        infotexts = []
        for args in jobs:
            state.job = f"{state.job_no + 1} out of {state.job_count}"
            copy_p = copy.copy(p)
            for k, v in args.items():
                setattr(copy_p, k, v)
            proc = process_images(copy_p)
            if checkbox_iterate:
                p.seed = p.seed + (p.batch_size * p.n_iter)
            all_seeds += proc.all_seeds
            all_prompts += proc.all_prompts
            all_negative += proc.all_negative_prompts
            images += proc.images
            infotexts += proc.infotexts
        return Processed(p, images, p.seed, "", all_prompts=all_prompts, all_seeds=all_seeds, all_negative_prompts=all_negative, infotexts=infotexts)
