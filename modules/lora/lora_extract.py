import os
import time
import json
import datetime
import torch
from safetensors.torch import save_file
import gradio as gr
from rich import progress as rp
from modules import shared, devices
from modules.logger import log, console
from modules.ui_common import create_refresh_button
from modules.call_queue import wrap_gradio_gpu_call


class SVDHandler:
    def __init__(self, maxrank=0, rank_ratio=1):
        self.network_name: str = None
        self.U: torch.Tensor = None
        self.S: torch.Tensor = None
        self.Vh: torch.Tensor = None
        self.maxrank: int = maxrank
        self.rank_ratio: float = rank_ratio
        self.rank: int = 0
        self.out_size: int = None
        self.in_size: int = None
        self.kernel_size: tuple[int, int] = None
        self.conv2d: bool = False

    def decompose(self, weight, backupweight):
        self.conv2d = len(weight.size()) == 4
        self.kernel_size = None if not self.conv2d else weight.size()[2:4]
        self.out_size, self.in_size = weight.size()[0:2]
        diffweight = weight.clone().to(devices.device)
        diffweight -= backupweight.to(devices.device)
        if self.conv2d:
            if self.conv2d and self.kernel_size != (1, 1):
                diffweight = diffweight.flatten(start_dim=1)
            else:
                diffweight = diffweight.squeeze()
        self.U, self.S, self.Vh = torch.svd_lowrank(diffweight.to(device=devices.device, dtype=torch.float), self.maxrank, 2)
        # del diffweight
        self.U = self.U.to(device=devices.cpu, dtype=torch.bfloat16)
        self.S = self.S.to(device=devices.cpu, dtype=torch.bfloat16)
        self.Vh = self.Vh.t().to(device=devices.cpu, dtype=torch.bfloat16)  # svd_lowrank outputs a transposed matrix

    def findrank(self):
        if self.rank_ratio < 1:
            S_squared = self.S.pow(2)
            S_fro_sq = float(torch.sum(S_squared))
            sum_S_squared = torch.cumsum(S_squared, dim=0) / S_fro_sq
            index = int(torch.searchsorted(sum_S_squared, self.rank_ratio ** 2)) + 1
            index = max(1, min(index, len(self.S) - 1))
            self.rank = index
            if self.maxrank > 0:
                self.rank = min(self.rank, self.maxrank)
        else:
            self.rank = min(self.in_size, self.out_size, self.maxrank)

    def makeweights(self):
        self.findrank()
        up = self.U[:, :self.rank] @ torch.diag(self.S[:self.rank])
        down = self.Vh[:self.rank, :]
        if self.conv2d and self.kernel_size is not None:
            up = up.reshape(self.out_size, self.rank, 1, 1)
            down = down.reshape(self.rank, self.in_size, self.kernel_size[0], self.kernel_size[1]) # pylint: disable=unsubscriptable-object
        return_dict = {f'{self.network_name}.lora_up.weight': up.contiguous(),
                       f'{self.network_name}.lora_down.weight': down.contiguous(),
                       f'{self.network_name}.alpha': torch.tensor(down.shape[0]),
                       }
        return return_dict


def loaded_lora():
    if not shared.sd_loaded:
        return ""
    loaded = set()
    if hasattr(shared.sd_model, 'unet'):
        for _name, module in shared.sd_model.unet.named_modules():
            current = getattr(module, "network_current_names", None)
            if current is not None:
                current = [item[0] for item in current]
                loaded.update(current)
    return list(loaded)


def loaded_lora_str():
    return ", ".join(loaded_lora())


def make_meta(fn, maxrank, rank_ratio):
    meta = {
        "model_spec.sai_model_spec": "1.0.0",
        "model_spec.title": os.path.splitext(os.path.basename(fn))[0],
        "model_spec.author": "SD.Next",
        "model_spec.implementation": "https://github.com/vladmandic/sdnext",
        "model_spec.date": datetime.datetime.now().astimezone().replace(microsecond=0).isoformat(),
        "model_spec.base_model": shared.opts.sd_model_checkpoint,
        "model_spec.dtype": str(devices.dtype),
        "model_spec.base_lora": json.dumps(loaded_lora()),
        "model_spec.config": f"maxrank={maxrank} rank_ratio={rank_ratio}",
    }
    if shared.sd_model_type == "sdxl":
        meta["model_spec.architecture"] = "stable-diffusion-xl-v1-base/lora" # sai standard
        meta["ss_base_model_version"] = "sdxl_base_v1-0" # kohya standard
    elif shared.sd_model_type == "sd":
        meta["model_spec.architecture"] = "stable-diffusion-v1/lora"
        meta["ss_base_model_version"] = "sd_v1"
    elif shared.sd_model_type == "f1":
        meta["model_spec.architecture"] = "flux-1-dev/lora"
        meta["ss_base_model_version"] = "flux1"
    elif shared.sd_model_type == "chroma":
        meta["model_spec.architecture"] = "chroma/lora"
        meta["ss_base_model_version"] = "chroma"
    elif shared.sd_model_type == "sc":
        meta["model_spec.architecture"] = "stable-cascade-v1-prior/lora"
    return meta


def make_lora(fn, maxrank, auto_rank, rank_ratio, modules, overwrite):
    if not shared.sd_loaded:
        msg = "LoRA extract: model not loaded"
        log.warning(msg)
        yield msg
        return
    if loaded_lora() == "":
        msg = "LoRA extract: no LoRA detected"
        log.warning(msg)
        yield msg
        return
    if not fn:
        msg = "LoRA extract: target filename required"
        log.warning(msg)
        yield msg
        return
    t0 = time.time()
    maxrank = int(maxrank)
    rank_ratio = 1 if not auto_rank else rank_ratio
    log.debug(f'LoRA extract: modules={modules} maxrank={maxrank} auto={auto_rank} ratio={rank_ratio} fn="{fn}"')
    jobid = shared.state.begin('LoRA extract')

    with rp.Progress(rp.TextColumn('[cyan]LoRA extract'), rp.BarColumn(), rp.TaskProgressColumn(), rp.TimeRemainingColumn(), rp.TimeElapsedColumn(), rp.TextColumn('[cyan]{task.description}'), console=console) as progress:

        if 'te' in modules and getattr(shared.sd_model, 'text_encoder', None) is not None:
            modules = shared.sd_model.text_encoder.named_modules()
            task = progress.add_task(description="te1 decompose", total=len(list(modules)))
            for name, module in shared.sd_model.text_encoder.named_modules():
                progress.update(task, advance=1)
                weights_backup = getattr(module, "network_weights_backup", None)
                if weights_backup is None or getattr(module, "network_current_names", None) is None:
                    continue
                prefix = "lora_te1_" if hasattr(shared.sd_model, 'text_encoder_2') else "lora_te_"
                module.svdhandler = SVDHandler(maxrank, rank_ratio)
                module.svdhandler.network_name = prefix + name.replace(".", "_")
                with devices.inference_context():
                    module.svdhandler.decompose(module.weight, weights_backup)
            progress.remove_task(task)
        t1 = time.time()

        if 'te' in modules and getattr(shared.sd_model, 'text_encoder_2', None) is not None:
            modules = shared.sd_model.text_encoder_2.named_modules()
            task = progress.add_task(description="te2 decompose", total=len(list(modules)))
            for name, module in shared.sd_model.text_encoder_2.named_modules():
                progress.update(task, advance=1)
                weights_backup = getattr(module, "network_weights_backup", None)
                if weights_backup is None or getattr(module, "network_current_names", None) is None:
                    continue
                module.svdhandler = SVDHandler(maxrank, rank_ratio)
                module.svdhandler.network_name = "lora_te2_" + name.replace(".", "_")
                with devices.inference_context():
                    module.svdhandler.decompose(module.weight, weights_backup)
            progress.remove_task(task)
        t2 = time.time()

        if 'unet' in modules and getattr(shared.sd_model, 'unet', None) is not None:
            modules = shared.sd_model.unet.named_modules()
            task = progress.add_task(description="unet decompose", total=len(list(modules)))
            for name, module in shared.sd_model.unet.named_modules():
                progress.update(task, advance=1)
                weights_backup = getattr(module, "network_weights_backup", None)
                if weights_backup is None or getattr(module, "network_current_names", None) is None:
                    continue
                module.svdhandler = SVDHandler(maxrank, rank_ratio)
                module.svdhandler.network_name = "lora_unet_" + name.replace(".", "_")
                with devices.inference_context():
                    module.svdhandler.decompose(module.weight, weights_backup)
            progress.remove_task(task)
        t3 = time.time()

        # TODO: lora: support pre-quantized flux
        # if 'te' in modules and getattr(shared.sd_model, 'transformer', None) is not None:
        #     for name, module in shared.sd_model.transformer.named_modules():
        #         if "norm" in name and "linear" not in name:
        #             continue
        #         weights_backup = getattr(module, "network_weights_backup", None)
        #         if weights_backup is None:
        #             continue
        #         module.svdhandler = SVDHandler()
        #         module.svdhandler.network_name = "lora_transformer_" + name.replace(".", "_")
        #         module.svdhandler.decompose(module.weight, weights_backup)
        #         module.svdhandler.findrank(rank, rank_ratio)

        lora_state_dict = {}
        for sub in ['text_encoder', 'text_encoder_2', 'unet', 'transformer']:
            submodel = getattr(shared.sd_model, sub, None)
            if submodel is not None:
                modules = submodel.named_modules()
                task = progress.add_task(description=f"{sub} exctract", total=len(list(modules)))
                for _name, module in submodel.named_modules():
                    progress.update(task, advance=1)
                    if not hasattr(module, "svdhandler"):
                        continue
                    lora_state_dict.update(module.svdhandler.makeweights())
                    del module.svdhandler
                progress.remove_task(task)
        t4 = time.time()

    if not os.path.isabs(fn):
        fn = os.path.join(shared.cmd_opts.lora_dir, fn)
    if not fn.endswith('.safetensors'):
        fn += '.safetensors'
    if os.path.exists(fn):
        if overwrite:
            os.remove(fn)
        else:
            msg = f'LoRA extract: fn="{fn}" file exists'
            log.warning(msg)
            yield msg
            return

    shared.state.end(jobid)
    meta = make_meta(fn, maxrank, rank_ratio)
    log.debug(f'LoRA metadata: {meta}')
    try:
        save_file(tensors=lora_state_dict, metadata=meta, filename=fn)
    except Exception as e:
        msg = f'LoRA extract error: fn="{fn}" {e}'
        log.error(msg)
        yield msg
        return
    t5 = time.time()
    log.debug(f'LoRA extract: time={t5-t0:.2f} te1={t1-t0:.2f} te2={t2-t1:.2f} unet={t3-t2:.2f} save={t5-t4:.2f}')
    keys = list(lora_state_dict.keys())
    msg = f'LoRA extract: fn="{fn}" keys={len(keys)}'
    log.info(msg)
    yield msg


def create_ui():
    def gr_show(visible=True):
        return {"visible": visible, "__type__": "update"}

    with gr.Tab(label="Extract LoRA"):
        with gr.Row():
            gr.HTML('<h2>&nbspExtract currently loaded LoRA(s)<br></h2>')
        with gr.Row():
            loaded = gr.Textbox(placeholder="Press refresh to query loaded LoRA", label="Loaded LoRA", interactive=False)
            create_refresh_button(loaded, lambda: None, lambda: {'value': loaded_lora_str()}, "lora_extract_refresh")
        with gr.Group():
            with gr.Row():
                modules = gr.CheckboxGroup(label="Modules to extract", value=['unet'], choices=['te', 'unet'])
            with gr.Row():
                auto_rank = gr.Checkbox(value=False, label="Automatically determine rank")
                rank_ratio = gr.Slider(label="Autorank ratio", value=1, minimum=0, maximum=1, step=0.05, visible=False)
                rank = gr.Slider(label="Maximum rank", value=32, minimum=1, maximum=256)
        with gr.Row():
            filename = gr.Textbox(label="LoRA target filename")
            overwrite = gr.Checkbox(value=False, label="Overwrite existing file")
        with gr.Row():
            extract = gr.Button(value="Extract LoRA", variant='primary')
            status = gr.HTML(value="", show_label=False)

        auto_rank.change(fn=lambda x: gr_show(x), inputs=[auto_rank], outputs=[rank_ratio])
        extract.click(
            fn=wrap_gradio_gpu_call(make_lora, extra_outputs=[], name='LoRA'),
            inputs=[filename, rank, auto_rank, rank_ratio, modules, overwrite],
            outputs=[status]
        )
