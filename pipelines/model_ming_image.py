import json
from diffusers.modular_pipelines import ModularPipeline, ModularPipelineBlocks
from modules import devices, model_quant, sd_models, shared, sd_hijack_vae, sd_hijack_te
from modules.logger import log
from pipelines import generic


class MingImageModularPipeline(ModularPipeline):
    pass


def before_prompt_encode(prompt):
    if isinstance(prompt, list):
        prompt = prompt[0] if len(prompt) > 0 else ''
    try:
        json_data = json.loads(prompt)
        json_str = json.dumps(json_data)
        return json_str
    except Exception: # not a json
        log.warning(f'MingImage: prompt="{prompt}" is not a valid JSON')
    json_str = '{ instructions: "' + prompt + '" }'
    return json_str


def load_ming_image(checkpoint_info, diffusers_load_config=None):
    if diffusers_load_config is None:
        diffusers_load_config = {}
    repo_id = sd_models.path_to_repo(checkpoint_info)
    generic.set_pipeline('MingImage', MingImageModularPipeline)

    if repo_id is None or repo_id.lower() == 'none':
        return None
    sd_models.hf_auth_check(checkpoint_info)

    load_args, model_quant_args = model_quant.get_dit_args(diffusers_load_config, module='Model', device_map=True)
    load_args.pop('dtype', None)
    dtype = load_args.pop('torch_dtype', devices.dtype)
    device_map = load_args.pop('device_map', None)

    te_load_args, te_quant_args = model_quant.get_dit_args(diffusers_load_config, module='TE', device_map=True)
    te_device_map = te_load_args.pop('device_map', None)
    te_load_args.pop('torch_dtype', None)
    te_load_args.pop('dtype', None)
    if device_map is not None:
        load_args['device_map'] = {'transformer': device_map}
    if te_device_map is not None:
        load_args['device_map'] = {**load_args.get('device_map', {}), 'text_encoder': te_device_map}
    quant_args = {}
    for name, value in model_quant_args.items():
        quant_args.setdefault(name, {})['transformer'] = value
    for name, value in te_quant_args.items():
        quant_args.setdefault(name, {})['text_encoder'] = value

    offline_args = {'local_files_only': True} if shared.opts.offline_mode else {}
    log.debug(f'Load model: type=MingImage repo="{repo_id}" offload={shared.opts.diffusers_offload_mode} dtype={dtype} quant_model={model_quant.get_quant_type(model_quant_args)} quant_te={model_quant.get_quant_type(te_quant_args)} args={load_args}')
    sd_models.warn_group_offload(min_vram=20)

    blocks = ModularPipelineBlocks.from_pretrained(
        'OzzyGT/ming_image_custom_blocks',
        trust_remote_code=True,
        components_repo=repo_id,
        trust_components_code=True,
        cache_dir=shared.opts.hfcache_dir,
        **offline_args,
    )
    pipe = blocks.init_pipeline()
    pipe.__class__ = MingImageModularPipeline
    log.critical(quant_args)
    pipe.load_components(
        dtype=dtype,
        cache_dir=shared.opts.diffusers_dir,
        trust_remote_code=True,
        **load_args,
        **quant_args,
        **offline_args,
    )

    from modules import modular_load, sd_hijack_modular
    missing = modular_load.missing_components(pipe, workflow=None)
    if missing:
        raise RuntimeError(f'Ming-Image components failed to load: {missing}')
    pipe.task_args = {'output_type': 'np'}
    pipe.sd_checkpoint_info = checkpoint_info
    pipe.before_prompt_encode = before_prompt_encode
    modular_load.apply_progress_bar_config(pipe._blocks) # pylint: disable=protected-access
    sd_hijack_modular.install_state_hook(pipe)
    sd_hijack_modular.register_callbacks(pipe)
    sd_hijack_te.init_hijack(pipe)
    sd_hijack_vae.init_hijack(pipe)

    del blocks
    devices.torch_gc(force=True, reason='load')
    return pipe
