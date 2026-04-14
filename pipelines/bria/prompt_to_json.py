import time
import json


model = None


def generate_prompt_local(prompt, image=None, repo_id="briaai/FIBO-edit-prompt-to-JSON"):
    from modules import shared, devices, sd_models, model_quant
    from modules.logger import log

    global model # pylint: disable=global-statement
    if model is None:
        from diffusers.modular_pipelines import ModularPipelineBlocks
        quant_args = model_quant.create_config(module='LLM')
        pipeline = ModularPipelineBlocks.from_pretrained(repo_id,
                                                         trust_remote_code=True,
                                                         torch_dtype=devices.dtype,
                                                         cache_dir=shared.opts.hfcache_dir,
                                                         requirements=None,
                                                         **quant_args,
                                                        )
        model = pipeline.init_pipeline()
        log.debug(f'JSONEncode loaded: model={model} cls={model.__class__}')


    t0 = time.time()
    sd_models.move_model(model, devices.device)
    output = model(prompt=prompt, image=image)
    json_prompt = output.values["json_prompt"]
    sd_models.move_model(model, devices.cpu)
    devices.torch_gc()
    t1 = time.time()
    log.debug(f'JSONEncode: model={model} prompt="{prompt}" json="{json_prompt}" time={t1-t0:.2f}')
    return json_prompt


def before_prompt_encode(prompt):
    if isinstance(prompt, list):
        prompt = prompt[0] if len(prompt) > 0 else ''

    try:
        json_data = json.loads(prompt)
        json_str = json.dumps(json_data)
        return json_str
    except Exception: # not a json
        from modules.logger import log
        log.error(f'FIBO-Edit: prompt="{prompt}" is not a valid JSON')

    # dct = generate_prompt_local(prompt)
    # return dct

    json_str = f'{{ "instructions": "{prompt}" }}'
    return json_str
