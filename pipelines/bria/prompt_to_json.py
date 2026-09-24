import time
import json


model = None
repo_id_10 = "briaai/FIBO-edit-prompt-to-JSON"
repo_id_15 = 'briaai/FIBO-VLM-prompt-to-JSON'


def generate_prompt_local(prompt, image=None):
    global model # pylint: disable=global-statement
    from modules import shared, devices, sd_models
    from modules.logger import log

    if model is None:
        from diffusers.modular_pipelines import ModularPipelineBlocks
        model = ModularPipelineBlocks.from_pretrained(repo_id_15,
                                                      trust_remote_code=True,
                                                      # torch_dtype=devices.dtype,
                                                      cache_dir=shared.opts.hfcache_dir,
                                                     )
        model = model.init_pipeline()
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


def generate_prompt_gemini(prompt, image=None):
    global model # pylint: disable=global-statement
    from diffusers.modular_pipelines import ModularPipelineBlocks

    if model is None:
        model = ModularPipelineBlocks.from_pretrained("briaai/FIBO-gemini-prompt-to-JSON", trust_remote_code=True)
        model = model.init_pipeline()

    output = model(prompt=prompt, image=image)
    json_prompt = output.values["json_prompt"]
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
        log.warning(f'BriaFIBO: prompt="{prompt}" is not a valid JSON')

    # dct = generate_prompt_local(prompt)
    # return dct

    # dct = generate_prompt_gemini(prompt)
    # return dct

    json_str = '{ instructions: "' + prompt + '" }'
    return json_str
