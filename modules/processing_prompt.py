import os
import torch
from modules import shared, errors, timer, prompt_parser_diffusers


debug_enabled = os.environ.get('SD_PROMPT_DEBUG', None) is not None
debug_log = shared.log.trace if debug_enabled else lambda *args, **kwargs: None


def fix_prompt_batch(p, prompts, negative_prompts, prompts_2, negative_prompts_2):
    if hasattr(p, 'keep_prompts'):
        return prompts, negative_prompts, prompts_2, negative_prompts_2

    if type(prompts) is str:
        prompts = [prompts]
    if type(negative_prompts) is str:
        negative_prompts = [negative_prompts]

    if hasattr(p, '[init_images]') and p.init_images is not None and len(p.init_images) > 1:
        while len(prompts) < len(p.init_images):
            prompts.append(prompts[-1])
        while len(negative_prompts) < len(p.init_images):
            negative_prompts.append(negative_prompts[-1])

    while len(prompts) < p.batch_size:
        prompts.append(prompts[-1])
    while len(negative_prompts) < p.batch_size:
        negative_prompts.append(negative_prompts[-1])

    while len(negative_prompts) < len(prompts):
        negative_prompts.append(negative_prompts[-1])
    while len(prompts) < len(negative_prompts):
        prompts.append(prompts[-1])

    if type(prompts_2) is str:
        prompts_2 = [prompts_2]
    if type(prompts_2) is list:
        while len(prompts_2) < len(prompts):
            prompts_2.append(prompts_2[-1])
    if type(negative_prompts_2) is str:
        negative_prompts_2 = [negative_prompts_2]
    if type(negative_prompts_2) is list:
        while len(negative_prompts_2) < len(prompts_2):
            negative_prompts_2.append(negative_prompts_2[-1])
    return prompts, negative_prompts, prompts_2, negative_prompts_2


def fix_prompt_model(cls, prompts, negative_prompts, prompts_2, negative_prompts_2):
    if 'OmniGen' in cls:
        prompts = [p.replace('|image|', '<img><|image_1|></img>') for p in prompts]
    if 'PixArtSigmaPipeline' in cls: # pixart-sigma pipeline throws list-of-list for negative prompt
        negative_prompts = negative_prompts[0]
    return prompts, negative_prompts, prompts_2, negative_prompts_2


def set_fallback_prompt(args: dict, possible: list[str], prompts, negative_prompts, prompts_2, negative_prompts_2) -> dict:
    if ('prompt' in possible) and ('prompt' not in args) and (prompts is not None) and len(prompts) > 0:
        debug_log(f'Prompt fallback: prompt={prompts}')
        args['prompt'] = prompts
    if ('negative_prompt' in possible) and ('negative_prompt' not in args) and (negative_prompts is not None) and len(negative_prompts) > 0:
        debug_log(f'Prompt fallback: negative_prompt={negative_prompts}')
        args['negative_prompt'] = negative_prompts
    if ('prompt_2' in possible) and ('prompt_2' not in args) and (prompts_2 is not None) and len(prompts_2) > 0:
        debug_log(f'Prompt fallback: prompt_2={prompts_2}')
        args['prompt_2'] = prompts_2
    if ('negative_prompt_2' in possible) and ('negative_prompt_2' not in args) and (negative_prompts_2 is not None) and len(negative_prompts_2) > 0:
        debug_log(f'Prompt fallback: negative_prompt_2={negative_prompts_2}')
        args['negative_prompt_2'] = negative_prompts_2
    return args


def set_prompt(p,
               args: dict,
               possible: list[str],
               cls: str,
               prompt_attention: str,
               steps: int,
               clip_skip: int,
               prompts: list[str],
               negative_prompts: list[str],
               prompts_2: list[str],
               negative_prompts_2: list[str],
              ) -> dict:
    prompt_attention = prompt_attention or shared.opts.prompt_attention
    if (prompt_attention != 'fixed') and ('Onnx' not in cls) and ('prompt' not in p.task_args) and (
        ('StableDiffusion' in cls) or
        ('StableCascade' in cls) or
        ('Flux' in cls and 'Flux2' not in cls) or
        ('Chroma' in cls) or
        ('HiDreamImagePipeline' in cls)
    ):
        jobid = shared.state.begin('TE Encode')
        try:
            prompt_parser_diffusers.embedder = prompt_parser_diffusers.PromptEmbedder(prompts, negative_prompts, steps, clip_skip, p)
        except Exception as e:
            prompt_parser_diffusers.embedder = None
            shared.log.error(f'Prompt parser encode: {e}')
            if debug_enabled:
                errors.display(e, 'Prompt parser encode')
        timer.process.record('prompt', reset=False)
        shared.state.end(jobid)
    else:
        prompt_parser_diffusers.embedder = None
        prompt_attention = 'fixed'

    prompts, negative_prompts, prompts_2, negative_prompts_2 = fix_prompt_batch(p, prompts, negative_prompts, prompts_2, negative_prompts_2)
    prompts, negative_prompts, prompts_2, negative_prompts_2 = fix_prompt_model(cls, prompts, negative_prompts, prompts_2, negative_prompts_2)
    args = set_fallback_prompt(args, possible, prompts=None, negative_prompts=None, prompts_2=prompts_2, negative_prompts_2=negative_prompts_2) # we dont parse secondary prompts

    if prompt_parser_diffusers.embedder is not None:
        if 'prompt' in possible:
            debug_log(f'Prompt set embeds: positive={prompts}')
            prompt_embeds = prompt_parser_diffusers.embedder('prompt_embeds')
            prompt_pooled_embeds = prompt_parser_diffusers.embedder('positive_pooleds')
            prompt_attention_masks = prompt_parser_diffusers.embedder('prompt_attention_masks')

            if prompt_embeds is None:
                shared.log.warning('Prompt parser encode: empty prompt embeds')
                prompt_parser_diffusers.embedder = None
                args = set_fallback_prompt(args, possible, prompts=prompts, negative_prompts=None, prompts_2=None, negative_prompts_2=None)
                prompt_attention = 'fixed'
            elif prompt_embeds.device == torch.device('meta'):
                shared.log.warning('Prompt parser encode: embeds on meta device')
                prompt_parser_diffusers.embedder = None
                args = set_fallback_prompt(args, possible, prompts=prompts, negative_prompts=None, prompts_2=None, negative_prompts_2=None)
                prompt_attention = 'fixed'
            else:
                if 'prompt_embeds' in possible:
                    args['prompt_embeds'] = prompt_embeds
                if 'pooled_prompt_embeds' in possible:
                    args['pooled_prompt_embeds'] = prompt_pooled_embeds
                    if 'StableCascade' in cls:
                        args['prompt_embeds_pooled'] = prompt_pooled_embeds.unsqueeze(0)
                    if 'HiDreamImage' in cls:
                        args['prompt_embeds_t5'] = prompt_embeds[0]
                        args['prompt_embeds_llama3'] = prompt_embeds[1]
                if 'prompt_attention_mask' in possible:
                    args['prompt_attention_mask'] = prompt_attention_masks

        if 'negative_prompt' in possible:
            debug_log(f'Prompt set embeds: negative={negative_prompts}')
            negative_embeds = prompt_parser_diffusers.embedder('negative_prompt_embeds')
            negative_pooled_embeds = prompt_parser_diffusers.embedder('negative_pooleds')
            negative_attention_masks = prompt_parser_diffusers.embedder('negative_prompt_attention_masks')

            if negative_embeds is None:
                shared.log.warning('Prompt parser encode: empty negative prompt embeds')
                prompt_parser_diffusers.embedder = None
                args = set_fallback_prompt(args, possible, prompts=None, negative_prompts=negative_prompts, prompts_2=None, negative_prompts_2=None)
                prompt_attention = 'fixed'
            elif negative_embeds.device == torch.device('meta'):
                shared.log.warning('Prompt parser encode: negative embeds on meta device')
                prompt_parser_diffusers.embedder = None
                args = set_fallback_prompt(args, possible, prompts=None, negative_prompts=negative_prompts, prompts_2=None, negative_prompts_2=None)
                prompt_attention = 'fixed'
            else:
                if 'negative_prompt_embeds' in possible:
                    args['negative_prompt_embeds'] = negative_embeds
                if 'negative_pooled_prompt_embeds' in possible:
                    args['negative_pooled_prompt_embeds'] = negative_pooled_embeds
                    if 'StableCascade' in cls:
                        args['negative_prompt_embeds_pooled'] = negative_pooled_embeds.unsqueeze(0)
                    if 'HiDreamImage' in cls:
                        args['negative_prompt_embeds_t5'] = negative_embeds[0]
                        args['negative_prompt_embeds_llama3'] = negative_embeds[1]
                if 'negative_prompt_attention_mask' in possible:
                    args['negative_prompt_attention_mask'] = negative_attention_masks
    else:
        debug_log('Prompt fallback: no embedder')
        args = set_fallback_prompt(args, possible, prompts=prompts, negative_prompts=negative_prompts, prompts_2=None, negative_prompts_2=None)
        prompt_attention = 'fixed'

    if (prompt_parser_diffusers.embedder is not None) and (not prompt_parser_diffusers.embedder.scheduled_prompt):
        prompt_parser_diffusers.embedder = None # not scheduled so we dont need it anymore

    return prompt_attention, args
