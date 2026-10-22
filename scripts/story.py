import json
import os
import re
import time
import copy
import gradio as gr
from PIL import Image
from modules import scripts_manager, shared, processing
from modules.logger import log
from scripts.prompt_enhance.options import Options
from scripts.prompt_enhance.helpers import is_vision_model, get_model_repo_from_display


debug_log = log.trace if os.environ.get('SD_STORY_DEBUG', None) is not None else lambda *args, **kwargs: None


class StoryScript(scripts_manager.Script):
    def title(self):
        return 'Story mode'

    def show(self, _is_img2img):
        return scripts_manager.AlwaysVisible if self.is_control else False

    def ui(self, _is_img2img):
        vision_models = [model for model in Options.get_model_choices()if is_vision_model(get_model_repo_from_display(model))]
        default_model = Options.get_default_display()
        if default_model not in vision_models:
            default_model = vision_models[0] if vision_models else ''
        with gr.Accordion('Story mode', open=False, elem_id='story_mode'):
            with gr.Row():
                enabled = gr.Checkbox(label='Story mode enabled', value=False, elem_id='story_enabled')
            with gr.Row():
                llm_model = gr.Dropdown(label='LLM model', choices=vision_models, value=default_model)
            with gr.Row():
                with gr.Group():
                    slide_count = gr.Slider(label='Number of slides', value=4, minimum=2, maximum=99, step=1, elem_id='story_slide_count')
                    use_enhance = gr.Checkbox(label='Enhance slide prompts', value=False, elem_id='story_use_enhance')
                    use_captions = gr.Checkbox(label='Use image caption', value=False, elem_id='story_use_captions')
                    use_reference = gr.Checkbox(label='Use image as reference', value=False, elem_id='story_use_reference')
        return [enabled, slide_count, llm_model, use_enhance, use_captions, use_reference]

    def get_enhancer(self, p):
        runner = getattr(p, 'scripts', None)
        for script in getattr(runner, 'scripts', []):
            if script.title() == 'Prompt enhance' and callable(getattr(script, 'enhance', None)):
                return script
        return None

    def get_control_image(self, p):
        source = getattr(p, 'orig_init_images', None)
        if isinstance(source, list):
            source = next((image for image in source if image is not None), None)
        if isinstance(source, Image.Image):
            return source
        if isinstance(source, (str, os.PathLike)) and os.path.isfile(source):
            with Image.open(source) as image:
                return image.copy()
        return None

    def parse_json(self, response):
        response = re.sub(r'^\s*```(?:json)?\s*|\s*```\s*$', '', response.strip(), flags=re.IGNORECASE)
        match = re.search(r'(\[.*\]|\{.*\})', response, flags=re.DOTALL)
        if match is None:
            raise ValueError('Story: LLM response did not contain a JSON value')
        return json.loads(match.group(1))

    def parse_frame_prompts(self, response, slide_count):
        data = self.parse_json(response)
        if isinstance(data, dict):
            data = data.get('prompts', data.get('frames'))
        if not isinstance(data, list):
            raise ValueError('Story: LLM response must contain a list of frame prompts')
        prompts = []
        for item in data:
            if isinstance(item, str):
                prompt = item
            elif isinstance(item, dict):
                prompt = item.get('prompt')
            else:
                continue
            if isinstance(prompt, str) and prompt.strip():
                prompts.append(prompt.strip())
        if len(prompts) != slide_count:
            raise ValueError(f'Story: LLM returned {len(prompts)} frame prompts; expected {slide_count}')
        return prompts

    def process(self, p, *args, **kwargs): # pylint: disable=arguments-differ,unused-argument
        # called before actual processing and initializes all story mode related attributes
        if getattr(p, 'story_mode_processing', False):
            return
        enabled, slide_count, llm_model, use_enhance, use_captions, use_reference = args
        if not enabled:
            return

        task_id = shared.state.begin('Story')
        p.story_mode_error = None
        started = time.perf_counter()
        stage = 'validate'
        try:
            if not hasattr(p, 'story_mode_source_prompt'):
                story_prompt = p.prompt
                if p.styles:
                    story_prompt = shared.prompt_styles.apply_styles_to_prompt(story_prompt, p.styles)
                    p.negative_prompt = shared.prompt_styles.apply_negative_styles_to_prompt(p.negative_prompt, p.styles)
                    shared.prompt_styles.apply_styles_to_extra(p)
                    p.prompt = story_prompt
                    p.styles = []
                p.story_mode_source_prompt = story_prompt
            story_prompt = p.story_mode_source_prompt
            story_prompt = story_prompt.strip() if isinstance(story_prompt, str) else ''
            slide_count = int(slide_count)
            if slide_count < 2 or slide_count > 99:
                raise ValueError('Story: Number of slides must be between 2 and 99')
            enhancer = self.get_enhancer(p)
            if enhancer is None:
                raise RuntimeError('Story: Prompt enhance script is unavailable')

            model = get_model_repo_from_display(llm_model) if llm_model else enhancer.model or enhancer.options.default
            control_image = self.get_control_image(p)
            if not story_prompt and control_image is None:
                raise ValueError('Story: Control prompt and input image are both missing')
            story_context = story_prompt
            log.info(f'Story: slides={slide_count} model="{model}" image={control_image} captions={use_captions} reference={use_reference}')
            debug_log(f'Story: prompt="{story_prompt}"')
            if not is_vision_model(model):
                raise RuntimeError(f'Story: model="{model}" is not a vision model')
            if use_reference and control_image is None:
                raise RuntimeError('Story: image=None reference=True reference requires image')

            # caption
            if (use_captions or not story_prompt) and control_image is not None:
                stage = 'caption'
                caption = enhancer.enhance(
                    prompt='Analyze the attached image for use as a visual reference in story-slide generation.',
                    model=model,
                    image=control_image,
                    use_vision=True,
                    system=(
                        f'Return a detailed plain-text visual reference for {slide_count} story slides. '
                        f'Give each character a stable distinguishing description and keep their traits separate. '
                        f'Describe the image\'s artistic medium and rendering style precisely, including traits such as 2D anime illustration, cel shading, line art, or photographic realism. '
                        f'Describe only visible evidence; do not invent names, backstory, or hidden details. '
                        f'Include setting, props, lighting, composition, and visual style. '
                        f'Do not use JSON or markdown.'
                    ),
                    prompt_style='None',
                )
                if not isinstance(caption, str) or not caption.strip() or caption.startswith('Error:'):
                    raise RuntimeError(f'Story: image caption: {caption}')
                debug_log(f'Story: caption="{caption.strip()}"')
                if story_prompt:
                    story_context = f'{story_context}\n\nReference image description: {caption.strip()}'
                else:
                    story_context = f'Invent an imaginative, coherent story inspired by this image description:\n{caption.strip()}'
                debug_log(f'Story: context="{story_context}"')
            elif use_captions:
                log.warning('Story: image=None caption=True no image provided')

            # generate story prompts
            stage = 'story'
            is_edit_model = getattr(shared.sd_model, 'max_condition_images', None) is not None
            if is_edit_model and use_reference and control_image is not None:
                story_system = (
                    f'Break this story into exactly {slide_count} ordered consecutive frames. '
                    f'For each frame, write one concise, fully standalone image-edit instruction describing only the changes needed to show that story moment in the supplied reference image. '
                    f'Treat the supplied image as the visual baseline. Identify existing subjects unambiguously, but do not restate their unchanged appearance, clothing, setting, or style in detail. '
                    f'Focus on relevant changes to action, pose, expression, interaction, or visible objects. Preserve all unmentioned content and the source image\'s artistic medium and rendering style. '
                    f'Each instruction is applied independently to the original reference image, not to a previous slide. Do not refer to other frames or rely on previous prompts. '
                    f'Return only a JSON array of {slide_count} strings, with no titles, descriptions, markdown, or commentary.'
                )
            else:
                story_system = (
                    f'Break this story into exactly {slide_count} ordered consecutive frames. '
                    f'For each frame, write one short, fully standalone image-generation prompt describing the main visible moment. '
                    f'Preserve continuity by explicitly repeating each visible subject\'s identity, defining physical appearance, clothing, and other relevant traits in every prompt. '
                    f'Restate the setting and visual style details needed to generate that frame independently. '
                    f'When a source image description is provided, preserve its artistic medium and rendering style in every frame. '
                    f'Never refer to another frame or use shorthand such as "the same person", "same character", "again", "still", "as before", or "previous frame". '
                    f'Do not use pronouns or vague phrases in place of explicitly describing the subject. '
                    f'Return only a JSON array of {slide_count} prompt strings, with no titles, descriptions, markdown, or commentary.'
                )
            if use_reference and control_image is not None and not is_edit_model:
                story_system += (
                    f'Use the supplied reference image as visual context for the story. '
                    f'Preserve its identifiable subjects, style, artistic medium, and rendering details across all {slide_count} frame prompts.'
                )
            debug_log(f'Story: system="{story_system}"')
            story_max_tokens = min(8192, 256 + slide_count * 128)
            enhanced_story = enhancer.enhance(
                prompt=story_context,
                model=model,
                system=story_system,
                max_tokens=story_max_tokens,
                image=control_image if use_reference else None,
                use_vision=bool(use_reference and control_image is not None),
                prompt_style='None',
            )
            if not isinstance(enhanced_story, str) or not enhanced_story.strip() or enhanced_story.startswith('Error:'):
                raise RuntimeError(f'story enhancement failed: {enhanced_story}')
            enhanced_story = enhanced_story.strip()
            debug_log(f'Story: enhanced="{enhanced_story}"')

            story_prompts = self.parse_frame_prompts(enhanced_story, slide_count)
            debug_log(f'Story: story={story_prompts}')

            # enhance each story prompt
            prompts = []
            for index, frame_prompt in enumerate(story_prompts, start=1):
                if use_enhance:
                    stage = f'prompt-{index}'
                    debug_log(f'Story: generating detailed prompt frame={index}/{slide_count}')
                    prompt = enhancer.enhance(
                        prompt=(
                            f'Expand this short frame prompt into one detailed image-generation prompt for frame {index} of '
                            f'{slide_count}. Preserve continuity with the story and describe only what should be visible in '
                            f'this frame. Preserve the source image\'s artistic medium and rendering style; do not convert anime illustration '
                            f'into photography or change to another medium. Return only the prompt, with no explanation.\n\nStory:\n{story_context}\n\n'
                            f'Frame prompt: {frame_prompt}'
                        ),
                        model=model,
                        system=(
                            f'Return one detailed image-generation prompt for frame {index} of {slide_count} as plain text. '
                            f'Preserve the source image\'s artistic medium and rendering style; do not convert between illustration and photography. '
                            f'Do not include a heading, numbering, markdown, or commentary.'
                        ),
                        prompt_style='None',
                    )
                    if not isinstance(prompt, str) or not prompt.strip() or prompt.startswith('Error:'):
                        raise RuntimeError(f'Story: detailed prompt generation failed for frame {index}: {prompt}')
                    prompt = prompt.strip()
                else:
                    prompt = frame_prompt
                prompts.append(prompt)
                debug_log(f'Story: slide={index} total={slide_count} prompt="{prompt}"')
        except Exception as e:
            p.story_mode_error = f'{stage}: {e}'
            p.n_iter = 1
            shared.state.end(task_id)
            log.error(f'Story: failed stage={stage} {e}')
            return

        seed = processing.get_fixed_seed(p.seed)
        p.story_mode_enabled = True
        p.story_mode_prompts = prompts
        p.story_mode_reference = control_image if use_reference else None
        p.story_mode_seed = seed
        p.story_mode_started = started
        p.n_iter = 1
        p.batch_size = len(prompts)
        p.prompt = prompts
        p.all_prompts = prompts
        p.all_negative_prompts = [p.negative_prompt] * len(prompts)
        p.all_seeds = [seed] * len(prompts)
        p.extra_generation_params['Story'] = f'{len(prompts)} slides'
        processing.process_init(p)
        shared.state.end(task_id)

    def process_images(self, p, *args, **kwargs): # pylint: disable=arguments-differ,unused-argument
        # called instead of normal image processing when story mode is active
        if getattr(p, 'story_mode_processing', False):
            return None
        try:
            if getattr(p, 'story_mode_error', None):
                message = f'Story mode failed: {p.story_mode_error}'
                log.error(message)
                return processing.get_processed(p, [], p.seed, info=message, comments=message)
            if not getattr(p, 'story_mode_enabled', False):
                return None

            p.story_mode_processing = True
            p.do_not_save_grid = True
            p.styles = []
            all_images = []
            all_prompts = []
            all_negative_prompts = []
            all_seeds = []
            all_infotexts = []
            seed = p.story_mode_seed
            slide_error = None
            # loops over slides
            for index in range(len(p.story_mode_prompts)):
                if shared.state.interrupted:
                    break
                log.info(f'Story: slide={index + 1} total={len(p.story_mode_prompts)}')
                slide_p = copy.copy(p)
                slide_p.prompt = p.story_mode_prompts[index]
                slide_p.seed = seed
                slide_p.denoising_strength = 0.80 # high value so normal img2img models also have a chance
                slide_p.n_iter = 1
                slide_p.batch_size = 1
                slide_p.all_prompts = []
                slide_p.all_negative_prompts = []
                slide_p.all_seeds = []
                slide_p.do_not_save_samples = True
                slide_p.do_not_save_grid = True
                if p.story_mode_reference is not None:
                    slide_p.init_images = [p.story_mode_reference]
                else:
                    slide_p.init_images = []

                try:
                    slide = processing.process_images(slide_p)
                    if slide is None or not slide.images:
                        if shared.state.interrupted:
                            break
                        raise RuntimeError('Story: generation returned no images')

                    # collect results from the current slide and combine them
                    all_images.append(slide.images[0])
                    all_infotexts.append(slide.infotexts[0])
                    all_prompts.append(slide.all_prompts[0])
                    all_negative_prompts.append(slide.all_negative_prompts[0])
                    all_seeds.append(slide.all_seeds[0])
                    debug_log(f'Story: slide={index + 1} images={len(slide.images)} {slide}')
                except Exception as e:
                    slide_error = f'slide {index + 1}: {e}'
                    break
            if slide_error:
                p.story_mode_error = slide_error
                log.error(f'Story mode failed: {slide_error}')

            # create single output
            processed = processing.get_processed(
                p,
                all_images,
                seed,
                slide_error or '',
                all_prompts=all_prompts,
                all_negative_prompts=all_negative_prompts,
                all_seeds=all_seeds,
                infotexts=all_infotexts,
                comments=slide_error or '',
            )
            debug_log(f'Story: processed={processed}')
            log.info(f'Story: complete slides={len(all_images)} time={time.perf_counter() - p.story_mode_started:.2f}s')
            return processed
        finally:
            p.story_mode_processing = False
