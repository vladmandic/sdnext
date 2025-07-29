"""
Additional params for Text-to-Video
<https://huggingface.co/docs/diffusers/api/pipelines/text_to_video>

TODO text2video items:
- Video-to-Video upscaling: <https://huggingface.co/cerspense/zeroscope_v2_XL>, <https://huggingface.co/damo-vilab/MS-Vid2Vid-XL>
"""

import gradio as gr
from modules import scripts_manager, processing, shared, images, sd_models, modelloader


MODELS = [
    {'name': 'None'},
    {'name': 'ModelScope v1.7b', 'path': 'damo-vilab/text-to-video-ms-1.7b', 'params': [16,320,320]},
    {'name': 'ZeroScope v1', 'path': 'cerspense/zeroscope_v1_320s', 'params': [16,320,320]},
    {'name': 'ZeroScope v1.1', 'path': 'cerspense/zeroscope_v1-1_320s', 'params': [16,320,320]},
    {'name': 'ZeroScope v2', 'path': 'cerspense/zeroscope_v2_576w', 'params': [24,576,320]},
    {'name': 'ZeroScope v2 Dark', 'path': 'cerspense/zeroscope_v2_dark_30x448x256', 'params': [24,448,256]},
    {'name': 'Potat v1', 'path': 'camenduru/potat1', 'params': [24,1024,576]},
]


class Script(scripts_manager.Script):
    def title(self):
        return 'Video: ModelScope'

    def show(self, is_img2img):
        return not is_img2img

    # return signature is array of gradio components
    def ui(self, is_img2img):

        def model_info_change(model_name):
            if model_name == 'None':
                return gr.update(value='')
            else:
                model = next(m for m in MODELS if m['name'] == model_name)
                return gr.update(value=f'<span> &nbsp frames: {model["params"][0]} size: {model["params"][1]}x{model["params"][2]}</span> <a href="https://huggingface.co/{model["path"]}" target="_blank">link</a>')

        with gr.Row():
            gr.HTML('<span>&nbsp Text to video</span><br>')
        with gr.Row():
            model_name = gr.Dropdown(label='Model', value='None', choices=[m['name'] for m in MODELS])
        with gr.Row():
            model_info = gr.HTML()
            model_name.change(fn=model_info_change, inputs=[model_name], outputs=[model_info])
        with gr.Row():
            use_default = gr.Checkbox(label='Use defaults', value=True)
            num_frames = gr.Slider(label='Frames', minimum=1, maximum=50, step=1, value=0)
        with gr.Row():
            from modules.ui_sections import create_video_inputs
            video_type, duration, gif_loop, mp4_pad, mp4_interpolate = create_video_inputs(tab='img2img' if is_img2img else 'txt2img')
        return [model_name, use_default, num_frames, video_type, duration, gif_loop, mp4_pad, mp4_interpolate]

    def run(self, p: processing.StableDiffusionProcessing, model_name, use_default, num_frames, video_type, duration, gif_loop, mp4_pad, mp4_interpolate): # pylint: disable=arguments-differ, unused-argument
        if model_name == 'None':
            return None
        model = [m for m in MODELS if m['name'] == model_name][0]
        shared.log.debug(f'Text2Video: model={model} defaults={use_default} frames={num_frames}, video={video_type} duration={duration} loop={gif_loop} pad={mp4_pad} interpolate={mp4_interpolate}')

        if model['path'] in shared.opts.sd_model_checkpoint:
            shared.log.debug(f'Text2Video cached: model={shared.opts.sd_model_checkpoint}')
        else:
            checkpoint = sd_models.get_closet_checkpoint_match(model['path'])
            if checkpoint is None:
                shared.log.debug(f'Text2Video downloading: model={model["path"]}')
                checkpoint = modelloader.download_diffusers_model(hub_id=model['path'])
                sd_models.list_models()
            if checkpoint is None:
                shared.log.error(f'Text2Video: failed to find model={model["path"]}')
                return None
            shared.log.debug(f'Text2Video loading: model={checkpoint}')
            shared.opts.sd_model_checkpoint = checkpoint.name
            sd_models.reload_model_weights(op='model')

        p.ops.append('video')
        p.do_not_save_grid = True
        if use_default:
            p.task_args['num_frames'] = model['params'][0]
            p.width = model['params'][1]
            p.height = model['params'][2]
        elif num_frames > 0:
            p.task_args['num_frames'] = num_frames
        else:
            shared.log.error('Text2Video: invalid number of frames')
            return None

        shared.sd_model = sd_models.set_diffuser_pipe(shared.sd_model, sd_models.DiffusersTaskType.TEXT_2_IMAGE)
        shared.log.debug(f'Text2Video: args={p.task_args}')
        processed = processing.process_images(p)

        if video_type != 'None':
            images.save_video(p, filename=None, images=processed.images, video_type=video_type, duration=duration, loop=gif_loop, pad=mp4_pad, interpolate=mp4_interpolate)
        return processed
