import os
import gradio as gr
from modules import paths
from modules.options import OptionInfo, options_section


class LegacyOption(OptionInfo):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


legacy_options = options_section(('legacy_options', "Legacy options"), {
    "ldsr_models_path": LegacyOption(os.path.join(paths.models_path, 'LDSR'), "LDSR Path", gr.Textbox, { "visible": False}),
    "lora_legacy": LegacyOption(False, "LoRA load using legacy method", gr.Checkbox, {"visible": False}),
    "lora_preferred_name": LegacyOption("filename", "LoRA preferred name", gr.Radio, {"choices": ["filename", "alias"], "visible": False}),
    "img2img_extra_noise": LegacyOption(0.0, "Extra noise multiplier for img2img", gr.Slider, {"minimum": 0.0, "maximum": 1.0, "step": 0.01, "visible": False}),
    "disable_weights_auto_swap": LegacyOption(True, "Do not change selected model when reading generation parameters", gr.Checkbox, {"visible": False}),
    "sub_quad_q_chunk_size": LegacyOption(512, "Attention query chunk size", gr.Slider, {"minimum": 16, "maximum": 8192, "step": 8, "visible": False}),
    "sub_quad_kv_chunk_size": LegacyOption(512, "Attention kv chunk size", gr.Slider, {"minimum": 0, "maximum": 8192, "step": 8, "visible": False}),
    "sub_quad_chunk_threshold": LegacyOption(80, "Attention chunking threshold", gr.Slider, {"minimum": 0, "maximum": 100, "step": 1, "visible": False}),
    "upcast_attn": LegacyOption(False, "Upcast attention layer", gr.Checkbox, {"visible": False}),
    "cuda_cast_unet": LegacyOption(False, "Fixed UNet precision", gr.Checkbox, {"visible": False}),
    "comma_padding_backtrack": LegacyOption(20, "Prompt padding", gr.Slider, {"minimum": 0, "maximum": 74, "step": 1, "visible": False}),
    "sd_textencoder_cache": LegacyOption(True, "Cache text encoder results", gr.Checkbox, {"visible": False}),
    "rollback_vae": LegacyOption(False, "Attempt VAE roll back for NaN values", gr.Checkbox, {"visible": False}),
    "sd_vae_sliced_encode": LegacyOption(False, "VAE sliced encode", gr.Checkbox, {"visible": False}),
    "nan_skip": LegacyOption(False, "Skip Generation if NaN found in latents", gr.Checkbox, {"visible": False}),
    "sd_model_dict": LegacyOption('None', "Use separate base dict", gr.Dropdown, lambda: {"choices": ['None'], "visible": False}),
    "diffusers_move_base": LegacyOption(False, "Move base model to CPU when using refiner", gr.Checkbox, {"visible": False }),
    "diffusers_move_unet": LegacyOption(False, "Move base model to CPU when using VAE", gr.Checkbox, {"visible": False }),
    "diffusers_move_refiner": LegacyOption(False, "Move refiner model to CPU when not in use", gr.Checkbox, {"visible": False }),
    "diffusers_extract_ema": LegacyOption(False, "Use model EMA weights when possible", gr.Checkbox, {"visible": False }),
    "batch_cond_uncond": LegacyOption(True, "Do conditional and unconditional denoising in one batch", gr.Checkbox, {"visible": False}),
    "CLIP_stop_at_last_layers": LegacyOption(1, "Clip skip", gr.Slider, {"minimum": 1, "maximum": 8, "step": 1, "visible": False}),
    "dataset_filename_join_string": LegacyOption(" ", "Filename join string", gr.Textbox, { "visible": False }),
    "dataset_filename_word_regex": LegacyOption("", "Filename word regex", gr.Textbox, { "visible": False }),
    "diffusers_force_zeros": LegacyOption(False, "Force zeros for prompts when empty", gr.Checkbox, {"visible": False}),
    "disable_nan_check": LegacyOption(True, "Disable NaN check", gr.Checkbox, {"visible": False}),
    "embeddings_templates_dir": LegacyOption("", "Embeddings train templates directory", gr.Textbox, { "visible": False }),
    "extra_networks_card_fit": LegacyOption("cover", "UI image contain method", gr.Radio, {"choices": ["contain", "cover", "fill"], "visible": False}),
    "grid_extended_filename": LegacyOption(True, "Add extended info to filename when saving grid", gr.Checkbox, {"visible": False}),
    "grid_save_to_dirs": LegacyOption(False, "Save grids to a subdirectory", gr.Checkbox, {"visible": False}),
    "hypernetwork_enabled": LegacyOption(False, "Enable Hypernetwork support", gr.Checkbox, {"visible": False}),
    "img2img_fix_steps": LegacyOption(False, "For image processing do exact number of steps as specified", gr.Checkbox, { "visible": False }),
    "keyedit_delimiters": LegacyOption(r".,\/!?%^*;:{}=`~()", "Ctrl+up/down word delimiters", gr.Textbox, { "visible": False }),
    "keyedit_precision_attention": LegacyOption(0.1, "Ctrl+up/down precision when editing (attention:1.1)", gr.Slider, {"minimum": 0.01, "maximum": 0.2, "step": 0.001, "visible": False}),
    "keyedit_precision_extra": LegacyOption(0.05, "Ctrl+up/down precision when editing <extra networks:0.9>", gr.Slider, {"minimum": 0.01, "maximum": 0.2, "step": 0.001, "visible": False}),
    "live_preview_content": LegacyOption("Combined", "Live preview subject", gr.Radio, {"choices": ["Combined", "Prompt", "Negative prompt"], "visible": False}),
    "live_previews_enable": LegacyOption(True, "Show live previews", gr.Checkbox, {"visible": False}),
    "lora_functional": LegacyOption(False, "Use Kohya method for handling multiple LoRA", gr.Checkbox, { "visible": False }),
    "lyco_dir": LegacyOption(os.path.join(paths.models_path, 'LyCORIS'), "Folder with LyCORIS network(s)", gr.Text, {"visible": False}),
    "model_reuse_dict": LegacyOption(False, "Reuse loaded model dictionary", gr.Checkbox, {"visible": False}),
    "pad_cond_uncond": LegacyOption(True, "Pad prompt and negative prompt to be same length", gr.Checkbox, {"visible": False}),
    "pin_memory": LegacyOption(True, "Pin training dataset to memory", gr.Checkbox, { "visible": False }),
    "save_optimizer_state": LegacyOption(False, "Save resumable optimizer state when training", gr.Checkbox, { "visible": False }),
    "save_training_settings_to_txt": LegacyOption(True, "Save training settings to a text file", gr.Checkbox, { "visible": False }),
    "sd_disable_ckpt": LegacyOption(False, "Disallow models in ckpt format", gr.Checkbox, {"visible": False}),
    "sd_lora": LegacyOption("", "Add LoRA to prompt", gr.Textbox, {"visible": False}),
    "sd_vae_checkpoint_cache": LegacyOption(0, "Cached VAEs", gr.Slider, {"minimum": 0, "maximum": 10, "step": 1, "visible": False}),
    "show_progress_grid": LegacyOption(True, "Show previews as a grid", gr.Checkbox, {"visible": False}),
    "show_progressbar": LegacyOption(True, "Show progressbar", gr.Checkbox, {"visible": False}),
    "training_enable_tensorboard": LegacyOption(False, "Enable tensorboard logging", gr.Checkbox, { "visible": False }),
    "training_image_repeats_per_epoch": LegacyOption(1, "Image repeats per epoch", gr.Slider, {"minimum": 1, "maximum": 100, "step": 1, "visible": False }),
    "training_tensorboard_flush_every": LegacyOption(120, "Tensorboard flush period", gr.Number, { "visible": False }),
    "training_tensorboard_save_images": LegacyOption(False, "Save generated images within tensorboard", gr.Checkbox, { "visible": False }),
    "training_write_csv_every": LegacyOption(0, "Save loss CSV file every n steps", gr.Number, { "visible": False }),
    "ui_scripts_reorder": LegacyOption("", "UI scripts order", gr.Textbox, { "visible": False }),
    "unload_models_when_training": LegacyOption(False, "Move VAE and CLIP to RAM when training", gr.Checkbox, { "visible": False }),
    "upscaler_for_img2img": LegacyOption("None", "Default upscaler for image resize operations", gr.Dropdown, lambda: {"choices": [], "visible": False}),
    "use_save_to_dirs_for_ui": LegacyOption(False, "Save images to a subdirectory when using Save button", gr.Checkbox, {"visible": False}),
    "use_upscaler_name_as_suffix": LegacyOption(True, "Use upscaler as suffix", gr.Checkbox, {"visible": False}),
})


def get_legacy_options():
    return legacy_options
