from dataclasses import dataclass
import textwrap
import transformers


@dataclass
class Options:
    img2img = [
        # Gemma
        'google/gemma-3-4b-it',
        'google/gemma-3n-E2B-it',
        'google/gemma-3n-E4B-it',
        'google/gemma-4-E2B-it',
        'google/gemma-4-E4B-it',
        'google/gemma-4-12B-it-qat-w4a16-ct',
        # Qwen3.5
        'Qwen/Qwen3.5-2B',
        'Qwen/Qwen3.5-4B',
        'Qwen/Qwen3.5-9B',
        # Qwen3-VL
        'Qwen/Qwen3-VL-2B-Instruct',
        'Qwen/Qwen3-VL-2B-Thinking',
        'Qwen/Qwen3-VL-4B-Instruct',
        'Qwen/Qwen3-VL-4B-Thinking',
        'Qwen/Qwen3-VL-8B-Instruct',
        'Qwen/Qwen3-VL-8B-Thinking',
        # Qwen2.5-VL
        'Qwen/Qwen2.5-VL-3B-Instruct',
        # Mistral
        'mistralai/Ministral-3-3B-Instruct-2512-BF16',
        'mistralai/Ministral-3-8B-Instruct-2512-BF16',
        'mistralai/Ministral-3-3B-Reasoning-2512',
        'mistralai/Ministral-3-8B-Reasoning-2512',
        # Finetunes
        'trohrbaugh/gemma-4-E4B-it-heretic-ara',
        'trohrbaugh/Qwen3.5-9B-heretic-v2',
    ]
    cloud = [
        'google/gemini-3.8-flash',
        'google/gemini-3.7-flash',
        'google/gemini-3.6-flash',
        'google/gemini-3.5-flash',
        'google/gemini-3.5-flash-lite',
        'google/gemini-3.1-flash-lite',
        'google/gemini-3.1-pro-preview',
    ]
    models = {
        # Gemma
        'google/gemma-3-1b-it': {},
        'google/gemma-3-4b-it': {},
        'google/gemma-3n-E2B-it': {},
        'google/gemma-3n-E4B-it': {},
        'google/gemma-4-E2B-it': {},
        'google/gemma-4-E4B-it': {},
        'google/gemma-4-12B-it-qat-w4a16-ct': {}, # compressed-tensor model
        # Qwen3.5
        'Qwen/Qwen3.5-0.8B': {},
        'Qwen/Qwen3.5-2B': {},
        'Qwen/Qwen3.5-4B': {},
        'Qwen/Qwen3.5-9B': {},
        # Qwen3
        'Qwen/Qwen3-0.6B': {},
        'Qwen/Qwen3-1.7B': {},
        'Qwen/Qwen3-4B': {},
        'Qwen/Qwen3-4B-Instruct-2507': {},
        # Qwen3-VL
        'Qwen/Qwen3-VL-2B-Instruct': {},
        'Qwen/Qwen3-VL-2B-Thinking': {},
        'Qwen/Qwen3-VL-4B-Instruct': {},
        'Qwen/Qwen3-VL-4B-Thinking': {},
        'Qwen/Qwen3-VL-8B-Instruct': {},
        'Qwen/Qwen3-VL-8B-Thinking': {},
        # Qwen2.5
        'Qwen/Qwen2.5-0.5B-Instruct': {},
        'Qwen/Qwen2.5-1.5B-Instruct': {},
        'Qwen/Qwen2.5-3B-Instruct': {},
        # Qwen2.5-VL
        'Qwen/Qwen2.5-VL-3B-Instruct': {},
        # Llama
        'meta-llama/Llama-3.2-1B-Instruct': {},
        'meta-llama/Llama-3.2-3B-Instruct': {},
        'meta-llama/Llama-3.2-8B-Instruct': {},
        'cognitivecomputations/Dolphin3.0-Llama3.2-1B': {},
        'cognitivecomputations/Dolphin3.0-Llama3.2-3B': {},
        # Gemini
        'google/gemini-3.8-flash': {},
        'google/gemini-3.7-flash': {},
        'google/gemini-3.6-flash': {},
        'google/gemini-3.5-flash': {},
        'google/gemini-3.5-flash-lite': {},
        'google/gemini-3.1-flash-lite': {},
        'google/gemini-3.1-pro-preview': {},
        # SmolLM
        'HuggingFaceTB/SmolLM2-135M-Instruct': {},
        'HuggingFaceTB/SmolLM2-360M-Instruct': {},
        'HuggingFaceTB/SmolLM2-1.7B-Instruct': {},
        'HuggingFaceTB/SmolLM3-3B': {},
        # Phi
        'microsoft/Phi-4-mini-instruct': {},
        # Mistral
        'mistralai/Ministral-3-3B-Instruct-2512-BF16': {},
        'mistralai/Ministral-3-8B-Instruct-2512-BF16': {},
        'mistralai/Ministral-3-3B-Reasoning-2512': {},
        'mistralai/Ministral-3-8B-Reasoning-2512': {},
        # Finetunes
        'p-e-w/gemma-4-E2B-it-heretic-ara': {},
        'trohrbaugh/gemma-4-E4B-it-heretic-ara': {},
        'trohrbaugh/Qwen3.5-9B-heretic-v2': {},
        # GGUF
        'mradermacher/Llama-3.2-1B-Instruct-Uncensored-i1-GGUF': { # kept primarily as an example how to add gguf model
            'repo': 'meta-llama/Llama-3.2-1B-Instruct', # original repo so we can load missing components
            'type': 'llama', # required so gguf loader knows what to do
            'gguf': 'mradermacher/Llama-3.2-1B-Instruct-Uncensored-i1-GGUF', # gguf repo
            'file': 'Llama-3.2-1B-Instruct-Uncensored.i1-Q4_0.gguf', # gguf file inside repo
        },
    }
    models_cls = {
        'qwen3_5': 'Qwen3_5ForConditionalGeneration',
        'qwen3_5_moe': 'Qwen3_5MoeForConditionalGeneration',
        'qwen3_vl': 'Qwen3VLForConditionalGeneration',
        'qwen2_5_vl': 'Qwen2_5_VLForConditionalGeneration',
        'qwen2_vl': 'Qwen2VLForConditionalGeneration',
        'mistral3': 'Mistral3ForConditionalGeneration',
        'gemma4': 'Gemma4ForConditionalGeneration',
    }

    # default = list(models)[1] # gemma-3-4b-it
    default = 'google/gemma-3-4b-it'
    supported = list(transformers.integrations.ggml.GGUF_CONFIG_MAPPING)
    t2i_prompt: str = textwrap.dedent('''\
        You are an expert AI image prompt engineer.
        You will receive a user prompt for image generation.
        Your sole job is to rewrite user inputs into highly detailed, visually rich prompts for image generation models.
        Improve the prompt by adding relevant visual specificity for composition, lighting, color, texture, and atmosphere.
        Keep the result faithful to the original prompt and the intended image.
        Do not add unrelated concepts, non-visual commentary, or fluff.
        ''')
    i2i_prompt: str = textwrap.dedent('''\
        You are an expert AI image prompt engineer.
        You will receive an image and a user prompt for editing or refinement.
        Your sole job is to rewrite user inputs into highly detailed, visually rich prompts for image generation models while taking the provided image into account.
        Improve the prompt with concrete visual detail that remains faithful to the image and edit intent.
        Keep the result grounded in image-generation language.
        Do not invent unrelated objects, actions, or concepts.
        ''')
    i2i_noprompt: str = textwrap.dedent('''\
        You are an expert AI image prompt engineer.
        You will receive an image and must create an editing prompt from the visible content.
        Your sole job is to create highly detailed, visually rich prompts for image generation models based on the provided image.
        Base the prompt on what is present in the image, its style, and its atmosphere.
        Use concrete, image-generation language and avoid unrelated fluff or commentary.
        ''')
    t2v_prompt: str = textwrap.dedent('''\
        You are an expert AI video prompt engineer.
        You will receive a user prompt for video generation.
        Your sole job is to rewrite user inputs into highly detailed, visually rich prompts for video generation models.
        Improve the prompt by adding relevant visual specificity for composition, lighting, color, texture, and atmosphere.
        Specify subject and camera movement, scene transitions, and temporal dynamics.
        Keep the result faithful to the original prompt and the intended video.
        Do not add unrelated concepts, non-visual commentary, or fluff.
        ''')
    i2v_prompt: str = textwrap.dedent('''\
        You are an expert AI video prompt engineer.
        You will receive an image and a user prompt for editing or refinement.
        Your sole job is to rewrite user inputs into highly detailed, visually rich prompts for video generation models while taking the provided image into account.
        Improve the prompt with concrete visual detail that remains faithful to the image and edit intent.
        Specify subject and camera movement, scene transitions, and temporal dynamics.
        Keep the result grounded in video-generation language.
        Do not invent unrelated objects, actions, or concepts.
        ''')
    i2v_noprompt: str = textwrap.dedent('''\
        You are an expert AI video prompt engineer.
        You will receive an image and must create a video prompt from the visible content.
        Your sole job is to create highly detailed, visually rich prompts for video generation models based on the provided image.
        Base the prompt on what is present in the image, its style, and its atmosphere.
        Specify subject and camera movement, scene transitions, and temporal dynamics.
        Use concrete, video-generation language and avoid unrelated fluff or commentary.
        ''')
    nsfw_ok: str = textwrap.dedent('''\
        NSFW, nudity and explicit content and language are allowed.
        ''')
    nsfw_no: str = textwrap.dedent('''\
        NSFW, nudity or explicit content and language are not allowed; if present, remove those elements from the enhanced prompt.
        ''')
    details_prompt: str = textwrap.dedent('''\
        Enhance the prompt with relevant image details.
        Identify the main subject(s), physical attributes, clothing if non-nude, and distinctive visual features.
        Describe the scene, environment, key objects, and overall mood or atmosphere.
        Be precise and efficient; avoid redundancy, abstract commentary, unrelated fluff, or instructions.
        Do not invent any objects, settings, or themes not implied by the input.
        Do not add era, background props, or atmosphere unless explicitly present in the prompt.
        ''')
    details_format: str = textwrap.dedent('''\
        Output exactly one enhanced prompt string.
        Do not add greetings, comments, explanations, follow-up questions, labels, formatting, or numbering.
        Do not include any extra prose or analysis.
        Start immediately with the prompt content.
        No stray tokens!
        ''')

    censored = ["i cannot", "i can't", "i am sorry", "against my programming", "i am not able", "i am unable", 'i am not allowed']

    max_delim_index: int = 60
    min_tokens: int = 0
    max_tokens: int = 256
    do_sample: bool = True
    temperature: float = 0.6
    repetition_penalty: float = 1.2
    top_k: int = 0
    top_p: float = 0.0
    thinking_mode: bool = False

    @staticmethod
    def get_model_choices():
        """Return list of display names for dropdown."""
        from .helpers import get_model_display_name
        return [get_model_display_name(repo) for repo in Options.models.keys()]

    @staticmethod
    def get_default_display():
        """Return display name for default model."""
        from .helpers import get_model_display_name
        return get_model_display_name(Options.default)
