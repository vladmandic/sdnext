# Change Log for SD.Next

## Update for 2026-02-11

- **Image manipulation**
  - use high-quality [sharpfin](https://github.com/drhead/Sharpfin) accelerated library  
    when available (cuda-only), thanks @CalamitousFelicitousness  
  - add support for [spandrel](https://github.com/chaiNNer-org/spandrel)  
    **upscaling** engine with suport for new upscaling model families  
  - add two new ai upscalers: *RealPLKSR NomosWebPhoto* and *RealPLKSR AnimeSharpV2*  
  - add two new **interpolation** methods: *HQX* and *ICB*  
- **Features**
  - pipelines: add **ZImageInpaint**, thanks @CalamitousFelicitousness  
  - add `--remote` command line flag that reduces client/server chatter and improves link stability  
    for long-running generates, useful when running on remote servers  
  - hires: allow using different lora in refiner prompt  
  - **nunchaku** models are now listed in networks tab as reference models  
    instead of being used implicitly via quantization, thanks @CalamitousFelicitousness  
- **UI**
  - ui: **themes** add *CTD-NT64Light* and *CTD-NT64Dark*, thanks @resonantsky  
  - ui: **gallery** add option to auto-refresh gallery, thanks @awsr  
- **Internal**
  - refactor: to/from image/tensor logic, thanks @CalamitousFelicitousness  
  - refactor: switch to `pyproject.toml` for tool configs  
  - refactor: reorganize `cli` scripts  
  - refactor: move tests to dedicated `/test/`  
  - update `lint` rules, thanks @awsr  
  - update `requirements`  
- **Fixes**
  - fix: handle `clip` installer doing unwanted `setuptools` update  
  - fix: cleanup for `uv` installer fallback  
  - fix: add metadata restore to always-on scripts  
  - fix: improve wildcard weights parsing, thanks @Tillerz  
  - fix: ui gallery cace recursive cleanup, thanks @awsr  
  - fix: `anima` model detection  
  - fix: lora unwanted unload  
  - fix: improve preview error handler  

## Update for 2026-02-04

### Highlights for 2026-02-04

Refresh release two weeks after prior release, yet we still somehow managed to pack in *~150 commits*!  
Highlights would be two new models: **Z-Image-Base** and **Anima**, *captioning* support for **tagger** models and a massive addition of new **schedulers**  
Also here are updates to `torch` and additional GPU archs support for `ROCm` backends, plus a lot of internal improvements and fixes.

[ReadMe](https://github.com/vladmandic/automatic/blob/master/README.md) | [ChangeLog](https://github.com/vladmandic/automatic/blob/master/CHANGELOG.md) | [Docs](https://vladmandic.github.io/sdnext-docs/) | [WiKi](https://github.com/vladmandic/automatic/wiki) | [Discord](https://discord.com/invite/sd-next-federal-batch-inspectors-1101998836328697867) | [Sponsor](https://github.com/sponsors/vladmandic)  

### Details for 2026-02-04

- **Models**
  - [Tongyi-MAI Z-Image Base](https://tongyi-mai.github.io/Z-Image-blog/)  
    yup, its finally here, the full base model of **Z-Image**  
  - [CircleStone Anima](https://huggingface.co/circlestone-labs/Anima)  
    2B anime optimized model based on a modified Cosmos-Predict, using Qwen3-0.6B as a text encoder  
- **Features**
  - **caption** tab support for Booru tagger models, thanks @CalamitousFelicitousness  
  - add SmilingWolf WD14/WaifuDiffusion tagger models, thanks @CalamitousFelicitousness  
  - support comments in wildcard files, using `#`  
  - support aliases in metadata skip params, thanks @CalamitousFelicitousness  
  - ui gallery improve cache cleanup and add manual option, thanks @awsr  
  - selectable options to add system info to metadata, thanks @Athari  
    see *settings -> image metadata*  
- **Schedulers**
  - schedulers documentation has new home: <https://vladmandic.github.io/sdnext-docs/Schedulers/>
  - add 13(!) new scheduler families
    not a port, but more of inspired-by [res4lyf](https://github.com/ClownsharkBatwing/RES4LYF) library  
    all schedulers should be compatible with both `epsilon` and `flow` prediction style!  
    *note*: each family may have multiple actual schedulers, so the list total is 56(!) new schedulers     
    - core family: *RES*
    - exponential: *DEIS, ETD, Lawson, ABNorsett*
    - integrators: *Runge-Kutta, Linear-RK, Specialized-RK, Lobatto, Radau-IIA, Gauss-Legendre*
    - flow: *PEC, Riemannian, Euclidean, Hyperbolic, Lorentzian, Langevin-Dynamics*
  - add 3 additional schedulers: *CogXDDIM, DDIMParallel, DDPMParallel*  
    not originally intended to be a general purpose schedulers, but they work quite nicely and produce good results  
  - image metadata: always log scheduler class used  
- **API**  
  - add `/sdapi/v1/xyz-grid` to enumerate xyz-grid axis options and their choices  
    see `/cli/api-xyzenum.py` for example usage  
  - add `/sdapi/v1/sampler` to get current sampler config  
  - modify `/sdapi/v1/samplers` to enumerate available samplers possible options  
    see `/cli/api-samplers.py` for example usage  
- **Internal**
  - tagged release history: <https://github.com/vladmandic/sdnext/tags>  
    each major for the past year is now tagged for easier reference  
  - **torch** update
    *note*: may cause slow first startup/generate  
    **cuda**: update to `torch==2.10.0`  
    **xpu**: update to `torch==2.10.0`  
    **rocm**: update to `torch==2.10.0`  
    **openvino**: update to `torch==2.10.0` and `openvino==2025.4.1`  
  - rocm: expand available gfx archs, thanks @crashingalexsan  
  - rocm: set `MIOPEN_FIND_MODE=2` by default, thanks @crashingalexsan  
  - relocate all json data files to `data/` folder  
    existing data files are auto-migrated on startup  
  - refactor and improve connection monitor, thanks @awsr  
  - further work on type consistency and type checking, thanks @awsr  
  - log captured exceptions  
  - improve temp folder handling and cleanup  
  - remove torch errors/warings on fast server shutdown  
  - add ui placeholders for future agent-scheduler work, thanks @ryanmeador  
  - implement abort system on repeated errors, thanks @awsr  
    currently used by lora and textual-inversion loaders  
  - update package requirements  
- **Fixes**
  - add video ui elem_ids, thanks @ryanmeador  
  - use base steps as-is for non sd/sdxl models  
  - ui css fixes for modernui  
  - support lora inside prompt selector  
  - framepack video save  
  - metadata save for manual saves  

## Update for 2026-01-22

Bugfix refresh

- add `SD_DEVICE_DEBUG` env variable to trace rocm/xpu/directml init failures  
- fix detailer double save  
- fix lora load when using peft/diffusers loader  
- fix rocm hipblaslt detection  
- fix image delete, thanks @awsr  
- fix `all_seeds` error  
- fix qwen settings typo, thanks @liutyi  
- improve `wrap_gradio` error handling  
- use refiner/detail steps as-is for non sd/sdxl models  

## Update for 2026-01-20

### Highlights for 2026-01-20

First release of 2026 brings quite a few new models: **Flux.2-Klein, Qwen-Image-2512, LTX-2-Dev, GLM-Image**  
There are also improvements to *SDNQ* quantization engine, updated *Prompt Enhance*, *Image Preview* and many others.  
Plus some significant under-the-hood changes to improve code coverage and quality which resulted in more than usual levels of bug-fixes and some ~330 commits!  
For full list of changes, see full changelog.

[ReadMe](https://github.com/vladmandic/automatic/blob/master/README.md) | [ChangeLog](https://github.com/vladmandic/automatic/blob/master/CHANGELOG.md) | [Docs](https://vladmandic.github.io/sdnext-docs/) | [WiKi](https://github.com/vladmandic/automatic/wiki) | [Discord](https://discord.com/invite/sd-next-federal-batch-inspectors-1101998836328697867) | [Sponsor](https://github.com/sponsors/vladmandic)  

### Details for 2026-01-20

- **Models**
  - [Flux.2 Klein](https://bfl.ai/blog/flux2-klein-towards-interactive-visual-intelligence)  
    Flux.2-Klein is a new family of compact models from BFL in *4B and 9B sizes* and avaialable as *destilled and base* variants  
    also includes are *sdnq prequantized variants*  
    *note*: 9B variant is [gated](https://vladmandic.github.io/sdnext-docs/Gated/)  
  - [Qwen-Image-2512](https://qwen.ai/blog?id=qwen-image-2512)  
    Qwen-Image successor, significantly reduces the AI-generated look and adds finer natural detailils and improved text rendering  
    available in both *original*, *sdnq-svd prequantized* and *sdnq-dynamic prequantized*  variants  
    thanks @CalamitousFelicitousness  
  - [LTX-2 19B Dev](https://ltx.io/model/ltx-2)  
    LTX-2 is a new very large 19B parameter video generation model from Lightricks using Gemma-3 text encoder  
    available for T2I/I2I workflows in original and sdnq prequantized variants  
    *note*: model is very sensitive to input params and will result in errors otherwise  
  - [GLM-Image](https://z.ai/blog/glm-image)  
    GLM-image is a new image generation model that adopts a hybrid autoregressive with diffusion decoder architecture  
    available in both *original* and *sdnq-dynamic prequantized* variants  
    thanks @CalamitousFelicitousness  
    *note*: model requires pre-release versions of `transformers` package:  
    > pip install --upgrade git+https://github.com/huggingface/transformers.git  
    > ./webui.sh --experimental  
  - [Nunchaku Z-Image Turbo](https://huggingface.co/nunchaku-tech/nunchaku-z-image-turbo)  
    nunchaku optimized z-image turbo  
- **Feaures**
  - **SDNQ**: add *dynamic* quantization method  
    sdnq can dynamically determine best quantization method for each module layer  
    slower to quantize on-the-fly, but results in better quality with minimal resource usage  
  - **SDNQ** now has *19 int* based and *69 float* based quantization types  
    *note*: not all are exposed via ui purely for simplicity, but all are available via api and scripts  
  - **wildcards**: allow weights, thanks @Tillerz  
  - **sampler**: add laplace beta schedule  
    results in better prompt adherence and smoother infills  
  - **prompt enhance**: improve handling and refresh ui, thanks @CalamitousFelicitousness  
    new models such moondream-3 and xiaomo-mimo  
    add support for *thinking* mode where model can reason about the prompt  
    add support for *vision* processing where prompt enhance can also optionally analyze input image  
    add support for *pre-fill* mode where prompt enhance can continue from existing caption  
  - **chroma**: add inpaint pipeline support  
  - **taesd preview**: support for more models, thanks @alerikaisattera  
  - **image ouput paths**: better handling of relative/absolute paths, thanks @CalamitousFelicitousness  
- **UI**
  - kanvas add send-to functionality  
  - kanvas improve support for standardui  
  - improve extensions tab layout and behavior, thanks @awsr  
  - indicate collapsed/hidden sections  
  - persistent panel minimize/maximize state  
  - gallery improve sorting behavior  
  - gallery implement prev/next navigation in full screen viewer, thanks @ryanmeador  
- **Internal**
  - **lora** native support by default will now skip text-encoder  
    can be enabled in *settings -> networks*
  - update core js linting to `eslint9`, thanks @awsr  
  - update modernui js linting to `eslint9`, thanks @awsr  
  - update kanvas js linting to `eslint9`, thanks @awsr  
  - update strong typing checks, thanks @awsr  
  - update reference models previews, thanks @liutyi  
  - update models specs page, thanks @alerikaisattera  
  - sdnq improvements  
  - startup sequence optimizations  
  - rocm/hip/hipblast detection and initialization improvements  
  - zluda detection and initialization improvements  
  - new env variable `SD_VAE_DEFAULT` to force default vae processing  
  - update `nunchaku==1.1.0`  
  - lora switch logic from force-diffusers to allow-native  
  - split `reference.json`  
  - print system env on startup  
  - disable fallback on models with custom loaders  
  - refactor triggering of prompt parser and set secondary prompts when needed  
  - refactor handling of seeds  
  - allow unsafe ssl context for downloads  
- **Fixes**
  - controlnet: controlnet with non-english ui locales  
  - core: add skip_keys to offloading logic, fixes wan frames mismatch, thanks @ryanmeador  
  - core: force model move on offload=none  
  - core: hidiffusion tracing  
  - core: hip device name detection  
  - core: reduce triton test verbosity  
  - core: switch processing class not restoring params  
  - extension tab: update checker, date handling, formatting etc., thanks @awsr  
  - lora force unapply on change  
  - lora handle null description, thanks @CalamitousFelicitousness  
  - lora loading when using torch without distributed support  
  - lora skip with strength zero  
  - lora: generate slowdown when consequtive lora-diffusers enabled  
  - model: google-genai auth, thanks @CalamitousFelicitousness  
  - model: improve qwen i2i handling  
  - model: kandinsky-5 image and video on non-cuda platforms  
  - model: meituan-longca-image-edit missing image param  
  - model: wan 2.2 i2v  
  - model: z-image single-file loader  
  - other: update civitai base models, thanks @trojaner  
  - ui: gallery save/delete  
  - ui: mobile auto-collapse when using side panel, thanks @awsr  
  - ui: networks filter by model type  
  - ui: networks icon/list view type switch, thanks @awsr  
  - vae: force align width/height to vae scale factor  
  - wildards with folder specification  

## Update for 2025-12-26

### Highlights for 2025-12-26

End of year release update, just two weeks after previous one, with several new models and features:
- Several new models including highly anticipated **Qwen-Image-Edit 2511** as well as **Qwen-Image-Layered**, **LongCat Image** and **Ovis Image**  
- New features including support for **Z-Image** *ControlNets* and *fine-tunes* and **Detailer** segmentation support  

[ReadMe](https://github.com/vladmandic/automatic/blob/master/README.md) | [ChangeLog](https://github.com/vladmandic/automatic/blob/master/CHANGELOG.md) | [Docs](https://vladmandic.github.io/sdnext-docs/) | [WiKi](https://github.com/vladmandic/automatic/wiki) | [Discord](https://discord.com/invite/sd-next-federal-batch-inspectors-1101998836328697867) | [Sponsor](https://github.com/sponsors/vladmandic)  

### Details for 2025-12-26

- **Models**
  - [LongCat Image](https://github.com/meituan-longcat/LongCat-Image) in *Image* and *Image Edit* variants  
    LongCat is a new 8B diffusion base model using Qwen-2.5 as text encoder  
  - [Qwen-Image-Edit 2511](https://huggingface.co/Qwen/Qwen-Image-Edit-2511) in *base* and *pre-quantized* variants  
    Key enhancements: mitigate image drift, improved character consistency, enhanced industrial design generation, and strengthened geometric reasoning ability  
  - [Qwen-Image-Layered](https://huggingface.co/Qwen/Qwen-Image-Layered) in *base* and *pre-quantized* variants  
    Qwen-Image-Layered, a model capable of decomposing an image into multiple RGBA layers  
    *note*: set number of desired output layers in *settings -> model options*  
  - [Ovis Image 7B](https://huggingface.co/AIDC-AI/Ovis-Image-7B)  
    Ovis Image is a new text-to-image base model based on Qwen3 text-encoder and optimized for text-rendering  
- **Features**
  - Google **Gemini** and **Veo** models support for both *Dev* and *Vertex* access methods  
    see [docs](https://vladmandic.github.io/sdnext-docs/Google-GenAI/) for details  
  - **Z-Image Turbo** support loading transformer file-tunes in safetensors format  
    as with any transformers/unet finetunes, place them then `models/unet`  
    and use **UNET Model** to load safetensors file as they are not complete models  
  - **Z-Image Turbo** support for **ControlNet Union**  
    includes 1.0, 2.0 and 2.1 variants  
  - **Detailer** support for segmentation models  
    some detection models can produce exact segmentation mask and not just box  
    to enable, set `use segmentation` option  
    added segmentation models: *anzhc-eyes-seg*, *anzhc-face-1024-seg-8n*, *anzhc-head-seg-8n*  
- **Internal**
  - update nightlies to `rocm==7.1`  
  - mark `python==3.9` as deprecated  
  - extensions improved status indicators, thanks @awsr  
  - additional type-safety checks, thanks @awsr  
  - add model info to ui overlay  
- **Wiki/Docs/Illustrations**  
  - update models page, thanks @alerikaisattera  
  - update reference models samples, thanks @liutyi  
- **Fixes**  
  - generate forever fix loop checks, thanks @awsr  
  - tokenizer expclit use for flux2, thanks @CalamitousFelicitousness  
  - torch.compile skip offloading steps  
  - kanvas css with standardui  
  - control input media with non-english locales  
  - handle embeds when on meta device  
  - improve offloading when model has manual modules  
  - ui section colapsible state, thanks @awsr  
  - ui filter by model type  

## Update for 2025-12-11

### Highlights for 2025-12-11

*What's new?*  
New native [kanvas](https://vladmandic.github.io/sdnext-docs/Kanvas/) module for image manipulation that fully replaces *img2img*, *inpaint* and *outpaint* controls, massive update to **Captioning/VQA** models and features  
New generation of **Flux.2** large image model, new **Z-Image** model that is creating a lot of buzz, new **Kandinsky 5 Lite** image model and new **Photoroom PRX** model  
And first cloud models with **Google Nano Banana** *2.5 Flash and 3.0 Pro* and **Google Veo** *3.1* video model  
Also new are **HunyuanVideo 1.5** and **Kandinsky 5 Pro** video models  
Plus a lot of internal improvements and fixes  

![Screenshot](https://github.com/user-attachments/assets/54b25586-b611-4d70-a28f-ee3360944034)

[ReadMe](https://github.com/vladmandic/automatic/blob/master/README.md) | [ChangeLog](https://github.com/vladmandic/automatic/blob/master/CHANGELOG.md) | [Docs](https://vladmandic.github.io/sdnext-docs/) | [WiKi](https://github.com/vladmandic/automatic/wiki) | [Discord](https://discord.com/invite/sd-next-federal-batch-inspectors-1101998836328697867) | [Sponsor](https://github.com/sponsors/vladmandic)  

### Details for 2025-12-11

- **Models**
  - [Black Forest Labs FLUX.2 Dev](https://bfl.ai/blog/flux-2) and prequantized variation [SDNQ-SVD-Uint4](https://huggingface.co/Disty0/FLUX.2-dev-SDNQ-uint4-svd-r32)  
    **FLUX.2-Dev** is a brand new model from BFL and uses large 32B DiT together with Mistral 24B as text encoder  
    model is available for text, image and edit tasks and can optionally use control input as second input image  
    this is a very large model at ~100GB, so use of prequantized model at ~32GB is strongly advised
    using prequant version and default offloading, model runs on GPUs with ~20GB  
    *note*: model is [gated](https://vladmandic.github.io/sdnext-docs/Gated/)  
  - [Z-Image Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) and prequantized variation [SDNQ-SVD-Uint4](https://huggingface.co/Disty0/Z-Image-Turbo-SDNQ-uint4-svd-r32)  
    **Z-Image** is a powerful and highly efficient image generation model with 6B parameters and using Qwen-3 as text encoder  
    unlike most of new models that are far larger, Z-Image architecture allows it to run with good performance even on mid-range hardware  
    *note*: initial release is *Turbo* variant only with *Base* and *Edit* variants to follow  
  - [Kandinsky 5.0 Lite]() is a new 6B model using Qwen-2.5 as text encoder  
    it comes in text-to-image and image-edit variants  
  - **Google Gemini Nano Banana** [2.5 Flash](https://blog.google/products/gemini/gemini-nano-banana-examples/) and [3.0 Pro](https://deepmind.google/models/gemini-image/pro/) 
    first cloud-based model directly supported in SD.Next UI  
    *note*: need to set `GOOGLE_API_KEY` environment variable with your key to use this model  
  - [Photoroom PRX 1024 Beta](https://huggingface.co/Photoroom/prx-1024-t2i-beta)  
    PRX (Photoroom Experimental) is a small 1.3B parameter t2i model trained entirely from scratch, it uses T5-Gemma text-encoder  
- **Video**  
  - [HunyuanVideo 1.5](https://huggingface.co/tencent/HunyuanVideo-1.5) in T2V and I2V variants, both standard and distilled and both 720p and 480p resolutions  
    **HunyuanVideo 1.5** improves upon previous 1.0 version with better quality and higher resolution outputs, it uses Qwen2.5-VL text-encoder  
    distilled variants provide faster generation with slightly reduced quality  
  - [Kandinsky 5.0 Pro Video](https://huggingface.co/kandinskylab/Kandinsky-5.0-T2V-Pro-sft-5s-Diffusers) in T2V and I2V variants  
    larger 19B (and more powerful version) of previously released Lite 2B models  
  - [Google Veo 3.1](https://gemini.google/us/overview/video-generation/) for T2V and I2V workflows  
    *note*: need to set `GOOGLE_API_KEY` environment variable with your key to use this model  
- **Kanvas**: new module for native canvas-based image manipulation  
  kanvas is a full replacement for *img2img, inpaint and outpaint* controls  
  see [docs](https://vladmandic.github.io/sdnext-docs/Kanvas/) for details  
  *experimental*: report any feedback in master [issue](https://github.com/vladmandic/sdnext/issues/4358)  
- **Captioning** and **VQA: Visual Question & Answer**  
  massive update to both features and supported models, thanks @CalamitousFelicitousness  
  models:  
  - additional `mooondream-2` features  
  - support for `moondream-3-preview`  
  - support for `qwen3-vl` with thinking  
  - additional `gemma-3-vl` finetunes  
  - support for `XiaomiMiMo`  
  ui:  
  - ability to annotate actual image, not just generate captions/answers  
    e.g. actualy mark detected regions/points  
  features:  
  - ui indicator of model capabilities  
  - support for *prefill* style of prompting/answering  
  - support for *reasoning* mode for supported models  
    with option to output answer-only or reasoning-process   
  - additional debug logging  
- **Other Features**
  - **wildcards**: allow recursive inline wildcards using curly braces syntax  
  - **sdnq**: simplify pre-quantization saved config  
  - **attention**: additional torch attention settings  
  - **lora**: separate fuse setting for native-vs-diffuser implementations  
  - **auth**: strong-enforce auth check on all api endpoints  
  - **amdgpu**: prefer rocm-on-windows over zluda  
  - **amdgpu**: improve rocm-on-windows installer  
  - **sdnq**: improve dequant logic  
  - **gallery**: significant performance improvements, thanks @awsr  
- **API**
  - `/control` endpoint is now fully compatible with scripts  
  - `/control` additional params to to control *xyz grid*  
    see `cli/api-xyz.py` for simple example  
  - `/detailers` new endpoint to list available detailers, both built-in and any custom downloaded  
  - `/face-restorers` expanded to list model folders  
- **Internal**
  - python: set 3.10 as minimum supported version  
  - sdnq: multiple improvements to quantization and dequantization logic
  - torch: update to `torch==2.9.1` for *cuda, ipex, openvino, rocm* backends
  - attention: refactor attention handling  
  - scripts: remove obsolete video scripts  
  - lint: update global lint rules  
  - chrono: switch to official pipeline  
  - pipeline: add optional preprocess and postprocess hooks  
  - auth: wrap all internal api calls with auth check and use token when possible  
  - installer: reduce requirements  
  - installer: auto-restart on self-update  
  - server: set correct mime-types  
  - sdnq: unconditional register on startup  
  - python: start work on future-proofing for modern python versions, thanks @awsr  
  - nunchaku: update to `1.0.2`  
  - lint: add rules for run-on-windows  
  - gallery: setting to enable/disable client-side caching, thanks @awsr  
  - gallery: faster thumbnail generation, thanks @awsr  
  - gallery: purge old thumbnails, thanks @awsr  
- **Docs**
  - update supported models table with VAE information, thanks @alerikaisattera
- **Fixes**
  - xyz-grid: improve parsing of axis lists, thanks @awsr  
  - hires: strength save/load in metadata, thanks @awsr  
  - imgi2img: fix initial scale tab, thanks @awsr  
  - img2img: fix restoring refine sampler from metadata, thanks @awsr
  - log: client log formatting, thanks @awsr  
  - rocm: check if installed before forcing install  
  - pony-v7: fix text-encoder  
  - detailer: with face-restorers  
  - detailer: using lora in detailer prompt  
  - detailer: fail on unsupported models instead of corrputing results  
  - ui: fix collapsible panels  
  - svd: fix stable-video-diffusion dtype mismatch  
  - animatediff: disable sdnq if used  
  - lora: restore pipeline type if reload/recompile needed  
  - process: improve send-to functionality  
  - control: safe load non-sparse controlnet  
  - control: fix marigold preprocessor with bfloat16  
  - auth: fix password being shown in clear text during login  
  - firefox: remove obsolete checks, thanks @awsr  
  - runai streamer: cleanup logging, thanks @CalamitousFelicitousness  
  - gradio: event handlers, thanks @awsr  

## Update for 2025-11-06

### Highlights for 2025-11-06

Service pack release that handles critical issues and improvements for **ROCm-on-Windows** and **ZLUDA** backends  
Also included are several new features, notably improvements to **detailer** and ability to run [SD.Next](https://github.com/vladmandic/sdnext) with specific modules disabled  
And new video model, **nVidia SANA 2B**  

![Screenshot](https://github.com/user-attachments/assets/d6119a63-6ee5-4597-95f6-29ed0701d3b5)

[ReadMe](https://github.com/vladmandic/automatic/blob/master/README.md) | [ChangeLog](https://github.com/vladmandic/automatic/blob/master/CHANGELOG.md) | [Docs](https://vladmandic.github.io/sdnext-docs/) | [WiKi](https://github.com/vladmandic/automatic/wiki) | [Discord](https://discord.com/invite/sd-next-federal-batch-inspectors-1101998836328697867) | [Sponsor](https://github.com/sponsors/vladmandic)  

### Details for 2025-11-06

- **Models**
  - [SANA Video_2B_480p T2V](https://huggingface.co/Efficient-Large-Model/SANA-Video_2B_480p_diffusers) is a small 2B ultra-efficient diffusion model  
    designed for rapid generation of high-quality videos and uses Gemma2 text encoder  
- **Features**
  - **ROCm for Windows** switch to using **TheRock** `torch` builds when available  
    recommended to run: `webui --use-rocm --reinstall`  
  - **ZLUDA** improve detection and handling of unsupported GPUs  
    recommended to run: `webui --use-zluda --reinstall`  
  - **detailer**  
    optional include detection image to output results  
    optional sort detection objects left-to-right for improved prompt consistency  
    enable multi-subject and multi-model prompts  
  - **disable modules**  
    ability to disable parts of the app  
    useful for custom deployments where some features are not desired  
    *note*: this doesn't just hide it from user, it completely disables the code paths  
    use `--disable x,y,z`  
    possible values:  
    - main tabs: *control,txt2img,img2img,video,extras,caption,gallery*  
    - aside tabs: *extensions,models,info,update,history,monitor,onnx,system,networks,logs*  
    - special: *settings,config* (hidden instead of disabled)  
  - **wildcards**: add inline processing using curly braces syntax  
  - add setting to control `cudnn` enable/disable  
    *note*: this can also be used to enable/disable `MIOpen` on ROCm backends  
  - change `vlm` beams to 1 by default for faster response  
  - **controlnet** allow processor to keep aspect-ratio for override images based on i2i or t2i resolution  
  - **networks** info details now displays image metadata from preview image  
  - **networks** new model previews, thanks @liutyi  
- **Fixes**
  - zluda: test and disable MIOpen as needed  
  - qwen: improve lora compatibility  
  - chrono: transformers handling  
  - chrono: extract last frame  
  - chrono: add vae scale override, thanks @CalamitousFelicitousness  
  - runai: improve streamer integration  
  - transformers: `dtype` use new syntax  
  - rocm: possible endless loop during hip detection  
  - rocm: auto-disable `miopen` for gfx120x  
  - detailer: better handling of settings, thanks @awsr  
  - installer: cleanup `--optional`  
  - hires: guard against multi-controlnet  
  - inpaint: fix init  
  - version: detection when cloned with .git suffix, thanks @awsr  
  - sdnq: init on video model load  
  - model type: detection  
  - model type: add tracing to model detection  
  - settings: guard against non-string values, thanks @awsr  
  - ui: wait for server options to be ready before initializing ui  
  - ui: fix full-screen image viewer buttons with non-standard ui theme  
  - ui: control tab show override section  
  - ui: mobile layout for video tab  
  - ui: increase init timeout  
  - video: save to subfolder  
  - taesd: warn on long decode times  
  - metadata: keep exif on thumbnail generation  
  - wildcard: obey seed for reproducible results  
  - sageattention: handle possible triton issues on some nvidia gpus, thanks @CalamitousFelicitousness  

## Update for 2025-10-31

### Highlights for 2025-10-31

Less than 2 weeks since last release, here's a service-pack style update with a lot of fixes and improvements:
- Reorganization of **Reference Models** into *Base, Quantized, Distilled and Community* sections for easier navigation  
  and introduction of optimized **pre-quantized** variants for many popular models - use this as your quick start!  
- New models:  
  **HunyuanImage 2.1** capable of 2K images natively, **HunyuanImage 3.0** large unified multimodal autoregressive model,  
  **ChronoEdit** that re-purposes temporal consistency of generation for image editing  
  **Pony 7** based on AuraFlow architecture, **Kandinsky 5** 10s video models  
- New **offline mode** to use previously downloaded models without internet connection  
- Optimizations to **WAN-2.2** given its popularity  
  plus addition of native **VAE Upscaler** and optimized **pre-quantized** variants  
- New SOTA model loader using **Run:ai streamer**  
- Updates to `rocm` and `xpu` backends  
- Fixes, fixes, fixes... too many to list here!  

![Screenshot](https://github.com/user-attachments/assets/d6119a63-6ee5-4597-95f6-29ed0701d3b5)

[ReadMe](https://github.com/vladmandic/automatic/blob/master/README.md) | [ChangeLog](https://github.com/vladmandic/automatic/blob/master/CHANGELOG.md) | [Docs](https://vladmandic.github.io/sdnext-docs/) | [WiKi](https://github.com/vladmandic/automatic/wiki) | [Discord](https://discord.com/invite/sd-next-federal-batch-inspectors-1101998836328697867) | [Sponsor](https://github.com/sponsors/vladmandic)  

### Details for 2025-10-31

- **Reference** networks section is now split into actual *Base* models plus:  
  - **Quantized**: pre-quantized variants of the base models using SDNQ-SVD quantization for optimal quality and smallest possible resource usage  
    examples: *FLUX.1-Dev/Krea/Kontext/Schnell, Qwen-Image/Edit/2509, Chroma1-HD, WAN-2.2-A44B, etc.*  
    *note*: pre-quantized *WAN-2.2-14B* is also available in video models and runs with only 12GB VRAM!  
  - **Distilled**: distilled variants of base models  
    examples: *Turbo, Lightning, Lite, SRPO, Distill, Pruning, etc.*  
  - **Community**: community highlights  
    examples: *Tempest, Juggernaut, Illustrious, Pony, NoobAI, etc.*  
    and all reference models have new preview images, thanks @liutyi  
- **Models Reference**  
  - [Tencent HunyuanImage 2.1](https://huggingface.co/tencent/HunyuanImage-2.1) in *full*, *distilled* and *refiner* variants  
    *HunyuanImage-2.1* is a large (51GB) T2I model capable of natively generating 2K images and uses Qwen2.5 + T5 text-encoders and 32x VAE  
  - [Tencent HunyuanImage 3.0](https://huggingface.co/tencent/HunyuanImage-3.0) in [pre-quant](https://huggingface.co/Disty0/HunyuanImage3-SDNQ-uint4-svd-r32) only variant due to massive size  
    *HunyuanImage 3.0* is very large at 47GB pre-quantized (oherwise its 157GB) that unifies multimodal understanding and generation within an autoregressive framework  
  - [nVidia ChronoEdit](https://huggingface.co/nvidia/ChronoEdit-14B-Diffusers)  
    *ChronoEdit* is a 14B image editing model based on *WAN*  
    this model reframes image editing as a video generation task, using input and edited images as start/end frames to leverage pretrained video models with temporal consistency  
    to extend temporal consistency for image editing, set *settings -> model options -> chrono temporal steps* to desired number of temporaly reasoning steps  
  - [Kandinsky 5 Lite 10s](https://huggingface.co/ai-forever/Kandinsky-5.0-T2V-Lite-sft-10s-Diffusers') in *SFT, CFG-distilled and Steps-distilled* variants  
    second series of models in *Kandinsky5* series is T2V model optimized for 10sec videos and uses Qwen2.5 text encoder  
  - [Pony 7](https://huggingface.co/purplesmartai/pony-v7-base)  
    Pony 7 steps in a different direction from previous Pony models and is based on AuraFlow architecture and UMT5 encoder  
- **Models Auxiliary**  
  - [Qwen 3-VL](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct) VLM for interrogate and prompt enhance, thanks @CalamitousFelicitousness  
    this includes *2B, 4B and 8B* variants  
  - [WAN Asymettric Upscale](https://huggingface.co/spacepxl/Wan2.1-VAE-upscale2x)  
    available as general purpose upscaler that can be used during standard workflow or process tab  
    available as VAE for compatible video models: *WAN-2.x-14B, SkyReels-v2* models  
  - [Apple DepthPro](https://huggingface.co/apple/DepthPro) controlnet processor, thanks @nolbert82  
  - [LibreFlux controlnet](https://huggingface.co/neuralvfx/LibreFlux-ControlNet) segmentation controlnet for FLUX.1  
- **Features**
  - **offline mode**: enable in *settings -> hugginface*  
    enables fully offline mode where previously downloaded models can be used as-is  
    *note*: must be enabled only after all packages have been installed and model has been run online at least once  
  - **model load**: SOTA method using nVidia's [Run:ai streamer](https://github.com/run-ai/runai-model-streamer)  
    enable in *settings -> model options -> runai streamer*  
    applies to *diffusers, transformers and sdnq* loaders, note this is linux-only feature  
    *experimental* but shows significant model load speedups, 20-40% depending on model and hardware  
- **Backend**
  - switch to `torch==2.9` for *ipex, rocm and openvino*  
  - switch to `rocm==7.0` for nightlies  
  - log `triton` availability on startup  
  - add `xpu` stats in gpu monitor  
- **Other**
  - improved **SDNQ SVD** and low-bit matmul performance  
  - reduce RAM usage on model load using **SDNQ SVD**
  - change default **schedulers** for sdxl  
  - warn on `python==3.9` end-of-life and `python==3.10` not actively supported  
  - **scheduler** add base and max shift parameters for flow-matching samplers  
  - enhance `--optional` flag to pre-install optional packages  
  - add `[lora]` to recognized filename patterns  
  - when using **shared-t5** *(default)*, it will load standard or pre-quant depending on model  
  - enhanced LoRA support for **Wan-2.2-14B**  
  - log available attention mechanisms on startup  
  - support for switching back-and-forth **t2i** and **t2v** for *wan-2.x* models  
  - control `api` cache controlnets  
  - additional model modules **deduplication** for both normal and pre-quant models: *umt5, qwen25-vl*  
- **Fixes**
  - startup error with `--profile` enabled if using `--skip`  
  - restore orig init image for each batch sequence  
  - fix modernui hints layout  
  - fix `wan-2.2-a14b` stage selection  
  - fix `wan-2.2-5b` vae decode  
  - disabling live preview should not disable progress updates  
  - video tab create `params.txt` with metadata  
  - fix full-screen image-viewer toolbar actions with control tab  
  - improve filename sanitization  
  - lora auto-detect low/high stage if not specified  
  - lora disable fuse on partially applied network  
  - fix networks display with extended characters, thanks @awsr  
  - installer handle different `opencv` package variants  
  - fix using pre-quantized shared-t5  
  - fix `wan-2.2-14b-vace` single-stage exectution  
  - fix `wan-2.2-5b` tiled vae decode  
  - fix `controlnet` loading with quantization  
  - video use pre-quantized text-encoder if selected model is pre-quantized  
  - handle sparse `controlnet` models  
  - catch `xet` warnings  
  - avoid unnecessary pipe variant switching  
  - validate pipelines on import  
  - fix `nudenet` process tab operations  
  - `controlnet` input validation  
  - log metadata keys that cannot be applied  
  - fix `framepack` with image input  

## Update for 2025-10-18

- **Models**
  [Kandinsky 5 Lite](https://huggingface.co/ai-forever/Kandinsky-5.0-T2V-Lite-sft-5s-Diffusers') in *SFT, CFG-distilled and Steps-distilled* variants  
  first model in Kandinsky5 series is T2V model optimized for 5sec videos and uses Qwen2.5 text encoder  
- **Fixes**
  - ROCm-on-Windows additional checks  
  - SDNQ-SVD fallback on incompatible layers  
  - Huggingface model download  
  - Video implement dynamic and manual sampler shift  
  - Fix interrupt batch processing  
  - Delay import of control processors until used  
  - Fix tiny VAE with batched results  
  - Fix CFG scale not added to metadata and set valid range to >=1.0  
- **Other**
  - Optimized Video tab layout  
  - Video enable VAE slicing and framewise decoding when possible  
  - Detect and log `flash-attn` and `sageattention` if installed  
  - Remove unused UI settings  

## Update for 2025-10-17

### Highlights for 2025-10-17

It's been a month since the last release and number of changes is yet again massive with over 300 commits!  
Highlight are:  
- **Torch**: ROCm on Windows for AMD GPUs  
  if you have a compatible GPU, performance gains are significant!  
- **Models**:  
  a lot of new stuff with **Qwen-Image-Edit** including multi-image edits and distilled variants,  
  new **Flux**, **WAN**, **LTX**, **HiDream** variants, expanded **Nunchaku** support and new SOTA upscaler with **SeedVR2**  
  plus improved video support in general, including new methods of video encoding  
- **Quantization**:  
  new **SVD**-style quantization using SDNQ offers almost zero-loss even with **4bit** quantization  
  and now you can also test your favorite quantization on-the-fly and then save/load model for future use  
- Other: support for **Huggingface** mirrors, changes to installer to prevent unwanted `torch-cpu` operations, improved VAE previews, etc.  

![Screenshot](https://github.com/user-attachments/assets/d6119a63-6ee5-4597-95f6-29ed0701d3b5)

[ReadMe](https://github.com/vladmandic/automatic/blob/master/README.md) | [ChangeLog](https://github.com/vladmandic/automatic/blob/master/CHANGELOG.md) | [Docs](https://vladmandic.github.io/sdnext-docs/) | [WiKi](https://github.com/vladmandic/automatic/wiki) | [Discord](https://discord.com/invite/sd-next-federal-batch-inspectors-1101998836328697867) | [Sponsor](https://github.com/sponsors/vladmandic)  

### Details for 2025-10-17

- **Models**
  - [WAN 2.2 14B VACE](https://huggingface.co/alibaba-pai/Wan2.2-VACE-Fun-A14B)  
    available for *text-to-image* and *text-to-video* and *image-to-video* workflows  
  - [Qwen Image Edit 2509](https://huggingface.co/Qwen/Qwen-Image-Edit-2509) and [Nunchaku Qwen Image Edit 2509](https://huggingface.co/nunchaku-tech/nunchaku-qwen-image-edit-2509)  
    updated version of Qwen Image Edit with improved image consistency  
  - [Qwen Image Pruning](https://huggingface.co/OPPOer/Qwen-Image-Pruning) and [Qwen Image Edit Pruning](https://huggingface.co/OPPOer/Qwen-Image-Edit-Pruning)  
    pruned versions of Qwen with 13B params instead of 20B, with some quality tradeoff  
  - [Tencent FLUX.1 Dev SRPO](https://huggingface.co/tencent/SRPO)  
    SRPO is trained by Tencent with specific technique: directly aligning the full diffusion trajectory with fine-grained human preference  
  - [Nunchaku SDXL](https://huggingface.co/nunchaku-tech/nunchaku-sdxl) and [Nunchaku SDXL Turbo](https://huggingface.co/nunchaku-tech/nunchaku-sdxl-turbo)  
    impact of nunchaku engine on unet-based model such as sdxl is much less than on a dit-based models, but its still significantly faster than baseline  
    note that nunchaku optimized and pre-quantized unet is replacement for base unet, so its only applicable to base models, not any of fine-tunes  
    *how to use*: enable nunchaku in settings -> quantization and then load either sdxl-base or sdxl-base-turbo reference models  
  - [HiDream E1.1](https://huggingface.co/HiDream-ai/HiDream-E1-1)  
    updated version of HiDream-E1 image editing model  
  - [LTXVideo 0.9.8](https://huggingface.co/Lightricks/LTX-Video-0.9.8-13B-distilled)  
    updated version of LTXVideo t2v/i2iv model  
  - [SeedVR2](https://iceclear.github.io/projects/seedvr/)  
    originally designed for video restoration, seedvr works great for image detailing and upscaling!  
    available in 3B, 7B and 7B-sharp variants, use as any other upscaler!  
    note: seedvr is a very large model (6.4GB and 16GB respectively) and not designed for lower-end hardware, quantization is highly recommended  
    note: seedvr is highly sensitive to its cfg scale, set in *settings -> postprocessing*  
    lower values will result in smoother output while higher values add details  
  - [X-Omni SFT](https://x-omni-team.github.io/)  
    *experimental*: X-omni is a transformer-only discrete auto-regressive image generative model trained with reinforcement learning  
- **Features**
  - **Model save**: ability to save currently loaded model as a new standalone model  
    why? SD.Next always prefers to start with full model and quantize on-demand during load  
    however, when you find your exact preferred quantization settings that work well for you,  
    saving such model as a new model allows for faster loads and reduced disk space usage  
    so its best of both worlds: you can experiment and test different quantization methods and once you find the one that works for you, save it as a new model  
    saved models appear in network tab as normal models and can be loaded as such  
    available in *models* tab  
  - [Qwen Image-Edit](https://huggingface.co/Qwen/Qwen-Image-Edit-2509) multi-image editing
    requires qwen-image-edit-2509 or its variant as multi-image edits are not available in original qwen-image  
    in ui control tab: inputs -> separate init image  
    add image for *input media* and *control media*  
    can be 
  - [Cache-DiT](https://github.com/vipshop/cache-dit)  
    cache-dit is a unified, flexible and training-free cache acceleration framework  
    compatible with many dit-based models such as FLUX.1, Qwen, HunyuanImage, Wan2.2, Chroma, etc.  
    enable in *settings -> pipeline modifiers -> cache-dit*  
  - [Nunchaku Flux.1 PulID](https://nunchaku.tech/docs/nunchaku/python_api/nunchaku.pipeline.pipeline_flux_pulid.html)  
    automatically enabled if loaded model is FLUX.1 with Nunchaku engine enabled and when PulID script is enabled  
  - **Huggingface mirror** in *settings -> huggingface*  
    if you're working from location with limited access to huggingface, you can now specify a mirror site  
    for example enter, `https://hf-mirror.com`  
- **Compute**
  - **ROCm** for Windows  
    support for both official torch preview release of `torch-rocm` for windows and **TheRock** unofficial `torch-rocm` builds for windows  
    note that rocm for windows is still in preview and has limited gpu support, please check rocm docs for details  
  - **DirectML** warn as *end-of-life*  
    `torch-directml` received no updates in over 1 year and its currently superseded by `rocm` or `zluda`  
  - command line params `--use-zluda` and `--use-rocm` will attempt desired operation or fail if not possible  
    previously sdnext was performing a fallback to `torch-cpu` which is not desired  
  - **installer** if `--use-cuda` or `--use-rocm` are specified and `torch-cpu` is installed, installer will attempt to reinstall correct torch package  
  - **installer** warn if *cuda* or *rocm* are available and `torch-cpu` is installed  
  - support for `torch==2.10-nightly` with `cuda==13.0`  
- **Extensions**  
  - [Agent-Scheduler](https://github.com/SipherAGI/sd-webui-agent-scheduler)  
    was a high-value built-in extension, but it has not been maintained for 1.5 years  
    it also does not work with control and video tabs which are the core of sdnext nowadays  
    so it has been removed from built-in extensions: manual installation is still possible  
  - [DAAM: Diffusion Attentive Attribution Maps](https://github.com/castorini/daam)  
    create heatmap visualizations of which parts of the prompt influenced which parts of the image  
    available in scripts for sdxl text-to-image workflows  
- **Offloading**
  - improve offloading for pipelines with multiple stages such as *wan-2.2-14b*  
  - add timers to measure onload/offload times during generate  
  - experimental offloading using `torch.streams`  
    enable in settings -> model offloading  
  - new feature to specify which models types not to offload  
    in *settings -> model offloading -> model types not to offload*  
- **UI**
  - **connection monitor**  
    main logo in top-left corner now indicates server connection status and hovering over it shows connection details  
  - separate guidance and detail sections  
  - networks ability to filter lora by base model version  
  - add interrogate button to input images  
  - disable spellchecks on all text inputs  
- **SDNQ**
  - add `SVDQuant` quantization method support  
  - make sdnq scales compatible with balanced offload  
  - add int8 `matmul` support for RDNA2 GPUs via triton  
  - improve int8 `matmul` performance on Intel GPUs  
- **Other**
  - server will note when restart is recommended due to package updates  
  - **interrupt** will now show last known preview image  
    *keep incomplete* setting is now *save interrupted*  
  - **logging** enable `debug`, `docs` and `api-docs` by default  
  - **logging** add detailed ram/vram utilization info to log  
    logging frequency can be specified using `--monitor x` command line param, where x is number of seconds  
  - **ipex** simplify internal implementation  
  - refactor to use new libraries  
  - styles and wildcards now use same seed as main generate for reproducible results  
  - **api** new endpoint POST `/sdapi/v1/civitai` to trigger civitai models metadata update  
    accepts optional `page` parameter to search specific networks page  
  - **reference models** additional example images, thanks @liutyi  
  - **reference models** add model size and release date, thanks @alerikaisattera  
  - **video** support for configurable multi-stage models such as WAN-2.2-14B  
  - **video** new LTX model selection  
  - replace `pynvml` with `nvidia-ml-py` for gpu monitoring  
  - update **loopback** script with radon seed option, thanks @rabanti  
  - **vae** slicing enable for *lowvram/medvram*, tiling for *lowvram*, both disabled otherwise  
  - **attention** remove split-attention and add explicitly attention slicing enable/disable option  
    enable in *settings -> compute settings*  
    can be combined with sdp, enabling may improve stability when used on iGPU or shared memory systems  
  - **nunchaku** update to `1.0.1` and enhance installer  
  - **xyz-grid** add guidance section  
  - **preview** implement configurable layers for WAN, Qwen, HV  
  - update swagger `/docs` endpoint style  
  - add `[epoch]` to filename template  
  - starting `[seq]` for filename template is now higher of largest previous sequence or number of files in folder  
- **Video**
  - use shared **T5** text encoder for video models when possible  
  - use shared **LLama** text encoder for video models when possible  
  - unified video save code across all video models  
    also avoids creation of temporary files for each frame unless user wants to save them  
  - unified prompt enhance code across all video models  
  - add job state tracking for video generation  
  - fix quantization not being applied on load for some models  
  - improve offloading for **ltx** and **wan**  
  - fix model selection in **ltx** tab  
- **Experimental**
  - `new` command line flag enables new `pydantic` and `albumentations` packages  
  - **modular pipelines**: enable in *settings -> model options*  
    only compatible with some pipelines, invalidates preview generation  
  - **modular guiders**: automatically used for compatible pipelines when *modular pipelines* is enabled  
    allows for using many different guidance methods:  
    *CFG, CFGZero, PAG, APG, SLG, SEG, TCFG, FDG*  
- **Wiki**
  - updates to *AMD-ROCm, ZLUDA, LoRA, DirectML, SDNQ, Quantization, Prompting, LoRA* pages  
  - new *Stability-Matrix* page  
- **Fixes**
  - **Microsoft Florence 2** both base and large variants  
    *note* this will trigger download of the new variant of the model, feel free to delete older variant in `huggingface` folder  
  - **MiaoshouAI PromptGen** 1.5/2.0 in both base and large variants  
  - fix prompt scheduling, thanks @nolbert82  
  - ui: fix image metadata display when switching selected image in control tab  
  - framepack: add explicit hf-login before framepack load  
  - framepack: patch solver for unsupported gpus  
  - benchmark: remove forced sampler from system info benchmark  
  - xyz-grid: fix xyz grid with random seeds  
  - reference: fix download for sd15/sdxl reference models  
  - fix checks in init/mask image decode  
  - fix hf token with extra chars  
  - image viewer refocus on gallery after returning from full screen mode  
  - fix attention guidance metadata save/restore  
  - vae preview add explicity cuda.sync  

## Update for 2025-09-15

### Highlights for 2025-09-15

*What's new*? Big one is that we're (*finally*) switching the default UI to **ModernUI**, for both desktop and mobile use!  
**StandardUI** is still available and can be selected in settings, but ModernUI is now the default for new installs  

*What's else*? **Chroma** is in its final form, there are several new **Qwen-Image** variants and **Nunchaku** hit version 1.0!  
Also, there are quite a few offloading improvements and many quality-of-life changes to UI and overall workflows  
And check out new **history** tab in the right panel, it now shows visualization of entire processing timeline!  

![Screenshot](https://github.com/user-attachments/assets/d6119a63-6ee5-4597-95f6-29ed0701d3b5)

[ReadMe](https://github.com/vladmandic/automatic/blob/master/README.md) | [ChangeLog](https://github.com/vladmandic/automatic/blob/master/CHANGELOG.md) | [Docs](https://vladmandic.github.io/sdnext-docs/) | [WiKi](https://github.com/vladmandic/automatic/wiki) | [Discord](https://discord.com/invite/sd-next-federal-batch-inspectors-1101998836328697867) | [Sponsor](https://github.com/sponsors/vladmandic)  

### Details for 2025-09-15

- **Models**
  - **Chroma** final versions: [Chroma1-HD](https://huggingface.co/lodestones/Chroma1-HD), [Chroma1-Base](https://huggingface.co/lodestones/Chroma1-Base) and [Chroma1-Flash](https://huggingface.co/lodestones/Chroma1-Flash)  
  - **Qwen-Image** [InstantX ControlNet Union](https://huggingface.co/InstantX/Qwen-Image-ControlNet-Union) support  
    *note* qwen-image is already a very large model and controlnet adds 3.5GB on top of that so quantization and offloading are highly recommended!  
  - [Qwen-Lightning-Edit](https://huggingface.co/vladmandic/Qwen-Lightning-Edit) and [Qwen-Image-Distill](https://huggingface.co/SahilCarterr/Qwen-Image-Distill-Full) variants  
  - **Nunchaku** variants of [Qwen-Image-Lightning](https://huggingface.co/nunchaku-tech/nunchaku-qwen-image), [Qwen-Image-Edit](https://huggingface.co/nunchaku-tech/nunchaku-qwen-image-edit), [Nunchaku-Qwen-Image-Edit-Lightning](https://huggingface.co/nunchaku-tech/nunchaku-qwen-image-edit)
  - **Nunchaku** variant of [Flux.1-Krea-Dev](https://huggingface.co/nunchaku-tech/nunchaku-flux.1-krea-dev)  
    if you have a compatible nVidia GPU, Nunchaku is the fastest quantization & inference engine  
  - [HunyuanDiT ControlNet](https://huggingface.co/Tencent-Hunyuan/HYDiT-ControlNet-v1.2) Canny, Depth, Pose  
  - [KBlueLeaf/HDM-xut-340M-anime](https://huggingface.co/KBlueLeaf/HDM-xut-340M-anime)  
    highly experimental: HDM *Home-made-Diffusion-Model* is a project to investigate specialized training recipe/scheme  
    for pre-training T2I model at home based on super-light architecture  
    *requires*: generator=cpu, dtype=float16, offload=none, both positive and negative prompts are required and must be long & detailed  
  - [Apple FastVLM](https://huggingface.co/apple/FastVLM-0.5B) in 0.5B, 1.5B and 7B variants  
    available in captioning tab  
  - updated [SD.Next Model Samples Gallery](https://vladmandic.github.io/sd-samples/compare.html)  
- **UI**
  - default to **ModernUI**  
    standard ui is still available via *settings -> user interface -> theme type*  
  - mobile-friendly!  
  - new **History** section in the right panel  
    shows detailed job history plus timeline of the execution  
  - make hints touch-friendly: hold touch to display hint  
  - improved image scaling in img2img and control interfaces  
  - add base model type to networks display, thanks @Artheriax  
  - additional hints to ui, thanks @Artheriax  
  - add video support to gallery, thanks @CalamitousFelicitousness  
  - additional artwork for reference models in networks, thanks @liutyi  
  - improve ui hints display  
  - restyled all toolbuttons to be modernui native  
  - reordered system settings  
  - dynamic direction of dropdowns  
  - improve process tab layout  
  - improve detection of active tab  
  - configurable horizontal vs vertical panel layout  
    in settings -> user interface -> panel min width  
    *example*: if panel width is less than specified value, layout switches to vertical  
  - configurable grid images size  
    in *settings -> user interface -> grid image size*  
  - gallery now includes reference model images  
  - reference models now include indicator if they are *ready* or *need download*
- **Offloading**
  - **balanced**
    - enable offload during pre-forward by default  
    - improve offloading of models with multiple dits  
    - improve offloading of models with impliciy vae processing  
    - improve offloading of models with controlnet  
    - more aggressive offloading of controlnet with lowvram flag  
  - **group**
    - new offloading method, using *type=leaf* works on a similar level as sequential offloading  
      and can present significant savings on low-vram gpus, but comes at the higher performance cost  
- **Quantization**
  - option to specify models types not to quantize: *settings -> quantization*  
    allows for having quantization enabled, but skipping specific model types that do not need it  
    *example*: `sd, sdxl`  
  - **sdnq**
    - add quantized matmul support for all quantization types and group sizes  
    - improve the performance of low bit quants  
  - **nunchaku**: update to `nunchaku==1.0.0`  
    *note*: nunchaku updated the repo which will trigger re-download of nunchaku models when first used  
    nunchaku is currently available for: *Flux.1 Dev/Schnell/Kontext/Krea/Depth/Fill*, *Qwen-Image/Qwen-Lightning*, *SANA-1.6B*  
  - **tensorrt**: new quantization engine from nvidia  
    *experimental*: requires new pydantic package which *may* break other things, to enable start sdnext with `--new` flag  
    *note*: this is model quantization only, no support for tensorRT inference yet  
- **Other**
  - **LoRA** allow specifying module to apply lora on  
    *example*: `<lora:mylora:1.0:module=unet>` would apply lora *only* on unet regardless of lora content  
    this is particularly useful when you have multiple loras and you want to apply them on different parts of the model  
    *example*: `<lora:firstlora:1.0:high>` and `<lora:secondlora:1.0:low>`  
    *note*: `low` is shorthand for `module=transformer_2` and `high` is shortcut for `module=transformer`  
  - **Detailer** allow manually setting processing resolution  
    *note*: this does not impact the actual image resolution, only the resolution at which detailer internally operates  
  - refactor reuse-seed and add functionality to all tabs  
  - refactor modernui js codebase  
  - move zluda flash attenion to *Triton Flash attention* option  
  - remove samplers filtering  
  - allow both flow-matching and discrete samplers for sdxl models  
  - cleanup command line parameters  
  - add `--new` command line flag to enable testing of new packages without breaking existing installs  
  - downgrade rocm to `torch==2.7.1`  
  - set the minimum supported rocm version on linux to `rocm==6.0`  
  - disallow `zluda` and `directml` on non-windows platforms  
  - update openvino to `openvino==2025.3.0`  
  - add deprecation warning for `python==3.9`  
  - allow setting denoise strength to 0 in control/img2img  
    this allows to run workflows which only refine or detail existing image without changing it   
- **Fixes**
  - normalize path hanlding when deleting images  
  - unified compile upscalers  
  - fix OpenVINO with ControlNet  
  - fix hidden model tags in networks display  
  - fix networks reference models display on windows  
  - fix handling of pre-quantized `flux` models  
  - fix `wan` use correct pipeline for i2v models  
  - fix `qwen-image` with hires  
  - fix `omnigen-2` failure  
  - fix `auraflow` quantization  
  - fix `kandinsky-3` noise  
  - fix `infiniteyou` pipeline offloading  
  - fix `skyreels-v2` image-to-video  
  - fix `flex2` img2img denoising strength  
  - fix `flex2` contronet vs inpaint image selection, thanks @alerikaisattera  
  - fix some use cases with access via reverse-proxy  
  - fix segfault on startup with `rocm==6.4.3` and `torch==2.8`  
  - fix wildcards folders traversal, thanks @dymil  
  - fix zluda flash attention with enable_gqa  
  - fix `wan a14b` quantization  
  - fix reprocess workflow for control with hires  
  - fix samplers set timesteps vs sigmas  
  - fix `detailer` missing metadata  
  - fix `infiniteyou` lora load with  

## Update for 2025-08-20

A quick service release with several important hotfixes, improved localization support and adding new **Qwen** model variants...

[ReadMe](https://github.com/vladmandic/automatic/blob/master/README.md) | [ChangeLog](https://github.com/vladmandic/automatic/blob/master/CHANGELOG.md) | [Docs](https://vladmandic.github.io/sdnext-docs/) | [WiKi](https://github.com/vladmandic/automatic/wiki) | [Discord](https://discord.com/invite/sd-next-federal-batch-inspectors-1101998836328697867)

- **Models**
  - [Qwen-Image-Edit](https://huggingface.co/Qwen/Qwen-Image-Edit)  
    Image editing using natural language prompting, similar to `Flux.1-Kontext`, but based on larger 20B `Qwen-Image` model  
  - [Nunchaku-Qwen-Image](https://huggingface.co/nunchaku-tech/nunchaku-qwen-image)  
    if you have a compatible nVidia GPU, Nunchaku is the fastest quantization engine, currently available for Flux.1, SANA and Qwen-Image models  
    *note*: release version of `nunchaku==0.3.2` does NOT include support, so you need to build [nunchaku](https://nunchaku.tech/docs/nunchaku/installation/installation.html) from source  
- [SD.Next Model Samples Gallery](https://vladmandic.github.io/sd-samples/compare.html)  
  - updated with new models  
- **Features**  
  - new *setting -> huggingface -> download method*  
    default is `rust` as new `xet` is known to cause issues  
  - support for `flux.1-kontext` lora  
  - support for `qwen-image` lora  
  - new *setting -> quantization -> modules dtype dict*  
    used to manually override quant types per module  
- **UI**  
  - new artwork for reference models in networks  
    thanks @liutyi  
  - updated [localization](https://vladmandic.github.io/sdnext-docs/Locale/) for all 8 languages  
  - localization support for ModernUI  
  - single-click on locale rotates current locale  
    double-click on locale resets locale to `en`  
  - exclude ModernUI from list of extensions  
    ModernUI is enabled in settings, not by manually enabling extension  
- **Docs**  
  - Models and Video pages updated with links to original model repos, model licenses and original release dates  
    thanks @alerikaisattera  
- **Fixes**  
  - nunchaku use new download links and default to `0.3.2`  
    nunchaku wheels: <https://huggingface.co/nunchaku-tech/nunchaku/tree/main>  
  - fix OpenVINO with offloading  
  - add explicit offload calls on prompt encode  
  - error reporting on model load failure  
  - fix torch version checks  
  - remove extra cache clear  
  - enable explicit sync calls for `rocm` on windows  
  - note if restart-needed on initial startup import error  
  - bypass diffusers-lora-fuse on quantized models  
  - monkey-patch diffusers to use original weights shape when loading lora  
  - guard against null prompt  
  - install `hf_transfter` and `hf_xet` when needed  
  - fix ui cropped network tags  
  - enum reference models on startup  
  - dont report errors if agent scheduler is disabled  

## Update for 2025-08-15

### Highlights for 2025-08-15

New release two weeks after the last one and its a big one with over 150 commits!
- Several new models: [Qwen-Image](https://qwenlm.github.io/blog/qwen-image/) (plus *Lightning* variant) and [FLUX.1-Krea-Dev](https://www.krea.ai/blog/flux-krea-open-source-release)  
- Several updated models: [Chroma](https://huggingface.co/lodestones/Chroma), [SkyReels-V2](https://huggingface.co/Skywork/SkyReels-V2-DF-14B-720P-Diffusers), [Wan-VACE](https://huggingface.co/Wan-AI/Wan2.1-VACE-14B-diffusers), [HunyuanDiT](https://huggingface.co/Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers-Distilled)  
- Plus continuing with major **UI** work with new embedded **Docs/Wiki** search, redesigned real-time **hints**, **wildcards** UI selector, built-in **GPU monitor**, **CivitAI** integration and more!  
- On the compute side, new profiles for high-vram GPUs, offloading improvements, parallel-load for large models, support for new `torch` release and improved quality when using low-bit quantization!      
- [SD.Next Model Samples Gallery](https://vladmandic.github.io/sd-samples/compare.html): pre-generated image gallery with 60 models (45 base and 15 finetunes) and 40 different styles resulting in 2,400 high resolution images!  
  gallery additionally includes model details such as typical load and inference times as well as sizes and types of each model component (*e.g. unet, transformer, text-encoder, vae*)  
- And (*as always*) many bugfixes and improvements to existing features!  

![sd-samples](https://github.com/user-attachments/assets/3efc8603-0766-4e4e-a4cb-d8c9b13d1e1d)

[ReadMe](https://github.com/vladmandic/automatic/blob/master/README.md) | [ChangeLog](https://github.com/vladmandic/automatic/blob/master/CHANGELOG.md) | [Docs](https://vladmandic.github.io/sdnext-docs/) | [WiKi](https://github.com/vladmandic/automatic/wiki) | [Discord](https://discord.com/invite/sd-next-federal-batch-inspectors-1101998836328697867)

*Note*: Change-in-behavior - locations of downloaded HuggingFace models and components are changed to allow for de-duplication of common modules and switched from using system default cache folder to `models/huggingface`  
SD.Next will warn on startup on unused cache entries that can be removed. Also, to take advantage of de-duplication, you'll need to delete models from your `models/Diffusers` folder and let SD.Next re-download them!  

### Details for 2025-08-15

- **Models**  
  - [Qwen-Image](https://qwenlm.github.io/blog/qwen-image/)  
    new image foundational model with *20B* params DiT and using *Qwen2.5-VL-7B* as the text-encoder!  
    available via *networks -> models -> reference*  
    *note*: this model is almost 2x the size of Flux, quantization and offloading are highly recommended!  
    *recommended* params: *steps=50, attention-guidance=4*  
    also available is pre-packaged [Qwen-Lightning](https://huggingface.co/vladmandic/Qwen-Lightning)  
    which is an unofficial merge of [Qwen-Image](https://qwenlm.github.io/blog/qwen-image/) with [Qwen-Lightning-LoRA](https://github.com/ModelTC/Qwen-Image-Lightning/) to improve quality and allow for generating in 8-steps!  
  - [FLUX.1-Krea-Dev](https://www.krea.ai/blog/flux-krea-open-source-release)  
    new 12B base model compatible with FLUX.1-Dev from *Black Forest Labs* with opinionated aesthetics and aesthetic preferences in mind  
    available via *networks -> models -> reference*  
  - [Chroma](https://huggingface.co/lodestones/Chroma)  
    great model based on FLUX.1 and then redesigned and retrained by *lodestones*  
    update with latest **HD**, **HD Flash** and **HD Annealed** variants which are based on *v50* release  
    available via *networks -> models -> reference*  
  - [SkyReels-V2](https://huggingface.co/Skywork/SkyReels-V2-DF-14B-720P-Diffusers)  
    SkyReels-V2 is a genarative video model based on Wan-2.1 but with heavily modified execution to allow for infinite-length video generation  
    supported variants are:  
    - diffusion-forcing: *T2I DF 1.3B* for 540p videos, *T2I DF 14B* for 720p videos, *I2I DF 14B* for 720p videos  
    - standard: *T2I 14B* for 720p videos and *I2I 14B* for 720p videos  
  - [Wan-VACE](https://huggingface.co/Wan-AI/Wan2.1-VACE-14B-diffusers)  
    basic support for *Wan 2.1 VACE 1.3B* and *14B* variants  
    optimized support with granular guidance control will follow soon  
  - [HunyuanDiT-Distilled](https://huggingface.co/Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers-Distilled)  
    variant of HunyuanDiT with reduced steps and improved performance  
**Torch**  
  - Set default to `torch==2.8.0` for *CUDA, ROCm and OpenVINO*  
  - Add support for `torch==2.9.0-nightly`  
- **UI**  
  - new embedded docs/wiki search!  
    **Docs** search: fully-local and works in real-time on all document pages  
    **Wiki** search: uses github api to search online wiki pages  
  - updated real-time hints, thanks @CalamitousFelicitousness  
  - add **Wilcards** UI  
    in networks display  
  - every heading element is collapsible!  
  - quicksettings reset button to restore all quicksettings to default values  
    because things do sometimes get wrong...  
  - configurable image fit in all image views  
  - rewritten **CivitAI downloader**  
    in *models -> civitai*  
    *hint*: you can enter model id in a search bar to pull information on specific model directly  
    *hint*: you can download individual versions or batch-download all-at-once!  
  - redesigned **GPU monitor**  
    - standard-ui: *system -> gpu monitor*  
    - modern-ui: *aside -> console -> gpu monitor*  
    - supported for *nVidia CUDA* and *AMD ROCm* platforms  
    - configurable interval in *settings -> user interface*  
  - updated *models* tab
    - updated *models -> current* tab  
    - updated *models -> list models* tab  
    - updated *models -> metadata* tab  
  - updated *extensions* tab
  - redesigned *settings -> user interface*  
  - gallery bypass browser cache for thumbnails  
  - gallery safer delete operation  
  - networks display indicator for currently active items  
    applies to: *styles, loras*  
  - apply privacy blur to hf and civitai tokens  
  - image download will now use actual image filename  
  - increase default and maximum ui request timeout to 2min/5min  
  - *hint*: card layout  
    card layout is used by networks, gallery, civitai search, etc.  
    you can change card size in *settings -> user interface*  
- **Offloading**  
  - changed **default** values for offloading based on detected gpu memory  
    see [offloading docs](https://vladmandic.github.io/sdnext-docs/Offload/) for details  
  - new feature to specify which modules to offload always or never  
    in *settings -> model offloading -> offload always/never*  
  - new `highvram` profile provides significant performance boost on gpus with more than 24gb  
  - new `offload during pre-forward` option  
    in *settings -> model offloading*  
    switches from explicit offloading to implicit offloading on module execution change  
  - new `diffusers_offload_nonblocking` exerimental setting  
    instructs torch to use non-blocking move operations when possible  
- **Features**  
  - new `T5: Use shared instance of text encoder` option  
    in *settings -> text encoder*  
    since a lot of new models use T5 text encoder, this option allows to share  
    the same instance across all models without duplicate downloads  
    *note* this will not reduce size of your already downloaded models, but will reduce size of future downloads  
  - **Wan** select which stage to run: *first/second/both* with configurable *boundary ration* when running both stages  
    in settings -> model options  
  - prompt parser allow explict `BOS` and `EOS` tokens in prompt  
  - **Nunchaku** support for *FLUX.1-Fill* and *FLUX.1-Depth* models  
  - update requirements/packages  
  - use model vae scale-factor for image width/heigt calculations  
  - **SDNQ** add `modules_dtype_dict` to quantize *Qwen Image* with mixed dtype  
  - **prompt enhance**
    add `allura-org/Gemma-3-Glitter-4B`, `Qwen/Qwen3-4B-Instruct-2507`, `Qwen/Qwen2.5-VL-3B-Instruct` model support  
    improve system prompt  
  - **schedulers** add **Flash FlowMatch**  
  - **model loader** add parallel loader option  
    enabled by default, selectable in *settings -> model loading*  
  - **filename namegen** use exact sequence number instead of next available  
    this allows for more predictable and consistent filename generation  
  - **network delete** new feature that allows to delete network from disk  
    in *networks -> show details -> delete*  
    this will also delete description, metadata and previews associated with the network  
    only applicable to safetensors networks, not downloaded diffuser models  
- **Wiki**  
  - Models page updated with links to original model repos and model licenses, thanks @alerikaisattera  
  - Updated Model-Support with newly supported models  
  - Updated Offload, Prompting, API pages  
- **API**
  - add `/sdapi/v1/checkpoint` POST endpoint to simply load a model  
  - add `/sdapi/v1/modules` GET endpoint to get info on model components/modules  
  - all generate endpoints now support `sd_model_checkpoint` parameter  
    this allows to specify which model to use for generation without needing to use additional endpoints  
- **Refactor**
  - change default huggingface cache folder from system default to `models/huggingface`  
    sd.next will warn on startup on unused cache entries  
  - new unified pipeline component loader in `pipelines/generic`  
  - remove **LDSR**  
  - remove `api-only` cli option  
- **Docker**  
  - update cuda base image: `pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime`  
  - update official builds: <https://hub.docker.com/r/vladmandic/sdnext-cuda/tags>  
- **Fixes**  
  - refactor legacy processing loop  
  - fix settings components mismatch  
  - fix *Wan 2.2-5B I2V* workflow  
  - fix *Wan* T2I workflow  
  - fix OpenVINO  
  - fix video model vs pipeline mismatch  
  - fix video generic save frames  
  - fix inpaint image metadata  
  - fix processing image save loop  
  - fix progress bar with refine/detailer  
  - fix api progress reporting endpoint  
  - fix `openvino` backend failing to compile  
  - fix `zluda` with hip-sdk==6.4
  - fix `nunchaku` fallback on unsupported model  
  - fix `nunchaku` windows download links  
  - fix *Flux.1-Kontext-Dev* with variable resolution  
  - use `utf_16_be` as primary metadata decoding  
  - fix `sd35` width/height alignment  
  - fix `nudenet` api  
  - fix global state tracking  
  - fix ui tab detection for networks  
  - fix ui checkbox/radio styling for non-default themes  
  - fix loading custom transformers and t5 safetensors tunes  
  - add mtime to reference models  
  - patch torch version so 3rd party libraries can use expected format  
  - unified stat size/mtime calls  
  - reapply offloading on ipadapter load  
  - api set default script-name  
  - avoid forced gc and rely on thresholds  
  - add missing interrogate in output panel  

## Update for 2025-07-29

### Highlights for 2025-07-29

This is a big one: simply looking at number of changes, probably the biggest release since the project started!  

Feature highlights include:  
- [ModernUI](https://github.com/user-attachments/assets/6f156154-0b0a-4be2-94f0-979e9f679501) has quite some redesign which should make it more user friendly and easier to navigate plus several new UI themes  
  If you're still using **StandardUI**, give [ModernUI](https://vladmandic.github.io/sdnext-docs/Themes/) a try!  
- New models such as [WanAI 2.2](https://wan.video/) in 5B and A14B variants for both *text-to-video* and *image-to-video* workflows as well as *text-to-image* workflow!  
  and also [FreePik F-Lite](https://huggingface.co/Freepik/F-Lite), [Bria 3.2](https://huggingface.co/briaai/BRIA-3.2) and [bigASP 2.5](https://civitai.com/models/1789765?modelVersionId=2025412)  
- Redesigned [Video](https://vladmandic.github.io/sdnext-docs/Video) interface with support for general video models plus optimized [FramePack](https://vladmandic.github.io/sdnext-docs/FramePack) and [LTXVideo](https://vladmandic.github.io/sdnext-docs/LTX) support  
- Fully integrated nudity detection and optional censorship with [NudeNet](https://vladmandic.github.io/sdnext-docs/NudeNet)  
- New background replacement and relightning methods using **Latent Bridge Matching** and new **PixelArt** processing filter  
- Enhanced auto-detection of default sampler types/settings results in avoiding common mistakes  
- Additional **LLM/VLM** models available for captioning and prompt enhance  
- Number of workflow and general quality-of-life improvements, especially around **Styles**, **Detailer**, **Preview**, **Batch**, **Control**  
- Compute improvements  
- [Wiki](https://github.com/vladmandic/automatic/wiki) & [Docs](https://vladmandic.github.io/sdnext-docs/) updates, especially new end-to-end [Parameters](https://vladmandic.github.io/sdnext-docs/Parameters/) page  

In this release we finally break with legacy with the removal of the original [A1111](https://github.com/AUTOMATIC1111/stable-diffusion-webui/) codebase which has not been maintained for a while now  
This plus major cleanup of codebase and external dependencies resulted in ~55k LoC (*lines-of-code*) reduction and spread over [~750 files](https://github.com/vladmandic/sdnext/pull/4017) in ~200 commits!  

We also switched project license to [Apache-2.0](https://github.com/vladmandic/sdnext/blob/dev/LICENSE.txt) which means that SD.Next is now fully compatible with commercial and non-commercial use and redistribution regardless of modifications!  

And (*as always*) many bugfixes and improvements to existing features!  
For details, see [ChangeLog](https://github.com/vladmandic/automatic/blob/master/CHANGELOG.md)  

> [!NOTE]  
> We recommend clean install for this release due to sheer size of changes  
> Although upgrades and existing installations are tested and should work fine!  

![Screenshot](https://github.com/user-attachments/assets/6f156154-0b0a-4be2-94f0-979e9f679501)

[ReadMe](https://github.com/vladmandic/automatic/blob/master/README.md) | [ChangeLog](https://github.com/vladmandic/automatic/blob/master/CHANGELOG.md) | [Docs](https://vladmandic.github.io/sdnext-docs/) | [WiKi](https://github.com/vladmandic/automatic/wiki) | [Discord](https://discord.com/invite/sd-next-federal-batch-inspectors-1101998836328697867)

### Details for 2025-07-29

- **License**  
  - SD.Next [license](https://github.com/vladmandic/sdnext/blob/dev/LICENSE.txt) switched from **aGPL-v3.0** to **Apache-v2.0**  
    this means that SD.Next is now fully compatible with commercial and non-commercial use and redistribution regardless of modifications!  
- **Models**
  - [WanAI Wan 2.2](https://github.com/Wan-Video/Wan2.2) both 5B and A14B variants, for both T2V and I2V support  
    go to: *video -> generic -> wan -> pick variant*  
    optimized support with *VACE*, etc. will follow soon  
    *caution* Wan2.2 on its own is ~68GB, but also includes optional second-stage for later low-noise processing which is absolutely massive at additional ~54GB  
    you can enable second stage processing in *settings -> model options*, its disabled by default  
    *note*: quantization and offloading are highly recommended regardless of first-stage only or both stages!  
  - [WanAI Wan](https://wan.video/) T2V models for T2I workflows  
    Wan is originally designed for *video* workflows, but now also be used for *text-to-image* workflows!  
    supports *Wan-2.1 in 1.3B* and 14B variants and *Wan-2.2 in 5B and A14B* variants  
    supports all standard features such as quantization, offloading, TAESD preview generation, LoRA support etc.  
    can also load unet/transformer fine-tunes in safetensors format using UNET loader  
    simply select in *networks -> models -> reference*  
    *note* 1.3B model is a bit too small for good results and 14B is very large at 78GB even without second-stage so aggressive quantization and offloading are recommended  
  - [FreePik F-Lite](https://huggingface.co/Freepik/F-Lite) in *7B, 10B and Texture* variants  
    F-Lite is a 7B/10B model trained exclusively on copyright-safe and SFW content, trained on internal dataset comprising approximately 80 million copyright-safe images  
    available via *networks -> models -> reference*  
  - [Bria 3.2](https://huggingface.co/briaai/BRIA-3.2)  
    Bria is a smaller 4B parameter model built entirely on licensed data and safe for commercial use  
    *note*: this is a gated model, you need to [accept terms](https://huggingface.co/briaai/BRIA-3.2) and set your [huggingface token](https://vladmandic.github.io/sdnext-docs/Gated/)  
    available via *networks -> models -> reference*  
  - [bigASP 2.5](https://civitai.com/models/1789765)  
    bigASP is an experimental SDXL finetune using Flow matching method  
    load as usual, and leave sampler set to *Default*  
    or you can use following samplers: *UniPC, DPM, DEIS, SA*  
    required sampler settings: *prediction-method=flow-prediction*, *sigma-method=flowmatch*  
    recommended sampler settings: *flow-shift=1.0*  
  - [LBM: Latent Bridge Matching](https://github.com/gojasper/LBM)  
    very fast automatic image background replacement methods with relightning!  
    *simple*: automatic background replacement using [BiRefNet](https://github.com/ZhengPeng7/BiRefNet)  
    *relighting*: automatic background replacement with reglighting so source image fits desired background  
    with optional composite blending  
    available in *img2img or control -> scripts*  
  - add **FLUX.1-Kontext-Dev** inpaint workflow  
  - add **FLUX.1-Kontext-Dev** **Nunchaku** support  
    *note*: FLUX.1 Kontext is about 2-3x faster with Nunchaku vs standard execution!  
  - support **FLUX.1** all-in-one safetensors  
  - support for [Google Gemma 3n](https://huggingface.co/google/gemma-3n-E4B-it) E2B and E4B LLM/VLM models  
    available in **prompt enhance** and process **captioning**  
  - support for [HuggingFace SmolLM3](https://huggingface.co/HuggingFaceTB/SmolLM3-3B) 3B LLM model  
    available in **prompt enhance**  
  - add [fal AuraFlow 0.2](https://huggingface.co/fal/AuraFlow-v0.2) in addition to existing [fal AuraFlow 0.3](https://huggingface.co/fal/AuraFlow-v0.3) due to large differences in model behavior  
    available via *networks -> models -> reference*  
  - add integrated [NudeNet](https://vladmandic.github.io/sdnext-docs/NudeNet) as built-in functionality  
    *note*: used to be available as a separate [extension](https://github.com/vladmandic/sd-extension-nudenet)  
- **Video**
  - redesigned **Video** interface  
  - support for **Generic** video models  
    includes support for many video models without specific per-model optimizations  
    included: *Hunyuan, LTX, WAN, Mochi, Latte, Allegro, Cog*  
    supports quantization, offloading, frame interpolation, etc.  
  - support for optimized [FramePack](https://vladmandic.github.io/sdnext-docs/FramePack)  
    with *t2i, i2i, flf2v* workflows  
    LoRA support, prompt enhance, etc.  
    now fully integrated instead of being a separate extension  
  - support for optmized [LTXVideo](https://vladmandic.github.io/sdnext-docs/LTX)  
    with *t2i, i2i, v2v* workflows  
    optional native upsampling and video refine workflows  
    LoRA support with different conditioning types such as Canny/Depth/Pose, etc.  
  - support for post load quantization  
- **UI**  
  - major update to modernui layout  
  - add new Windows-like *Blocks* UI theme  
  - redesign of the *Flat* UI theme  
  - enhanced look&feel for *Gallery* tab with better search and collapsible sections, thanks to @CalamitousFelicitousness
- **WIKI**  
  - new [Parameters](https://vladmandic.github.io/sdnext-docs/Parameters/) page that lists and explains all generation parameters  
    massive thanks to @CalamitousFelicitousness for bringing this to life!  
  - updated *Models, Video, LTX, FramePack, Styles*, etc.
- **Compute**  
  - support for [SageAttention2++](https://github.com/thu-ml/SageAttention)  
    provides 10-15% performance improvement over default SDPA for transformer-based models!  
    enable in *settings -> compute settings -> sdp options*  
    *note*: SD.Next will use either SageAttention v1/v2/v2++, depending which one is installed  
    until authors provide pre-build wheels for v2++, you need to install it manually or SD.Next will auto-install v1  
  - support for `torch.compile` for LLM: captioning/prompt-enhannce  
  - support for `torch.compile` with repeated-blocks  
    reduces time-to-compile 5x without loss of performance!  
    enable in *settings -> model compile -> repeated*  
    *note*: torch.compile is not compatible with balanced offload  
- **Other**  
  - **Styles** can now include both generation params and server settings  
    see [Styles docs](https://vladmandic.github.io/sdnext-docs/Styles/) for details  
  - **TAESD** is now default preview type since its the only one that supports most new models  
  - support **TAESD** preview and remote VAE for **HunyuanDit**  
  - support **TAESD** preview and remote VAE for **AuraFlow**  
  - support **TAESD** preview for **WanAI**  
  - SD.Next now starts with *locked* state preventing model loading until startup is complete  
  - warn when modifying legacy settings that are no longer supported, but available for compatibilty  
  - warn on incompatible sampler and automatically restore default sampler  
  - **XYZ grid** can now work with control tab:  
    if controlnet or processor are selected in xyz grid, they will overwrite settings from first unit in control tab,  
    when using controlnet/processor selected in xyz grid, behavior is forced as control-only  
    also freely selectable are control strength, start and end values  
  - **Batch** warn on unprocessable images and skip operations on errors so that other images can still be processed  
  - **Metadata** improved parsing and detect foreign metadata  
    detect ComfyUI images  
    detect InvokeAI images  
  - **Detailer** add `expert` mode where list of detailer models can be converted to textbox for manual editing  
    see [docs](https://vladmandic.github.io/sdnext-docs/Detailer/) for more information  
  - **Detailer** add option to merge multiple results from each detailer model  
    for example, hands model can result in two hands each being processed separately or both hands can be merged into one composite job  
  - **Control** auto-update width/height on image upload  
  - **Control** auto-determine image save path depending on operations performed  
  - autodetect **V-prediction** models and override default sampler prediction type as needed  
- **SDNQ**  
  - use inference context during quantization  
  - use static compile  
  - rename quantization type for text encoders `default` option to `Same as model`  
- **API**  
  - add `/sdapi/v1/lock-checkpoint` endpoint that can be used to lock/unlock model changes  
    if model is locked, it cannot be changed using normal load or unload methods  
- **Fixes**
  - allow theme type `None` to be set in config  
  - installer dont cache installed state  
  - fix Cosmos-Predict2 retrying TAESD download  
  - better handle startup import errors  
  - fix traceback width preventing copy&paste  
  - fix ansi controle output from scripts/extensions  
  - fix diffusers models non-unique hash  
  - fix loading of manually downloaded diffuser models  
  - fix api `/sdapi/v1/embeddings` endpoint  
  - fix incorrect reporting of deleted and modified files  
  - fix SD3.x loader and TAESD preview  
  - fix xyz with control enabled  
  - fix control order of image save operations  
  - fix control batch-input processing  
  - fix modules merge save model  
  - fix torchvision bicubic upsample with ipex  
  - fix instantir pipeline  
  - fix prompt encoding if prompts within batch have different segment counts  
  - fix detailer min/max size  
  - fix loopback script  
  - fix networks tags display  
  - fix yolo refresh models  
  - cleanup control infotext  
  - allow upscaling with models that have implicit VAE processing  
  - framepack improve offloading  
  - improve prompt parser tokenizer loader  
  - improve scripts error handling  
  - improve infotext param parsing  
  - improve extensions ui search  
  - improve model type autodetection  
  - improve model auth check for hf repos  
  - improve Chroma prompt padding as per recommendations  
  - lock directml torch to `torch-directml==0.2.4.dev240913`  
  - lock directml transformers to `transformers==4.52.4`  
  - improve install of `sentencepiece` tokenizer  
  - add int8 matmul fallback for ipex with onednn qlinear  
- **Refactoring**  
  *note*: none of the removals result in loss-of-functionality since all those features are already re-implemented  
  goal here is to remove legacy code, code duplication and reduce code complexity  
  - obsolete **original backend**  
  - remove majority of legacy **a1111** codebase  
  - remove legacy ldm codebase: `/repositories/ldm`  
  - remove legacy blip codebase: `/repositories/blip`  
  - remove legacy codeformer codebase: `/repositories/codeformer`  
  - remove legacy clip patch model: `/models/karlo`  
  - remove legacy model configs: `/configs/*.yaml`  
  - remove legacy submodule: `/modules/k-diffusion`  
  - remove legacy hypernetworks support: `/modules/hypernetworks`  
  - remove legacy lora support: `/extensions-builtin/Lora`  
  - remove legacy clip/blip interrogate module  
  - remove modern-ui remove `only-original` vs `only-diffusers` code paths  
  - refactor control processing and separate preprocessing and image save ops  
  - refactor modernui layouts to rely on accordions more than individual controls  
  - refactore pipeline apply/unapply optional components & features  
  - split monolithic `shared.py`  
  - cleanup `/modules`: move pipeline loaders to `/pipelines` root  
  - cleanup `/modules`: move code folders used by pipelines to `/pipelines/<pipeline>` folder  
  - cleanup `/modules`: move code folders used by scripts to `/scripts/<script>` folder  
  - cleanup `/modules`: global rename `modules.scripts` to avoid conflict with `/scripts`  
  - override `gradio` installer  
  - major refactoring of requirements and dependencies to unblock `numpy>=2.1.0`  
  - patch `insightface`  
  - patch `facelib`  
  - patch `numpy`  
  - stronger lint rules  
    add separate `npm run lint`, `npm run todo`, `npm run test`, `npm run format` macros  

## Update for 2025-06-30

### Highlights for 2025-06-30

New release with ~100 commits...So what's new? Well, its been a busy few weeks with new models coming out quite frequently:  
- New T2I/I2I models: **OmniGen-2, Cosmos-Predict2, FLUX.1-Kontext, Chroma**  
- Additional VLM models: **JoyCaption Beta, MoonDream 2**  
- Additional upscalers: **UltraSharp v2**  

And (as always) many bugfixes and improvements to existing features!  

[ReadMe](https://github.com/vladmandic/automatic/blob/master/README.md) | [ChangeLog](https://github.com/vladmandic/automatic/blob/master/CHANGELOG.md) | [Docs](https://vladmandic.github.io/sdnext-docs/) | [WiKi](https://github.com/vladmandic/automatic/wiki) | [Discord](https://discord.com/invite/sd-next-federal-batch-inspectors-1101998836328697867)

### Details for 2025-06-30

- **Models**
  - [Models Wiki page](https://vladmandic.github.io/sdnext-docs/Models/) is updated will all new models  
    *note* all new image models larger than 30GB, so [offloading](https://vladmandic.github.io/sdnext-docs/Offload/) and [quantization](https://vladmandic.github.io/sdnext-docs/Quantization/) are necessary!  
  - [OmniGen2](https://huggingface.co/OmniGen2/OmniGen2)  
    - OmniGen2 is a powerful unified multimodal model that supports t2i and i2i workflows and uses 4B transformer with Qwen-VL-2.5 4B VLM  
    - available via *networks -> models -> reference*  
  - [nVidia Cosmos-Predict2 T2I](https://research.nvidia.com/labs/dir/cosmos-predict2/) *2B and 14B*  
    - Cosmos-Predict2 T2I is a new foundational model from Nvidia in two variants: small 2B and large 14B
    - available via *networks -> models -> reference*  
    - *note*: 14B variant is a very large model at 36GB
    - *note*: this is a gated model, you need to [accept terms](https://huggingface.co/nvidia/Cosmos-Predict2-2B-Text2Image) and set your [huggingface token](https://vladmandic.github.io/sdnext-docs/Gated/)  
  - [Black Forest Labs FLUX.1 Kontext I2I](https://bfl.ai/announcements/flux-1-kontext-dev) *Dev* variant  
    - FLUX.1-Kontext is a 12B model billion parameter capable of editing images based on text instructions  
    - model is primarily designed for image editing workflows, but also works for text-to-image workflows  
    - requirements are similar to regular FLUX.1 although 2x slower  
    - available via *networks -> models -> reference*  
    - *note*: this is a gated model, you need to [accept terms](https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev) and set your [huggingface token](https://vladmandic.github.io/sdnext-docs/Gated/)  
  - [lodestones Chroma](https://huggingface.co/lodestones/Chroma)  
    - Chroma is a 8.9B parameter model based on *FLUX.1-schnell* and fully Apache 2.0 licensed  
    - available via *networks -> models -> reference*  
    - *note*: model is still in training so future updates will trigger re-download  
    - large credits to @Trojaner for work on bringing Chroma support to SD.Next and all the optimizations around it!  
  - [JoyCaption Beta](https://huggingface.co/fancyfeast/llama-joycaption-beta-one-hf-llava) support (in addition to existing JoyCaption Alpha)  
    - new version of highly popular captioning model  
    - available via *caption -> vlm caption*  
  - [MoonDream 2](https://huggingface.co/vikhyatk/moondream2) support (updated)  
    - really good 2B captioning model that can work on different levels of detail  
    - available via *caption -> vlm caption*  
  - [UltraSharp v2](https://huggingface.co/Kim2091/UltraSharpV2) support  
    - one of the best upscalers (traditional, non-diffusion) available today!  
    - available via *process -> upscale -> chainner*  
- **Changes**  
  - Update all core requirements  
  - Support Remote VAE with *Omnigen, Lumina 2 and PixArt*  
  - Enable quantization for captioning: *Gemma, Qwen, SMOL, Florence, JoyCaption*  
  - Add `--trace` command line param that enables trace logging  
  - Use Diffusers version of *OmniGen*  
  - Control move global settings to control elements -> control settings tab  
  - Control add setting to run hires with or without control  
  - Update OpenVINO to 2025.2.0  
  - Simplified and unified quantization enabled for options  
  - Add **PixelArt** filter to processing tab  
- **SDNQ Quantization**  
  - Add `auto` quantization mode  
  - Add `modules_to_not_convert` support for post mode  
  - Improve offload compatibility  
  - Fix Qwen 2.5 with int8 matmul  
  - Fix Dora loading  
  - Remove per layer GC  
  - Add support for XYZ grid to test quantization modes  
    *note*: you need to enable quantization and choose what it applies on, then xyz grid can change quantization mode  
    *note*: you can also enable 'add time info' to compare performance of different quantization modes  
- **API**
  - Add `/sdapi/v1/network?page=<page_name>&item=<item_name>` endpoint that returns full network info  
  - Add `/sdapi/v1/lora?lora=<lora_name>` endpoint that returns full lora info and metadata  
  - Add `/sdapi/v1/controlnets?model_type=<model_type|all|None>` endpoints that returns list of available controlnets for specific model type  
  - Set default sampler to `Default`  
- **Fixes**  
  - IPEX with DPM2++ FlowMatch samplers  
  - Invalid attention processor with ControlNet  
  - LTXVideo default scheduler  
  - Balanced offload with OmniGen  
  - Quantization with OmniGen  
  - Do not save empty `params.txt` file  
  - Override `params.txt` using `SD_PATH_PARAMS` env variable  
  - Add `wheel` to requirements due to `pip` change  
  - Case-insensitive sampler name matching  
  - Fix delete file with gallery views  
  - Add `SD_SAVE_DEBUG` env variable to report all params and metadata save operations as they happen  
  - Fix TAESD model type detection  
  - Fix LoRA loader incorrectly reporting errors  
  - Fix hypertile for img2img and inpaint operations  
  - Fix prompt parser batch size  
  - Fix process batch with batch count  
  - Fix process batch double image save  
  - Fix unapply texture tiling  
  - Fix nunchaku batch support  
  - Fix LoRA change detection on pipeline type change  
  - Fix LoRA load order when it includes text-encoder data  
  - Suppress torch empty logging  
  - Improve TAESD live preview downscale handling  

## Update for 2025-06-16

- **Feature**  
  - Support for Python 3.13  
  - TeaCache support for Lumina 2  
  - Custom UNet and VAE loading support for Lumina 2  
- **Changes**  
  - Increase the medvram mode threshold from 8GB to 12GB  
  - Set CPU backend to use FP32 by default  
  - Relax Python version checks for Zluda  
  - Make VAE options not require model reload  
  - Add warning about incompatible attention processors  
- **Torch**  
  - Set default to `torch==2.7.1`  
  - Force upgrade pip when installing Torch  
- **ROCm**  
  - Support ROCm 6.4 with `--use-nightly`  
  - Don't override user set gfx version  
  - Don't override gfx version with RX 9000  
  - Fix flash-atten repo  
- **SDNQ Quantization**  
  - Add group size support for convolutional layers  
  - Add quantized matmul support for for convolutional layers  
  - Add 7-bit, 5-bit and 3-bit quantization support  
  - Add separate quant mode option for Text Encoders  
  - Fix forced FP32 with tensorwise FP8 matmul  
  - Fix PyTorch <= 2.4 compatibility with FP8 matmul  
  - Fix VAE with conv quant  
  - Don't ignore the Quantize with GPU option with offload mode `none` and `model`  
  - High VRAM usage with Lumina 2  
- **Fixes**  
  - Meissonic with multiple generators  
  - OmniGen with new transformers  
  - Invalid attention processors  
  - PixArt Sigma Small and Large loading  
  - TAESD previews with PixArt and Lumina 2  
  - VAE Tiling with non-default tile sizes  
  - Lumina 2 with IPEX  
  - Nunchaku updated repo  
  - Double loading of models with custom UNets  

## Update for 2025-06-02

### Highlights for 2025-06-02

This release is all about quantization: with new SD.Next own quantization method: **SDNQ**  
**SDNQ** is based on **NNCF**, but has been re-implemented, optimized and evolved enough to become its own quantization method!  
It's fully cross-platform, supports all GPUs and includes tons of quantization methods:
- *8-bit, 6-bit, 4-bit, 2-bit and 1-bit int and uint*
- *8-bit e5, e4 and fnuz float*

Also unlike most traditional methods, its also applicable to nearly all model types  

*Hint*: Even if you may not need quantization for your current model, it may be worth trying it out as it can significantly improve performance or capabilities of your existing workflow! For example, you may not have issues with SD15 or SDXL, but you may have been limited running at high resolutions or with multiple ControlNet due to VRAM requirements - this will significantly reduce memory requirements. And on-the-fly quantization takes just few seconds during model load, there is no need to have multiple quant models permanently saved.  

On a different topic, **SD.Next Wiki & Docs** and its **UI Hints** and **UI Localization** system are community efforts and any contributions are welcome!  
You dont need any coding experience, but if you learned something and you find documentation either wrong or insufficient, please do suggest edits!  
Take a look at [Docs](https://github.com/vladmandic/sdnext/wiki/Docs), [Hints](https://github.com/vladmandic/sdnext/wiki/Hints) and [Localization](https://github.com/vladmandic/sdnext/wiki/Locale) contribution guides

[ReadMe](https://github.com/vladmandic/automatic/blob/master/README.md) | [ChangeLog](https://github.com/vladmandic/automatic/blob/master/CHANGELOG.md) | [Docs](https://vladmandic.github.io/sdnext-docs/) | [WiKi](https://github.com/vladmandic/automatic/wiki) | [Discord](https://discord.com/invite/sd-next-federal-batch-inspectors-1101998836328697867)

### Details for 2025-06-02

- **SDNQ Quantization**  
  - Renamed `NNCF` to `SDNQ`  
  - Renamed quantization scheme names to the underlying dtype names instead of NNCF names  
    - `INT8_SYM` -> `int8`  
    - `INT8` -> `uint8`  
    - `INT4_SYM` -> `int4`  
    - `INT4` -> `uint4`  
  - Add `float8_e4m3fn`, `float8_e5m2`, `float8_e4m3fnuz`, `float8_e5m2fnuz`, `int6`, `uint6`, `int2`, `uint2` and `uint1` support  
  - Add quantized matmul support for `float8_e4m3fn` and `float8_e5m2`  
  - Set the default quant mode to `pre`  
  - Use per token input quant with int8 and fp8 quantized matmul  
  - Implement better layer hijacks  
  - Add an option to toggle quantize with GPU  
  - Fix conv quant and add support for conv quant with asym modes  
  - Fix lora weight change  
  - Fix high RAM usage with pre mode  
  - Fix scale and zero_point not being offloaded  
- **IPEX**  
  - Disabe Dynamic Attention by default on PyTorch 2.7  
  - Remove GradScaler hijack and use `torch.amp.GradScaler` instead  
- **Feature**  
  - TeaCache support for HiDream I1  
- **Changes**  
  - Set the default attention optimizer to Scaled-Dot-Product on all backends  
  - Enable Dynamic attention for Scaled-Dot-Product with ROCm, DirectML, MPS and CPU backends  
- **Fixes**
  - Gallery duplicate entries  
  - Prompt enhancement args mismatch  

## Update for 2025-05-17

*Curious how your system is performing?*  
Run a built-in benchmark and compare to over 15k unique results world-wide: [Benchmark data](https://vladmandic.github.io/sd-extension-system-info/pages/benchmark.html)!  
From slowest 0.02 it/s running on 6th gen CPU without acceleration up to 275+ it/s running on tuned GH100 system!  

Also, since quantization is becoming a necessity for almost all new models, see comparison of different quantization methods available in SD.Next: [Quantization](https://vladmandic.github.io/sdnext-docs/Quantization/)  
*Hint*: Even if you may not need quantization for your current model, it may be worth trying it out as it can significantly improve performance!  

For ZLUDA users, this update adds [compatibility](https://github.com/vladmandic/sdnext/issues/3918) with with latest AMD Adrenaline drivers  

Btw, last few releases have been smaller, but more regular so do check posts about previous releases as features do quickly add up!  

- **Wiki**  
  - Updates for: *Quantization, NNCF, WSL, ZLUDA, ROCm*  
- **Models**  
  - [Index AniSora v1 5B](https://huggingface.co/IndexTeam/Index-anisora) I2V  
    Based on CogVideoX architecture, trained as animated video generation model: This Project presenting Bilibili's gift to the anime world!  
  - [Index AniSora v1 RL 5B](https://github.com/bilibili/Index-anisora?tab=readme-ov-file#anisorav10_rl) I2V  
    RL-optimized AniSoraV1.0 for enhanced anime-style output  
- **Compute**  
  - ZLUDA: update to `zluda==3.9.5` with `torch==2.7.0`  
    *Note*: delete `.zluda` folder so that newest zluda will be installed if you are using the latest AMD Adrenaline driver  
  - NNCF: added experimental support for direct INT8 MatMul  
- **Feature**  
  - Prompt Enhance: option to allow/disallow NSFW content  
- **Fixes**  
  - OpenVINO: force cpu device  
  - Gradio: major cleanup and fixing defaults and ranges  
  - Pydantic: update to api types  
  - UI defaults: match correct prompt components  
  - NNCF with ControlNet  
  - NNCF with CogVideo
  - IPEX with CogVideo  
  - JXL image format metadata handling  

## Update for 2025-05-12

### Highlights for 2025-05-12

First of all NNCF quantization engine has gone through some major enhancements and its now much faster, both in quantization as well as actual inference!  
And its a only truly cross-platform solution for quantization as all other methods are platform specific.  

*Note* if you're a ZLUDA user, see notes on GPU driver compatibility as recent Andrenaline drivers do cause problems!  
And if you're a ROCm user, this release brings much faster compile times on Linux as well as first (experimental) builds for Windows!  

[ReadMe](https://github.com/vladmandic/automatic/blob/master/README.md) | [ChangeLog](https://github.com/vladmandic/automatic/blob/master/CHANGELOG.md) | [Docs](https://vladmandic.github.io/sdnext-docs/) | [WiKi](https://github.com/vladmandic/automatic/wiki) | [Discord](https://discord.com/invite/sd-next-federal-batch-inspectors-1101998836328697867)

### Details for 2025-05-12

- **Compute**
  - **NNCF**  
    - Faster quantization  
    - Faster inference with support for `torch.triton`  
      up to 3.5x faster with INT4 and 2x faster with INT8  
    - New settings: *NNCF -> Group size*  
      default is a balance between performance (higher size) and quality (lower size)  
      0 is default at 64, -1 disables grouping  
  - **ZLUDA**:
    - *warning*: AMD Adrenaline 25.5.1 drivers are NOT COMPATIBLE with ZLUDA
      see [issue](https://github.com/vladmandic/sdnext/issues/3918) for details
  - **ROCm**
    - first working builds of **Torch with ROCm on Windows**  
      highly experimental  
      reach out on Discord if you want to test it  
- **Features**
  - Prompt Enhancer: support for *img2img* workflows  
    in img2img prompt enhancer will first analyze input image and then incorporate user prompt to create enhanced prompt  
  - **FramePack**
    - improve LoRA compatibility  
    - add metadata to video  
  - **UI**
    - ModernUI: support for History tab  
    - ModernUI: support for FramePack tab  
  - **API**  
    - add `/sdapi/v1/framepack` endpoint with full support for FramePack including all optional settings  
      see example: `sd-extension-framepack/create-video.py`  
    - add `/sdapi/v1/checkpoint` endpoint to get info on currently loaded model/checkpoint  
      see example: `cli/api-checkpoint.py`  
    - add `/sdapi/v1/prompt-enhance` endpoint to enhance prompt using LLM  
      see example: `cli/api-enhance.py`  
      supports text, image and video prompts with or without input image  
      *note*: if input image is provided, model should be left at default `gemma-3-4b-it` as most other LLMs do not support hybrid workflows  
- **Fixes**
  - Latent Diffusion Upscale
  - Model load: support SDXL safetensors packaged without VAE  
  - ROCm: disable cuDNN benchmark, fixes slow MIOpen tuning with `torch==2.7`  
  - Extensions: use in-process installer for extensions-builtin, improves startup performance  
  - FramePack: monkey-patch for dynamically installed `av`  
  - Logging: reduce spam while progress is active  
  - LoRA: legacy handler enable/disable  
  - LoRA: force clear-cache on model unload  
  - ADetailer: fix enable/disable  
  - ZLUDA: improve compatibility with older GPUs  

## Update for 2025-05-06

Minor refesh with several bugfixes and updates to core libraries  
Plus new features with **FramePack** and **HiDream-E1**

- **Features**  
  - [FramePack](https://vladmandic.github.io/sdnext-docs/FramePack)  
    add **T2V** mode in addition to **I2V** and **FLF2V**  
    support for new **F1: forward-only** model variant in addition to regular **bi-directional**  
    add **prompt enhance** using VLM: it will analyze input image and then create enhanced prompt based on user prompt and image  
    add **prompt interpolation**, section prompts do not need to match exact video section count  
    and improved performance  
    [Docs](https://vladmandic.github.io/sdnext-docs/FramePack) rewrite!  
  - **Prompt-Enhhance**  
    add **Qwen3** *0.6B/1.7B/4B* models  
    add thinking mode support (for models that have it)  
  - [HiDream-E1](https://huggingface.co/HiDream-ai/HiDream-E1-Full) natural language image-editing model built on HiDream-I1  
    available via  *networks -> models -> reference*  
    *note*: right now HiDream-E1 is limited to 768x768 images, so you must force resize image before running it  
- **Other**  
  - CUDA: set default to `torch==2.7.0` with `cuda==12.8`  
  - ZLUDA: update to `zluda==3.9.4` and `flash-attn-2`  
  - Docker: pre-install `ffmpeg`  
  - Wiki: updated pages: *FramePack, Video, ROCm, ZLUDA, Quantization*  
  - Gallery: support JXL image format  
  - Scheduler: add sigmoid beta scheduler  
  - GitHub: updated issue template  
- **Fixes**  
  - FramePack: correct dtype  
  - NNCF: check dependencies and register quant type  
  - API: refresh checkpoint list  
  - API: vlm-api endpoint  
  - Styles: save style with prompt  
  - Texture tiling: fix apply when switching models  
  - Diffusers: slow initial startup  
  - Gated access: obfuscate and log token used for access  
  - SDXL refiner workflow  
  - Control: t2i-adapter workflow  
  - Control: xs-controlnet workflow  
  - Control: lllite-workflow  
  - Control: refiner workflow with multiple control elements  

## Highlights for 2025-04-28

Another major release with *over 120 commits*!  
Highlights include new [Nunchaku Wiki](https://github.com/vladmandic/sdnext/wiki/Nunchaku) inference engine that allows running FLUX.1 with **3-5x** higher performance!  
And a new [FramePack](https://github.com/vladmandic/sd-extension-framepack) extension for high-quality *I2V* and *FLF2V* video generation with unlimited duration!  

What else?
- New UI **History** tab  
- New models: **Flex.2, LTXVideo-0.9.6, WAN-2.1-14B-FLF2V**, schedulers: **UniPC and LCM FlowMatch**, features: **CFGZero**  
- Major updates to: **NNCF, OpenVINO, ROCm, ZLUDA**  
- Cumulative fixes since last release  

[ReadMe](https://github.com/vladmandic/automatic/blob/master/README.md) | [ChangeLog](https://github.com/vladmandic/automatic/blob/master/CHANGELOG.md) | [Docs](https://vladmandic.github.io/sdnext-docs/) | [WiKi](https://github.com/vladmandic/automatic/wiki) | [Discord](https://discord.com/invite/sd-next-federal-batch-inspectors-1101998836328697867)

## Details for 2025-04-28

- **Features**
  - [Nunchaku](https://github.com/mit-han-lab/nunchaku) inference engine with custom **SVDQuant** 4-bit execution  
    highly experimental and with limited support, but when it works, its magic: **Flux.1 at 6.0 it/s** *(not sec/it)*!  
    basically, it can speed up supported models by 2-5x by using custom quantization and execution engine  
    see [Nunchaku Wiki](https://github.com/vladmandic/sdnext/wiki/Nunchaku) for installation guide and list of supported models & features  
  - [FramePack](https://github.com/vladmandic/sd-extension-framepack) based on **HunyuanVideo-I2V**  
    full support and much more for **Lllyasviel** [FramePack](https://lllyasviel.github.io/frame_pack_gitpage/)  
    implemented as an extension for **SD.Next** (for the moment while dev is ongoing)  
    generate high-quality videos with pretty much unlimited duration and with limited VRAM!  
    install as any other extension and for details see extension [README](https://github.com/vladmandic/sd-extension-framepack/blob/main/README.md)  
    - I2V & FLF2V support with explicit strength controls  
    - complex actions: modify prompts for each section of the video  
    - LoRA support: use normal **HunyuanVideo** LoRAs  
    - decode: use local, tiny or remote VAE  
    - custom models: e.g. replace llama with one of your choice  
    - video: multiple codecs and with hw acceleration, raw export, frame export, frame interpolation  
    - compute: quantization support, new offloading, more configuration options, cross-platform, etc.  
  - [Ostris Flex.2 Preview](https://huggingface.co/ostris/Flex.2-preview)  
    more than a FLUX.1 finetune, FLEX.2 is created from *Flux.1 Schnell -> OpenFlux.1 -> Flex.1-alpha -> Flex.2-preview*  
    and it has universal control and inpainting support built in!  
    supported for text and control workflows  
    when using in control mode, simply choose preprocessor and do not load actual controlnet  
    supported control modes are: *line, pose and depth*  
    available via  *networks -> models -> reference*  
  - [LTXVideo 0.9.6](https://github.com/Lightricks/LTX-Video?tab=readme-ov-file) **T2V** and **I2V**  
    in both **Standard** and **Distilled** variants  
    available in *video tab*
  - [WAN 2.1 14B 720P](https://huggingface.co/Wan-AI/Wan2.1-FLF2V-14B-720P) **FLF2V**  
    new first-to-last image video model from WAN-AI  
    available in *video tab*
  - [CFG-Zero](https://github.com/WeichenFan/CFG-Zero-star) new guidance method optimized for flow-matching models  
    implemented for **FLUX.1, HiDream-I1, SD3.x, CogView4, HunyuanVideo, WanAI**  
    enable and configure in *settings -> pipeline modifiers -> cfg zero*  
    experiment with CFGZero support in XYZ-grid  
- **Optimizations**
  - **NNCF** update to 2.16.0  
    major refactoring of NNCF quantization code  
    new quant types: `INT8_SYM` (new default), `INT4` and `INT4_SYM`  
    quantization support for the convolutional layers on unet models with sym methods  
    pre-load quantization support  
    LoRA support  
    *if you're low on VRAM, NNCF is as close as a catch-all solution*  
  - **OpenVINO** update to 2025.1.0 and Torch to 2.7  
  - **IPEX** update to Torch 2.7  
  - **ROCm** update to Torch 2.7  
  - **HiDream-I1** optimized offloading and prompt-encode caching  
    it now works in 12GB VRAM / 26GB RAM!  
  - **CogView3** and **CogView4** model loader optimizations  
  - **Sana** model loader optimizations
  - add explicit offload after encode prompt  
    configure in *settings -> text encoder -> offload*  
- **UI**  
  - new History tab where you can see all jobs since the server startup  
    and optionally download any of the previously generated images/videos  
    access via *system -> history*  
  - server restart from ui now replaces currently running process  
    instead of trying to reload python modules in-place  
  - add option to enable/disable clip skip  
    disabled by default to avoid issues with frequent incorrect recommendations  
    in *settings -> pipeline modifiers*
  - configurable restore metadata from image to settings and to params  
    in *settings -> image metadata*  
- **API**  
  - new [API Wiki](https://github.com/vladmandic/sdnext/wiki/API)  
  - server will now maintain job history which can be queried via API  
    so you can check previous jobs as well as request any previously generated images/videos  
  - history endpoint: `/sdapi/v1/history?id={id}`  
  - download endpoint: `/file={filename}`  
  - progress api `/sdapi/v1/progress` now also include task id in the response  
- **Other**
  - **OMI** support for sd15/sdxl omi-standard LoRAs
  - text/image/control/video pipeline vs task compatibility check  
  - **HiDream-I1, FLUX.1, SD3.x** add HF gated access auth check  
  - **HiDream-I1** LoRA support  
    currently limited to diffusers-only LoRAs, CivitAI LoRA support is TBD  
  - **HiDream-I1** add LLM info to image metadata  
  - add `model_type` as option for image filename pattern  
  - add **UniPC FlowMatch** scheduler  
  - add **LCM FlowMatch** scheduler  
  - networks: set which networks to skip when scanning civitai  
    in *settings -> networks -> network scan*  
    comma-separate list of regex patterns to skip  
  - ui display reference models with subdued color  
  - xyz grid support bool  
  - do not force gc at end of processing  
  - add `SD_LORA_DUMP` env variable for dev/diag to dump lora/model keys  
- **Wiki**  
  - new *Nunchaku*, *API* pages  
  - updated *HiDream, Quantization, NNCF, Video, Docker, WSL, ZLUDA* pages  
- **Fixes**
  - HunyuanVideo-I2V with latest transformers  
  - NNCF with TE-only quant  
  - ONNX init fix  
  - Quanto with TE/LLM quant  
  - HiDream live preview  
  - FLUX.1 controlnet i2i  
  - SD35 InstantX IP-adapter  
  - OpenVINO device selection
  - xyz grid restore settings  
  - config save unnecessary keys  
  - recursive wildcards  
  - extension installer handling of PYTHONPATH  
  - trace logging  
  - api logging  
  - sd/sdxl-inpaint model loader  
  - settings list display only visible items  
  - checkpoint match when searching for model to load  
  - video vae selection load correct vae

## Update for 2025-04-12

### Highlights for 2025-04-12

Last release was just over a week ago and here we are again with another update as a new high-end image model, [HiDream-I1](https://github.com/vladmandic/sdnext/wiki/HiDream) jumped out and generated a lot of buzz!  
There are quite a few other performance and quality-of-life improvements in this release and 40 commits, so please take a look at the full [ChangeLog](https://github.com/vladmandic/automatic/blob/master/CHANGELOG.md)  

[ReadMe](https://github.com/vladmandic/automatic/blob/master/README.md) | [ChangeLog](https://github.com/vladmandic/automatic/blob/master/CHANGELOG.md) | [Docs](https://vladmandic.github.io/sdnext-docs/) | [WiKi](https://github.com/vladmandic/automatic/wiki) | [Discord](https://discord.com/invite/sd-next-federal-batch-inspectors-1101998836328697867)

### Details for 2025-04-12

- **Models**  
  - [HiDream-I1](https://huggingface.co/HiDream-ai/HiDream-I1-Full) in fast, dev and full variants!  
    new absolutely massive image generative foundation model with **17B** parameters and 4 text-encoders with additional **8.3B** parameters  
    simply select from *networks -> models -> reference*  
    due to size (over 25B params in 58GB), offloading and on-the-fly quantization are pretty much a necessity  
    see [HiDream Wiki page](https://github.com/vladmandic/sdnext/wiki/HiDream) for details  
- **Features**  
  - Custom model loader  
      can be used to load any known diffusion model with default or custom model components  
      in models -> custom tab  
      see docs for details: <https://vladmandic.github.io/sdnext-docs/Loader/>  
    - Pipe: [SoftFill](https://github.com/zacheryvaughn/softfill-pipelines)  
- **Caching**  
  - add `TeaCache` support to *Flux, CogVideoX, Mochi, LTX*  
  - add `FasterCache` support to *WanAI, LTX* (other video models already supported)  
  - add `PyramidAttentionBroadcast` support to *WanAI, LTX* (other video models already supported)  
- **UI**  
  - client polling speeds up and slows down depending if client page is visible or not  
    client polling does not ask for live preview if page is not visible  
    significantly reduces server load if you hide or minimize the page  
  - progress: use batch-count for progress  
  - grid: add of max-rows and max-columns in settings to control grid format  
  - gallery: add max-columns in settings for gradio gallery components  
- **Other**  
  - ZLUDA: add more GPUs to recognized list  
    select in scripts, available for sdxl in inpaint model  
  - LoRA: add option to force-reload LoRA on every generate  
  - settings: add **Model options** sections as placeholder for per-model settings
  - video: update *LTXVideo-0.9.5* pipeline  
  - te loader: allow free-form input in which case sdnext will attempt to load it as hf repo  
  - diag: add get-server-status to UI generate context menu  
  - diag: memory monitor detect gpu swapping  
  - use [hf-xet](https://huggingface.co/blog/xet-on-the-hub) for huggingface downloads where possible  
  - quant: update & fix `optimum-quanto` for transformers  
  - quant: update & fix `torchao`  
  - model load: new setting for model load initial device map  
    can be used to force gpu vs cpu when loading model to avoid oom before model offloading is even activated after load  
- **Changes**  
  - params: Reset default guidance-rescale from 0.7 to 0.0  
  - progress: add additional fields to progress API  
- **Fixes**  
  - styles: resize and bring quick-ui to forward on hover  
  - LoRA: obey configured device when performing calculations  
  - ZLUDA: startup issues  
  - offload: balanced offload remove non-blocking move op  
  - logging: debug causes invalid import  
  - logging: cleanup  
  - ROCm: flash attention repo with navi rotary fix  
  - prompt: prompt scheduling with te caching  
  - ui: progress allow for longer timeouts  
  - internal: cleanup defined pipelines

## Update for 2025-04-03

### Highlights for 2025-04-03

Time for another major release with ~120 commits and [ChangeLog](https://github.com/vladmandic/automatic/blob/master/CHANGELOG.md) that spans several pages!

*Highlights?*  
Video...Brand new Video processing module with support for all latest models: **WAN21, Hunyuan, LTX, Cog, Allegro, Mochi1, Latte1** in both *T2V* and *I2V* workflows  
And combined with *on-the-fly quantization*, support for *Local/Tiny/Remote* VAE, acceleration modules such as *FasterCache or PAB*, and more!  
Models...And support for new models: **CogView-4**, **SANA 1.5**,  

*Plus...*  
- New **Prompt Enhance** using LLM,
- New pipelines such as **InfiniteYou**  
- New **CLiP** models, improvements to **remote VAE**, additional wiki/docs/guides  
- More quantization options and granular control  
- Pretty big performance updates to a) Any model using DiT based architecture due to new caching methods, b) ZLUDA with new attention methods, c) LoRA with much lower memory usage  

[ReadMe](https://github.com/vladmandic/automatic/blob/master/README.md) | [ChangeLog](https://github.com/vladmandic/automatic/blob/master/CHANGELOG.md) | [Docs](https://vladmandic.github.io/sdnext-docs/) | [WiKi](https://github.com/vladmandic/automatic/wiki) | [Discord](https://discord.com/invite/sd-next-federal-batch-inspectors-1101998836328697867)

### Details for 2025-04-03

- **Video tab**  
  - see [Video Wiki](https://github.com/vladmandic/sdnext/wiki/Video) for details!  
  - new top-level tab, replaces previous *video* script in text/image tabs  
    old scripts are still present, but will be removed in the future  
  - support for all latest models:  
    - [Hunyuan](https://huggingface.co/Tencent/HunyuanVideo): *HunyuanVideo, FastHunyuan, SkyReels* | *T2V, I2V*  
    - [WAN21](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B-Diffusers): *1.3B, 14B* | *T2V, I2V*  
    - [LTXVideo](https://huggingface.co/Lightricks/LTX-Video): *0.9.0, 0.9.1, 0.9.5* | *T2V, I2V*  
    - [CogVideoX](https://huggingface.co/THUDM/CogVideoX-5b): *2B, 5B* | *T2V, I2V*  
    - [Allegro](https://huggingface.co/rhymes-ai/Allegro): *T2V*  
    - [Mochi1](https://huggingface.co/genmo/mochi-1-preview): *T2V*  
    - [Latte1](https://huggingface.co/maxin-cn/Latte-1): *T2V  
  - decoding:  
    - **Default**: use vae from model  
    - **Tiny VAE**: support for *Hunyuan, WAN, Mochi*  
    - **Remote VAE**: support for *Hunyuan*  
  - **LoRA**
    - support for *Hunyuan, LTX, WAN, Mochi, Cog*  
    - add option to apply LoRA directly on GPU or use CPU first in low-memory scenarios  
    - improve metadata and preview parallel fetch  
    - support for mp4 so first frame is extracted as used as lora preview  
  - additional key points:  
    - all models are auto-downloaded upon first use  
      uses *system paths -> huggingface* folder  
    - support for many video types  
    - optional video interpolation while creating video files  
    - optional video preview in ui  
      present if video output is selected  
    - support for balanced offloading and model offloading  
      uses system settings  
    - on-the-fly quantization: *BnB, Quanto, TorchAO*  
      uses system settings, granular for *transformer* and *text-encoder* separately  
    - different video models support different video resolutions, frame counts, etc.  
      and may require specific settings - see model links for details  
    - see *ToDo/Limitations* section for additional notes  
- **Models & Pipelines**  
  - [THUDM CogView 4](https://huggingface.co/THUDM/CogView4-6B) **6B** variant  
    new foundation model for image generation based o GLM-4 text encoder and a flow-based diffusion transformer  
    fully supports offloading and on-the-fly quantization  
    simply select from *networks -> models -> reference*  
    *note* cogview4 is compatible with flowmatching samplers  
  - [NVLabs SANA 1.5](https://huggingface.co/Efficient-Large-Model/SANA1.5_4.8B_1024px_diffusers) in **1.6B**, **4.8B** and [Sprint](https://huggingface.co/Efficient-Large-Model/Sana_Sprint_1.6B_1024px_diffusers) variations  
    big update to previous SANA model  
    fully supports offloading and on-the-fly quantization  
    simply select from *networks -> models -> reference*  
  - [ByteDance InfiniteYou](https://github.com/bytedance/InfiniteYou/): Flexible Photo Recrafting While Preserving Your Identity  
    face-transfer model for FLUX.1  
    select from *Scripts -> InfiniteYou*  
    its large, ~12GB on top of FLUX.1 base model so make sure you have offloading and quantization setup  
    *note* model will be auto-downloaded on first use  
  - New [zer0int CLiP-L](https://huggingface.co/zer0int/CLIP-Registers-Gated_MLP-ViT-L-14) models:  
    download text encoders into folder set in settings -> system paths -> text encoders (default is *models/Text-encoder*)  
    load using *settings -> text encoder*  
    *tip*: add *sd_text_encoder* to your *settings -> user interface -> quicksettings* list to have it appear at the top of the ui  
- **Prompt Enhance**  
  - see [Prompt Enhance Wiki](https://github.com/vladmandic/sdnext/wiki/Prompt-Enhance) for details!  
  - new built-in extension available in text/image/control tabs  
  - can be used to manually or automatically enhance prompts using LLM  
  - built-in presets for **Gemma-3, Qwen-2.5, Phi-4, Llama-3.2, SmolLM2, Dolphin-3**  
  - support for custom models  
    load any models hosted on huggingface  
    load either model in huggingface format or `gguf` format  
    *note*: any hf model in `transformers.AutoModelForCausalLM` standard should work  
    *note*: not all model architecture are supported for `gguf` format  
  - models are auto-downloaded on first use  
  - support quantization and offloading  
  - auto-detect censored output  
  - debug using `SD_LLM_DEBUG=true` env variable  
- **Acceleration**  
  - Support for most DiT-based models, for example: *FLUX.1, SD35, Hunyuan, Mochi, Latte, Allegro, Cog*  
  - Enable and configure in *Settings -> Pipeline modifiers*  
  - [FasterCache](https://huggingface.co/papers/2410.19355)  
  - [PyramidAttentionBroadcast](https://huggingface.co/papers/2408.12588)  
- **Remote VAE**  
  - add support for remote vae encode in addition to remote vae decode  
  - used by *img2img, inpaint, hires, detailer*  
  - remote vae encode is disabled by default, you can enable it in *settings -> variable auto-encoder*  
  - add remote vae info to metadata, thanks @iDeNoh  
  - remote vae use `scaling_factor` and `shift_factor`  
- **Caption/VLM**  
  - [Google Gemma 3](https://huggingface.co/google/gemma-3-4b-it) 4B  
    simply select from list of available models in caption tab  
  - [ByteDance/Sa2VA](https://huggingface.co/ByteDance/Sa2VA-1B) 1B, 4B  
    simply select from list of available models in caption tab  
  - add option to set system prompt for vlm models that support it: *Gemma, Smol, Qwen*  
- [NudeNet](https://github.com/vladmandic/sd-extension-nudenet/) extension updates  
  - add detection of prompt language and alphabet and filter based on those values  
  - add image policy checks using `LlavaGuard` VLM to detect policy violations (and reasons)  
    against top-10 standard harmful content categories  
  - add banned words/expressions check against prompt variations  
- **LoRA**
  - enable memory cache by default  
  - significantly reduce memory usage  
  - improve performance  
  - improve detection of lora changes  
  - unload lora only when changes are detected  
  - refactor code for modularity  
- **IPEX**  
  - add `--upgrade` to torch_command when using `--use-nightly`  
  - add xpu to profiler  
  - fix untyped_storage, torch.eye and torch.cuda.device ops  
  - fix torch 2.7 compatibility  
  - fix performance with balanced offload  
  - fix triton and torch.compile  
- **ROCm**
  - add `--upgrade` to torch_command when using `--use-nightly`  
  - disable fp16 for gfx1102 (rx 7600 and rx 7500 series) gpus  
- **ZLUDA**  
  - [triton for ZLUDA v3.9.2](https://github.com/vladmandic/sdnext/wiki/ZLUDA#how-to-enable-triton)  
    - `torch.compile` is now available  
    - Flash Attention 2 is now available  
- **Other**  
  - **Command line** new option `--monitor PERIOD` to monitor CPU and GPU memory ever n seconds  
  - **Upscale** new [asymmetric vae v2](https://huggingface.co/Heasterian/AsymmetricAutoencoderKLUpscaler_v2) upscaling method  
  - **Upscale** new experimental support for `libvips` upscaling  
  - **Quantization** add support for `optimum-quanto` on-the-fly quantization during load for all models  
    note: previous method for quanto is still valid and is noted in settings as post-load quantization  
  - **Quantization** add support to **CogView-3Plus**  
  - **Default values** rename vae, unet and text-encoder settings *None* to *Default* to avoid confusion  
  - **Detailer**: add *renoise* option to increase/decrease noise during detailer pass  
    which can help with improving level of details
  - **CLI**: add `cli/api-grid.py` which can generate grids using params-from-file for x/y axis  
  - **Samplers** add ability to set sigma adjustment for each sampler  
  - **ModernUI** updates  
  - **CSS** updates  
  - **Video** interpolate do not skip duplicate frames  
  - **Settings UI** full refactor  
  - **Settings UI** vertical/dirty indicator restores to default setting instead to previous value  
  - update `diffusers` and other requirements  
- **Wiki/Docs**  
  - updated [Models](https://github.com/vladmandic/sdnext/wiki/Models) info  
  - new [Video](https://github.com/vladmandic/sdnext/wiki/Video) guide  
  - new [Caption](https://github.com/vladmandic/sdnext/wiki/Caption) guide  
  - new [VAE](https://github.com/vladmandic/sdnext/wiki/VAE) guide  
  - updated [SD3](https://github.com/vladmandic/sdnext/wiki/SD3) guide  
  - updated [ZLUDA](https://github.com/vladmandic/sdnext/wiki/ZLUDA) guide  
  - updated [OpenVINO](https://github.com/vladmandic/sdnext/wiki/OpenVINO) guide  
  - updated [AMD-ROCm](https://github.com/vladmandic/sdnext/wiki/AMD-ROCm) guide  
  - updated [Intel-ARC](https://github.com/vladmandic/sdnext/wiki/Intel-ARC) guide  
- **Fixes**  
  - fix installer not starting when older version of `rich` is installed  
  - fix circular imports when debug flags are enabled  
  - fix cuda errors with *directml*  
  - fix memory stats not displaying the ram usage  
  - fix **RunPod** memory limit reporting  
  - fix flux ipadapter with start/stop values  
  - fix progress api `eta_relative`  
  - fix `insightface` loader  
  - fix remove vae for flux.1  
  - guard against git returining invalid timestamp  
  - fix hires with latent upscale  
  - fix legacy diffusion latent upscalers  
  - fix upscaler selection in postprocessing  
  - fix sd35 with batch processing  
  - fix extra networks cover and inline views  
  - fix token counter error style with modernui  
  - fix sampler metadata when using default sampler  
  - fix paste incorrect float to int cast  
  - fix server restart from ui  
  - fix style apply params  
  - fix `wan22-i2v`  
  - do not allow edit of built-in styles  
  - improve lora compatibility with balanced offload  

## Update for 2025-02-28

Primarily a hotfix/service release plus few UI improvements and one exciting new feature: Remote-VAE!

- **Remote Decode**  
  - final step of image generate, VAE decode, is by far the most memory intensive operation and can easily result in out-of-memory errors  
    what can be done? Well, *Huggingface* is now providing *free-of-charge* **remote-VAE-decode** service!  
  - how to use? previous *Full quality* option in UI is replaced with VAE type selector: *Full, Tiny, Remote*  
    currently supports SD15, SDXL and FLUX.1 with more models expected in the near future  
    depending on your bandwidth select mode in *settings -> vae -> raw/png/jpg*  
    if remote processing fails SD.Next will fallback to using normal VAE decode process  
    *privacy note*: only passed item is final latent itself without any user or generate information and latent is not stored in the cloud  
- **UI**
  - modern ui reorg main tab  
    improve styling, improve scripts/extensions interface and separate ipadapters  
  - additional ui hints  
- **Other**  
  - add `--extensions-dir` cli arg and `SD_EXTENSIONSDIR` env variable to specify extensions directory  
  - update `zluda==3.9.0`
- **Fixes**  
  - skip trying to register legacy/incompatibile extensions in control ui  
  - add additional scripts/extensions callbacks  
  - remove ui splash screen on auth fail  
  - log full config path, full log path, system name, extensions path
  - zluda hotfixes  
  - zluda force sync  
  - fix torch import on compile  
  - infotext parser force delimiter before params  
  - handle pipeline class switch errors  
  - improve extensions options compatibility  
  - fix flux on ipex  
  - disable fp64 emulation on ipex  

## Update for 2025-02-18

### Highlight for 2025-02-18

We're back with another update with nearly 100 commits!  
- Starting with massive UI update with full [localization](https://vladmandic.github.io/sdnext-docs/Locale/) for 8 languages  
  and 100+ new [hints](https://vladmandic.github.io/sdnext-docs/Hints/)  
- Big update to [Docker](https://vladmandic.github.io/sdnext-docs/Docker/) containers  
  with support for all major compute platforms  
- A lot of [outpainting](https://vladmandic.github.io/sdnext-docs/Outpaint/) goodies  
- Support for new models: [AlphaVLLM Lumina 2](https://github.com/Alpha-VLLM/Lumina-Image-2.0) and [Ostris Flex.1-Alpha](https://huggingface.co/ostris/Flex.1-alpha)  
- And new **Mixture-of-Diffusers** regional prompting & tiling pipeline  
- Follow-up to last weeks **interrogate/captioning** rewrite  
  now with redesigned captioning UI, batch support, and much more  
  plus **JoyTag**, **JoyCaption**, **PaliGemma**, **ToriiGate**, **Ovis2** added to list of supported models  
- Some changes to **prompt parsing** to allow more control as well as  
  more flexibility when mouting SDNext server to custom URL  
- Of course, cumulative fixes...  

*...and more* - see [changelog](https://github.com/vladmandic/sdnext/blob/dev/CHANGELOG.md) for full details!  

### Details for 2025-02-20

- **User Interface**  
  - **Hints**  
    - added/updated 100+ ui hints!  
    - [hints](https://vladmandic.github.io/sdnext-docs/Hints/) documentation and contribution guide  
  - **Localization**  
    - full ui localization!  
      *english, croatian, spanish, french, italian, portuguese, chinese, japanese, korean, russian*  
    - set in *settings -> user interface -> language*  
    - [localization](https://vladmandic.github.io/sdnext-docs/Locale/) documentation  
  - **UI**  
    - force browser cache-invalidate on page load  
    - configurable request timeout  
    - modernui improve gallery styling  
    - modernui improve networks styling  
    - modernui support variable card size  
- **Docs**  
  - New [Outpaint](https://vladmandic.github.io/sdnext-docs/Outpaint/) step-by-step guide  
  - Updated [Docker](https://github.com/vladmandic/sdnext/wiki/Docker) guide  
    includes build and publish and both local and cloud examples  
- **Models**  
  - [AlphaVLLM Lumina 2](https://github.com/Alpha-VLLM/Lumina-Image-2.0)  
    new foundation model for image generation based o Gemma-2-2B text encoder and a flow-based diffusion transformer  
    fully supports offloading and on-the-fly quantization  
    simply select from *networks -> models -> reference*  
  - [Ostris Flex.1-Alpha](https://huggingface.co/ostris/Flex.1-alpha)  
    originally based on *Flux.1-Schnell*, but retrained and with different architecture  
    result is model smaller than *Flux.1-Dev*, but with similar capabilities  
    fully supports offloading and on-the-fly quantization  
    simply select from *networks -> models -> reference*  
- **Functions**  
  - [Mixture-of-Diffusers](https://huggingface.co/posts/elismasilva/251775641926329)  
    Regional tiling type of a solution for SDXL models  
    select from *scripts -> mixture of diffusers*  
  - [Automatic Color Inpaint]  
    Automatically creates mask based on selected color and triggers inpaint  
    simply select in *scripts -> automatic color inpaint* when in img2img mode  
  - [RAS: Region-Adaptive Sampling](https://github.com/microsoft/RAS) *experimental*  
    Speeds up SD3.5 models by sampling only regions of interest  
    Enable in *settings -> pipeline modifiers -> ras*  
- **Interrogate/Captioning**  
  - Redesigned captioning UI  
    split from Process tab into separate tab  
    split `clip` vs `vlm` models processing  
    direct *send-to* buttons on all tabs: txt/img/ctrl->process/caption, process/caption->txt/img/ctrl  
  - Advanced params:
    VLM: *max-tokens, num-beams, temperature, top-k, top-p, do-sample*  
    CLiP: *min-length, max-length, chunk-size, min-flavors, max-flavors, flavor-count, num-beams*  
    params are auto-saved in `config.json` and used when using quick interrogate  
    params that are set to 0 mean use model defaults  
  - Batch processing: VLM and CLiP  
    for example, can be used to caption your training dataset in one go  
    add option to append to captions file, can be used to run multiple captioning models in sequence  
    add option to run recursively on all subfolders  
    add progress bar  
  - Add additional VLM models:  
    [JoyTag](https://huggingface.co/fancyfeast/joytag)  
    [JoyCaption 2](https://huggingface.co/fancyfeast/llama-joycaption-alpha-two-hf-llava)  
    [Google PaliGemma 2](https://huggingface.co/google/paligemma2-3b-pt-224) 3B  
    [ToriiGate 0.4](https://huggingface.co/Minthy/ToriiGate-v0.4-7B) 7B  
    [AIDC Ovis2](https://huggingface.co/AIDC-AI/Ovis2-1B) 1B/2B/4B  
  - *Note* some models require `flash-attn` to be installed  
    due to binary/build dependencies, it should not be done automatically,  
    see [flash-attn](https://github.com/Dao-AILab/flash-attention) for installation instructions  
- **Docker**  
  - updated **CUDA** receipe to `torch==2.6.0` with `cuda==12.6` and add prebuilt image  
  - added **ROCm** receipe and prebuilt image  
  - added **IPEX** receipe and add prebuilt image  
  - added **OpenVINO** receipe and prebuilt image  
- **System**  
  - improve **python==3.12** compatibility  
  - **Torch**  
    - for **zluda** set default to `torch==2.6.0+cu118`  
    - for **openvino** set default to `torch==2.6.0+cpu`  
  - **OpenVINO**  
    - update to `openvino==2025.0.0`  
    - improve upscaler compatibility  
    - enable upscaler compile by default  
    - fix shape mismatch errors on too many resolution changes  
  - **ZLUDA**  
    - update to `zluda==3.8.8`  
- **Other**  
  - **Asymmetric tiling**  
    allows for configurable image tiling for x/y axis separately  
    enable in *scripts -> asymmetric tiling*  
    *note*: traditional symmetric tiling is achieved by setting circular mode for both x and y  
  - **Styles**  
    ability to save and/or restore prompts before or after parsing of wildcards  
    set in *settings -> networks -> styles*  
  - **Access tokens**  
    persist *models -> hugginface -> token*  
    persist *models -> civitai -> token*  
  - global switch to lancosz method for all interal resize ops and bicubic for interpolation ops  
  - **Text encoder**  
    add advanced per-model options for text encoder  
    set in *settings -> text encoder -> Optional*  
  - **Subpath**  
    allow setting additional mount subpath over which server url will be accessible  
    set in *settings -> user interface*  
  - **Prompt parsing**  
    better handling of prompt parsing when using masking char `\`  
- **Fixes**  
  - update torch nightly urls  
  - docs/wiki always use relative links  
  - ui use correct timezone for log display  
  - ui improve settings search behavior  
  - ui log scroll to bottom  
  - ui fix send to inpaint/sketch  
  - modernui add control init image toggle  
  - modernui fix sampler advanced options  
  - outpaint fixes  
  - validate output before hires/refine  
  - scheduler fix sigma index out of bounds  
  - force pydantic version reinstall/reload  
  - multi-unit when using controlnet-union  
  - pulid with hidiffusion  
  - api: stricter access control  
  - api: universal handle mount subpaths  

## Update for 2025-02-05

- refresh dev/master branches

## Update for 2025-02-04

### Highlights for 2025-02-04

Just one week after latest release and what a week it was with over 50 commits!  

*What's New?*  
- Rehosted core repo to new [home](https://github.com/vladmandic/sdnext)  
- Switched to using `torch==2.6.0` and added support for `nightly` builds required for **nVidia Blackwell** GPUs  
- Completely new **interrogate/captioning**, now supporting 150+ **OpenCLiP** models and 20+ built-in **VLMs**  
- Support for **new VLMs**, New SOTA **background removal**  
- Other: *torch tunable ops, extra networks search/filter, balanced offload, prompt parser, configurable tracebacks, etc.*  
- Cumulative fixes...  

### Details for 2025-02-04

- **GitHub**
  - rename core repo from <https://github.com/vladmandic/automatic> to <https://github.com/vladmandic/sdnext>  
    old repo url should automatically redirect to new one for seamless transition and in-place upgrades  
    all internal links have been updated  
    wiki content and docs site have been updated  
- **Docs**:
  - Updated [Debugging guide](https://github.com/vladmandic/automatic/wiki/Debug)  
- **Torch**:
  - for **cuda** set default to `torch==2.6.0+cu126`  
    for **rocm** set default to `torch==2.6.0+rocm6.2.4`  
    for **ipex** set default to `torch==2.6.0+xpu`  
    *note*: to avoid disruptions sdnext does not perform torch install during in-place upgrades  
    to force torch upgrade, either start with new installation or use `--reinstall` flag  
  - support for torch **nightly** builds and nvidia **blackwell** gpus!  
    use `--use-nightly` flag to install torch nightly builds  
    current defaults to `torch==2.7.0+cu128` prerelease  
    *note*: nightly builds are required for blackwell gpus  
  - add support for torch **tunable ops**, this can speed up operations by up to *10-30%* on some platforms  
    set in *settings -> backend settings -> torch options* and *settings -> system paths -> tunable ops cache*  
  - add support for stream-loading, this can speed up model loading when models are located on network drives  
    set in *settings -> models & loading -> model load using streams*  
  - enhanced error logging  
- **Interrogate/Captioning**  
  - single interrogate button for every input or output image  
  - behavior of interrogate configurable in *settings -> interrogate*  
    with detailed defaults for each model type also configurable  
  - select between 150+ *OpenCLiP* supported models, 20+ built-in *VLMs*, *DeepDanbooru*  
  - **VLM**: now that we can use VLMs freely, we've also added support for few more out-of-the-box  
    [Alibaba Qwen VL2](https://huggingface.co/Qwen/Qwen2-VL-2B), [Huggingface Smol VL2](https://huggingface.co/HuggingFaceTB/SmolVLM-Instruct), [ToriiGate 0.4](https://huggingface.co/Minthy/ToriiGate-v0.4-2B)  
- **Postprocess**  
  - new sota remove background model: [BEN2](https://huggingface.co/PramaLLC/BEN2)  
    select in *process -> remove background* or enable postprocessing for txt2img/img2img operations  
- **Other**:
  - **networks**: imporove search/filter and add visual indicators for types  
  - **balanced offload** new defaults: *lowvram/4gb min threshold: 0, medvram/8gb min threshold: 0, default min threshold 0.25*  
  - **prompt parser**: log stats with tokens, sections and min/avg/max weights  
  - **prompt parser**: add setting to ignore line breaks in prompt  
    set in *settings -> text encoder -> use line breaks*  
  - **visual query**: add list of predefined system prompts  
  - **onnx**: allow manually specifying `onnxruntime` package
    set env variable `ONNXRUNTIME_COMMAND` to override default package installation  
  - **nvml cli**: run nvidia-management-lib interrogate from cli  
    already available in ui in generate -> right click -> nvidia  
    > python modules/api/nvml.py  
- **Refactor**:
  - unified trace handler with configurable tracebacks  
  - refactor interrogate/analyze/vqa code  
- **Fixes**:  
  - photomaker with offloading  
  - photomaker with refine  
  - detailer with faceid modules  
  - detailer restore pipeline before run  
  - fix `python==3.9` compatibility  
  - improve `python>=3.12.3` compatibility
  - handle invalid `triton` on Linux  
  - correct library import order  
  - update requirements  
  - calculate dyn atten bmm slice rate  
  - dwpose update and patch `mmengine` installer  
  - ipex device wrapper with adetailer  
  - openvino error handling  
  - relax python version checks for rocm  
  - simplify and improve file wildcard matching  
  - fix `rich` version  
  - add cn active label

## Update for 2025-01-29

### Highlights for 2025-01-29

Two weeks since last release, time for update!  

*What's New?*  
- New **Detailer** functionality including ability to use several new  
  face-restore models: *RestoreFormer, CodeFormer, GFPGan, GPEN-BFR*
- Support for new models/pipelines:  
  face-swapper with **Photomaker-v2** and video with **Fast-Hunyuan**  
- Support for several new optimizations and accelerations:  
  Many **IPEX** improvements, native *torch fp8* support,  
  support for **PAB:Pyramid-attention-broadcast**, **ParaAttention** and **PerFlow**  
- Fully built-in both model **merge weights** as well as model **merge component**  
  Finally replace that pesky VAE in your favorite model with a fixed one!  
- Improved remote access control and reliability as well as running inside containers  
- And of course, hotfixes for all reported issues...  

### Details for 2025-01-29

- **Contributing**:  
  - if you'd like to contribute, please see updated [contributing](https://github.com/vladmandic/automatic/blob/dev/CONTRIBUTING) guidelines
- **Model Merge**
  - replace model components and merge LoRAs  
    in addition to existing model weights merge support  
    now also having ability to replace model components and merge LoRAs  
    you can also test merges in-memory without needing to save to disk at all  
    and you can also use it to convert diffusers to safetensors if you want  
    *example*: replace vae in your favorite model with a fixed one? replace text encoder? etc.  
    *note*: limited to sdxl for now, additional models can be added depending on popularity  
- **Detailer**:  
  - in addition as standard behavior of detect & run-generate, it can now also run face-restore models  
  - included models are: *CodeFormer, RestoreFormer, GFPGan, GPEN-BFR*  
- **Face**:  
  - new [PhotoMaker v2](https://huggingface.co/TencentARC/PhotoMaker-V2) and reimplemented [PhotoMaker v1](https://huggingface.co/TencentARC/PhotoMaker)  
    compatible with sdxl models, generates pretty good results and its faster than most other methods  
    select under *scripts -> face -> photomaker*  
  - new [ReSwapper](https://github.com/somanchiu/ReSwapper)  
    todo: experimental-only and unfinished, only noting in changelog for future reference  
- **Video**  
  - **hunyuan video** support for [FastHunyuan](https://huggingface.co/FastVideo/FastHunyuan)  
    simply select model variant and set appropriate parameters  
    recommended: sampler-shift=17, steps=6, resolution=720x1280, frames=125, guidance>6.0  
- [PAB: Pyramid Attention Broadcast](https://oahzxl.github.io/PAB/)  
  - speed up generation by caching attention results between steps  
  - enable in *settings -> pipeline modifiers -> pab*  
  - adjust settings as needed: wider timestep range means more acceleration, but higher accuracy drop  
  - compatible with most `transformer` based models: e.g. flux.1, hunyuan-video, lyx-video, mochi, etc.
- [ParaAttention](https://github.com/chengzeyi/ParaAttention)
  - first-block caching that can significantly speed up generation by dynamically reusing partial outputs between steps  
  - available for: flux, hunyuan-video, ltx-video, mochi  
  - enable in *settings -> pipeline modifiers -> para-attention*  
  - adjust residual diff threshold to balance the speedup and the accuracy:  
    higher values leads to more cache hits and speedups, but might also lead to a higher accuracy drop  
- **IPEX**
  - enable force attention slicing, fp64 emulation, jit cache  
  - use the us server by default on linux  
  - use pytorch test branch on windows  
  - extend the supported python versions  
  - improve sdpa dynamic attention  
- **Torch FP8**
  - uses torch `float8_e4m3fn` or `float8_e5m2` as data storage and performs dynamic upcasting to compute `dtype` as needed  
  - compatible with most `unet` and `transformer` based models: e.g. *sd15, sdxl, sd35, flux.1, hunyuan-video, ltx-video, etc.*  
    this is alternative to `bnb`/`quanto`/`torchao` quantization on models/platforms/gpus where those libraries are not available  
  - enable in *settings -> quantization -> layerwise casting*  
- [PerFlow](https://github.com/magic-research/piecewise-rectified-flow)  
  - piecewise rectified flow as model acceleration  
  - use `perflow` scheduler combined with one of the available pre-trained [models](https://huggingface.co/hansyan)  
- **Other**:  
  - **upscale**: new [asymmetric vae](https://huggingface.co/Heasterian/AsymmetricAutoencoderKLUpscaler) upscaling method
  - **gallery**: add http fallback for slow/unreliable links  
  - **splash**: add legacy mode indicator on splash screen  
  - **network**: extract thumbnail from model metadata if present  
  - **network**: setting value to disable use of reference models  
- **Refactor**:  
  - **upscale**: code refactor to unify latent, resize and model based upscalers  
  - **loader**: ability to run in-memory models  
  - **schedulers**: ability to create model-less schedulers  
  - **quantization**: code refactor into dedicated module  
  - **dynamic attention sdpa**: more correct implementation and new trigger rate control  
- **Remote access**:  
  - perform auth check on ui startup  
  - unified standard and modern-ui authentication method & cleanup auth logging  
  - detect & report local/external/public ip addresses if using `listen` mode  
  - detect *docker* enforced limits instead of system limits if running in a container  
  - warn if using public interface without authentication  
- **Fixes**:  
  - non-full vae decode  
  - send-to image transfer  
  - sana vae tiling  
  - increase gallery timeouts  
  - update ui element ids  
  - modernui use local font  
  - unique font family registration  
  - mochi video number of frames  
  - mark large models that should offload  
  - avoid repeated optimum-quanto installation  
  - avoid reinstalling bnb if not cuda  
  - image metadata civitai compatibility  
  - xyz grid handle invalid values  
  - omnigen pipeline handle float seeds  
  - correct logging of docker status on logs, thanks @kmscode  
  - fix omnigen  
  - fix docker status reporting  
  - vlm/vqa with moondream2  
  - rocm do not override triton installation  
  - port streaming model load to diffusers  

## Update for 2025-01-15

### Highlights for 2025-01-15

Two weeks since last release, time for update!  
This time a bit shorter highligh reel as this is primarily a service release, but still there is more than few updates  
*(actually, there are ~60 commits, so its not that tiny)*  

*What's New?"  
- Large [Wiki](https://github.com/vladmandic/automatic/wiki)/[Docs](https://vladmandic.github.io/sdnext-docs/) updates  
- New models: **Allegro Video**, new pipelines: **PixelSmith**, updates: **Hunyuan-Video**, **LTX-Video**, **Sana 4k**  
- New version for **ZLUDA**  
- New features in **Detailer**, **XYZ grid**, **Sysinfo**, **Logging**, **Schedulers**, **Video save/create**  
- And a tons of hotfixes...  

### Details for 2025-01-15

- [Wiki/Docs](https://vladmandic.github.io/sdnext-docs/):
  - updated: Detailer, Install, Update, Debug, Control-HowTo, ZLUDA  
- [Allegro Video](https://huggingface.co/rhymes-ai/Allegro)  
  - optimizations: full offload and quantization support  
  - *reference values*: width 1280 height 720 frames 88 steps 100 guidance 7.5  
  - *note*: allegro model is really sensitive to input width/height/frames/steps  
    and may result in completely corrupt output if those are not within expected range  
- [PixelSmith](https://github.com/Thanos-DB/Pixelsmith/)
  - available for SD-XL in txt2img and img2img workflows
  - select from *scripts -> pixelsmith*  
- [Hunyuan Video](https://github.com/Tencent/HunyuanVideo) LoRA support
  - example: <https://huggingface.co/Cseti/HunyuanVideo-LoRA-Arcane_Jinx-v1>
- [LTX Video](https://github.com/Lightricks/LTX-Video) framewise decoding  
  - enabled by default, allows generating longer videos with reduced memory requirements  
- [Sana 4k](https://huggingface.co/Efficient-Large-Model/Sana_1600M_4Kpx_BF16_diffusers)  
  - new Sana variation with support of directly generating 4k images  
  - simply select from *networks -> models -> reference*  
  - tip: enable vae tiling when generating very large images  
- **Logging**:
  - reverted enable debug by default  
  - updated [debug wiki](https://github.com/vladmandic/automatic/wiki/debug)  
  - sort logged timers by duration  
  - allow min duration env variable for timers: `SD_MIN_TIMER=0.1` (default)  
  - update installer messages  
- **Refactor**:
  - refactored progress monitoring, job updates and live preview  
  - improved metadata save and restore  
  - startup tracing and optimizations  
  - threading load locks on model loads  
  - refactor native vs legacy model loader  
  - video save/create
- **Schedulers**:
  - [TDD](https://github.com/RedAIGC/Target-Driven-Distillation) new super-fast scheduler that can generate images in 4-8 steps  
    recommended to use with [TDD LoRA](https://huggingface.co/RED-AIGC/TDD/tree/main)  
- **Detailer**:
  - add explicit detailer prompt and negative prompt  
  - add explicit detailer steps setting  
  - move steps, strength, prompt, negative from settings into ui params  
  - set/restore detailer metadata  
  - new [detailer wiki](https://github.com/vladmandic/automatic/wiki/Detailer)
- **Preview**
  - since different TAESD versions produce different results and latest is not necessarily greatest  
    you can choose TAESD version in settings -> live preview  
    also added is support for another finetuned version of TAESD [Hybrid TinyVAE](https://huggingface.co/cqyan/hybrid-sd-tinyvae-xl)  
- **Video**  
  - all video create/save code is now unified  
  - add support for video formats: GIF, PNG, MP4/MP4V, MP4/AVC1, MP4/JVT3, MKV/H264, AVI/DIVX, AVI/RGBA, MJPEG/MJPG, MPG/MPG1, AVR/AVR1
  - *note*: video format support is platform dependent and not all formats may be available on all platforms
  - *note*: avc1 and h264 need custom opencv due to oss licensing issues  
- **ZLUDA** v3.8.7  
  - new runtime compiler implementation: complex types, JIT are now available  
  - fast fourier transformation is implemented  
  - experimental BLASLt support via nightly build  
    - set `ZLUDA_NIGHTLY=1` to install nightly ZLUDA: newer torch such as 2.4.x (default) and 2.5.x are now available  
    - requirements: unofficial hipBLASLt  
- **Other**
  - **XYZ Grid**: add prompt search&replace options: *primary, refine, detailer, all*
  - **SysInfo**: update to collected data and benchmarks  
- **Fixes**:
  - explict clear caches on model load  
  - lock adetailer commit: `#a89c01d`  
  - xyzgrid progress calculation  
  - xyzgrid detailer
  - vae tiling use default value if not set  
  - sd35 img2img
  - samplers test for scale noise before using  
  - scheduler api  
  - sampler create error handling  
  - controlnet with hires  
  - controlnet with batch count  
  - apply settings skip hidden settings  
  - lora diffusers method apply only once  
  - lora diffusers method set prompt tags and metadata  
  - flux support on-the-fly quantization for bnb of unet only  
  - control restore pipeline before running hires  
  - restore args after batch run  
  - flux controlnet  
  - zluda installer  
  - control inherit parent pipe settings  
  - control logging  
  - hf cache folder settings  
  - fluxfill should not require base model

## Update for 2024-12-31

NYE refresh release with quite a few optimizatios and bug fixes...  
Commit hash: `master: #dcfc9f3` `dev: #935cac6`  

- **LoRA**:  
  - LoRA load/apply/unapply methods have been changed in 12/2024 Xmass release and further tuned in this release
  - for details on available methods, see <https://github.com/vladmandic/automatic/wiki/Lora#lora-loader>  
  - **Sana** support  
  - quantized models support  
  - add fuse support with on-demand apply/unapply (new default)  
  - add legacy option in *settings -> networks*  
- **HunyuanVideo**:  
  - optimizations: full offload, quantization and tiling support  
- **LTXVideo**:  
  - optimizations: full offload, quantization and tiling support  
  - [TeaCache](https://github.com/ali-vilab/TeaCache/blob/main/TeaCache4LTX-Video/README.md) integration  
- **VAE**:  
  - tiling granular options in *settings -> Variational Auto Encoder*  
- **UI**:  
  - live preview optimizations and error handling  
  - live preview high quality output, thanks @Disty0  
  - CSS optimizations when log view is disabled  
- **Samplers**:  
  - add flow shift options and separate dynamic thresholding from dynamic shifting  
  - autodetect matching sigma capabilities  
- **API**:  
  - better default values for generate  
- **Refactor**:  
  - remove all LDM imports if running in native mode  
  - startup optimizatios  
- **Torch**:  
  - support for `torch==2.6.0`  
- **OpenVINO**:  
  - disable re-compile on resolution change  
  - fix shape mismatch on resolution change  
- **Fixes**:  
  - flux pipeline switches: txt/img/inpaint  
  - flux custom unet loader for bnb  
  - flux do not requantize already quantized model
  - interrogate caption with T5  
  - on-the-fly quantization using TorchAO  
  - remove concurrent preview requests  
  - xyz grid recover on error  
  - hires batch  
  - sdxl refiner  
  - increase progress timeout
  - kandinsky matmul  
  - do not show disabled networks  
  - enable debug logging by default
  - image width/height calculation when doing img2img  
  - corrections with batch processing  
  - hires with refiner prompt and batch processing  
  - processing with nested calls  
  - ui networks initial sort  
  - esrgan on cpu devices  

## Update for 2024-12-24

### Highlights for 2024-12-24

### SD.Next Xmass edition: *What's new?*

While we have several new supported models, workflows and tools, this release is primarily about *quality-of-life improvements*:  
- New memory management engine  
  list of changes that went into this one is long: changes to GPU offloading, brand new LoRA loader, system memory management, on-the-fly quantization, improved gguf loader, etc.  
  but main goal is enabling modern large models to run on standard consumer GPUs  
  without performance hits typically associated with aggressive memory swapping and needs for constant manual tweaks  
- New [documentation website](https://vladmandic.github.io/sdnext-docs/)  
  with full search and tons of new documentation  
- New settings panel with simplified and streamlined configuration  

We've also added support for several new models such as highly anticipated [NVLabs Sana](https://huggingface.co/Efficient-Large-Model/Sana_1600M_1024px) (see [supported models](https://vladmandic.github.io/sdnext-docs/Model-Support/) for full list)  
And several new SOTA video models: [Lightricks LTX-Video](https://huggingface.co/Lightricks/LTX-Video), [Hunyuan Video](https://huggingface.co/tencent/HunyuanVideo) and [Genmo Mochi.1 Preview](https://huggingface.co/genmo/mochi-1-preview)  

And a lot of **Control** and **IPAdapter** goodies  
- for **SDXL** there is new [ProMax](https://huggingface.co/xinsir/controlnet-union-sdxl-1.0), improved *Union* and *Tiling* models  
- for **FLUX.1** there are [Flux Tools](https://blackforestlabs.ai/flux-1-tools/) as well as official *Canny* and *Depth* models,  
  a cool [Redux](https://huggingface.co/black-forest-labs/FLUX.1-Redux-dev) model as well as [XLabs](https://huggingface.co/XLabs-AI/flux-ip-adapter-v2) IP-adapter
- for **SD3.5** there are official *Canny*, *Blur* and *Depth* models in addition to existing 3rd party models  
  as well as [InstantX](https://huggingface.co/InstantX/SD3.5-Large-IP-Adapter) IP-adapter  

Plus couple of new integrated workflows such as [FreeScale](https://github.com/ali-vilab/FreeScale) and [Style Aligned Image Generation](https://style-aligned-gen.github.io/)  

And it wouldn't be a *Xmass edition* without couple of custom themes: *Snowflake* and *Elf-Green*!  
All-in-all, we're around ~180 commits worth of updates, check the changelog for full list  

[ReadMe](https://github.com/vladmandic/automatic/blob/master/README.md) | [ChangeLog](https://github.com/vladmandic/automatic/blob/master/CHANGELOG.md) | [Docs](https://vladmandic.github.io/sdnext-docs/) | [WiKi](https://github.com/vladmandic/automatic/wiki) | [Discord](https://discord.com/invite/sd-next-federal-batch-inspectors-1101998836328697867)

## Details for 2024-12-24

### New models and integrations

- [NVLabs Sana](https://huggingface.co/Efficient-Large-Model/Sana_1600M_1024px)
  support for 1.6B 2048px, 1.6B 1024px and 0.6B 512px models  
  **Sana** can synthesize high-resolution images with strong text-image alignment by using **Gemma2** as text-encoder  
  and its *fast* - typically at least **2x** faster than sd-xl even for 1.6B variant and maintains performance regardless of resolution  
  e.g., rendering at 4k is possible in less than 8GB vram  
  to use, select from *networks -> models -> reference* and models will be auto-downloaded on first use  
  *reference values*: sampler: default (or any flow-match variant), steps: 20, width/height: 1024, guidance scale: 4.5  
  *note* like other LLM-based text-encoders, sana prefers long and descriptive prompts  
  any short prompt below 300 characters will be auto-expanded using built in Gemma LLM before encoding while long prompts will be passed as-is  
- **ControlNet**
  - improved support for **Union** controlnets with granular control mode type
  - added support for latest [Xinsir ProMax](https://huggingface.co/xinsir/controlnet-union-sdxl-1.0) all-in-one controlnet  
  - added support for multiple **Tiling** controlnets, for example [Xinsir Tile](https://huggingface.co/xinsir/controlnet-tile-sdxl-1.0)  
    *note*: when selecting tiles in control settings, you can also specify non-square ratios  
    in which case it will use context-aware image resize to maintain overall composition  
    *note*: available tiling options can be set in settings -> control  
- **IP-Adapter**  
  - FLUX.1 [XLabs](https://huggingface.co/XLabs-AI/flux-ip-adapter-v2) v1 and v2 IP-adapter  
  - FLUX.1 secondary guidance, enabled using *Attention guidance* in advanced menu  
  - SD 3.5 [InstantX](https://huggingface.co/InstantX/SD3.5-Large-IP-Adapter) IP-adapter  
- [Flux Tools](https://blackforestlabs.ai/flux-1-tools/)  
  **Redux** is actually a tool, **Fill** is inpaint/outpaint optimized version of *Flux-dev*  
  **Canny** & **Depth** are optimized versions of *Flux-dev* for their respective tasks: they are *not* ControlNets that work on top of a model  
  to use, go to image or control interface and select *Flux Tools* in scripts  
  all models are auto-downloaded on first use  
  *note*: All models are [gated](https://github.com/vladmandic/automatic/wiki/Gated) and require acceptance of terms and conditions via web page  
  *recommended*: Enable on-the-fly [quantization](https://github.com/vladmandic/automatic/wiki/Quantization) or [compression](https://github.com/vladmandic/automatic/wiki/NNCF-Compression) to reduce resource usage  
  *todo*: support for Canny/Depth LoRAs  
  - [Redux](https://huggingface.co/black-forest-labs/FLUX.1-Redux-dev): ~0.1GB  
    works together with existing model and basically uses input image to analyze it and use that instead of prompt  
    *optional* can use prompt to combine guidance with input image  
    *recommended*: low denoise strength levels result in more variety  
  - [Fill](https://huggingface.co/black-forest-labs/FLUX.1-Fill-dev): ~23.8GB, replaces currently loaded model  
    *note*: can be used in inpaint/outpaint mode only  
  - [Canny](https://huggingface.co/black-forest-labs/FLUX.1-Canny-dev): ~23.8GB, replaces currently loaded model  
    *recommended*: guidance scale 30  
  - [Depth](https://huggingface.co/black-forest-labs/FLUX.1-Depth-dev): ~23.8GB, replaces currently loaded model  
    *recommended*: guidance scale 10  
- [Flux ControlNet LoRA](https://huggingface.co/black-forest-labs/FLUX.1-Canny-dev-lora)  
  alternative to standard ControlNets, FLUX.1 also allows LoRA to help guide the generation process  
  both **Depth** and **Canny** LoRAs are available in standard control menus  
- [StabilityAI SD35 ControlNets](https://huggingface.co/stabilityai/stable-diffusion-3.5-controlnets)
  - In addition to previously released `InstantX` and `Alimama`, we now have *official* ones from StabilityAI  
- [Style Aligned Image Generation](https://style-aligned-gen.github.io/)  
  enable in scripts, compatible with sd-xl  
  enter multiple prompts in prompt field separated by new line  
  style-aligned applies selected attention layers uniformly to all images to achive consistency  
  can be used with or without input image in which case first prompt is used to establish baseline  
  *note:* all prompts are processes as a single batch, so vram is limiting factor  
- [FreeScale](https://github.com/ali-vilab/FreeScale)  
  enable in scripts, compatible with sd-xl for text and img2img  
  run iterative generation of images at different scales to achieve better results  
  can render 4k sdxl images  
  *note*: disable live preview to avoid memory issues when generating large images  

### Video models

- [Lightricks LTX-Video](https://huggingface.co/Lightricks/LTX-Video)  
  model size: 27.75gb  
  support for 0.9.0, 0.9.1 and custom safetensor-based models with full quantization and offloading support  
  support for text-to-video and image-to-video, to use, select in *scripts -> ltx-video*  
  *refrence values*: steps 50, width 704, height 512, frames 161, guidance scale 3.0  
- [Hunyuan Video](https://huggingface.co/tencent/HunyuanVideo)  
  model size: 40.92gb  
  support for text-to-video, to use, select in *scripts -> hunyuan video*  
  basic support only  
  *refrence values*: steps 50, width 1280, height 720, frames 129, guidance scale 6.0  
- [Genmo Mochi.1 Preview](https://huggingface.co/genmo/mochi-1-preview)  
  support for text-to-video, to use, select in *scripts -> mochi.1 video*  
  basic support only  
  *refrence values*: steps 64, width 848, height 480, frames 19, guidance scale 4.5  

*Notes*:
- all video models are very large and resource intensive!  
  any use on gpus below 16gb and systems below 48gb ram is experimental at best  
- sdnext support for video models is relatively basic with further optimizations pending community interest  
  any future optimizations would likely have to go into partial loading and excecution instead of offloading inactive parts of the model  
- new video models use generic llms for prompting and due to that requires very long and descriptive prompt  
- you may need to enable sequential offload for maximum gpu memory savings  
- optionally enable pre-quantization using bnb for additional memory savings  
- reduce number of frames and/or resolution to reduce memory usage  

### UI and workflow improvements

- **Docs**:
  - New documentation site! <https://vladmandic.github.io/sdnext-docs/>
  - Additional Wiki content: Styles, Wildcards, etc.
- **LoRA** handler rewrite:  
  - LoRA weights are no longer calculated on-the-fly during model execution, but are pre-calculated at the start  
    this results in perceived overhead on generate startup, but results in overall faster execution as LoRA does not need to be processed on each step  
    thanks @AI-Casanova  
  - LoRA weights can be applied/unapplied as on each generate or they can store weights backups for later use  
    this setting has large performance and resource implications, see [Offload](https://github.com/vladmandic/automatic/wiki/Offload) wiki for details  
  - LoRA name in prompt can now also be an absolute path to a LoRA file, even if LoRA is not indexed  
    example: `<lora:/test/folder/my-lora.safetensors:1.0>`
  - LoRA name in prompt can now also be path to a LoRA file op `huggingface`  
    example: `<lora:/huggingface.co/vendor/repo/my-lora.safetensors:1.0>`
- **Model loader** improvements:  
  - detect model components on model load fail  
  - allow passing absolute path to model loader  
  - Flux, SD35: force unload model  
  - Flux: apply `bnb` quant when loading *unet/transformer*  
  - Flux: all-in-one safetensors  
    example: <https://civitai.com/models/646328?modelVersionId=1040235>  
  - Flux: do not recast quants  
- **Memory** improvements:  
  - faster and more compatible *balanced offload* mode  
  - balanced offload: units are now in percentage instead of bytes  
  - balanced offload: add both high and low watermark, defaults as below  
    `0.25` for low-watermark: skip offload if memory usage is below 25%  
    `0.70` high-watermark: must offload if memory usage is above 70%  
  - balanced offload will attempt to run offload as non-blocking and force gc at the end  
  - change-in-behavior:  
    low-end systems, triggered by either `lowvrwam` or by detection of <=4GB will use *sequential offload*  
    all other systems use *balanced offload* by default (can be changed in settings)  
    previous behavior was to use *model offload* on systems with <=8GB and `medvram` and no offload by default  
  - VAE upcase is now disabled by default on all systems  
    if you have issues with image decode, you'll need to enable it manually  
- **UI**:  
  - improved stats on generate completion  
  - improved live preview display and performance  
  - improved accordion behavior  
  - auto-size networks height for sidebar  
  - control: hide preview column by default
  - control: optionn to hide input column
  - control: add stats
  - settings: reorganized and simplified  
  - browser -> server logging framework  
  - add addtional themes: `black-reimagined`, thanks @Artheriax  
- **Batch**
  - image batch processing will use caption files if they exist instead of default prompt  

### Updates

- **Quantization**
  - Add `TorchAO` *pre* (during load) and *post* (during execution) quantization  
    **torchao** supports 4 different int-based and 3 float-based quantization schemes  
  This is in addition to existing support for:  
  - `BitsAndBytes` with 3 float-based quantization schemes  
  - `Optimium.Quanto` with 3 int-based and 2 float-based quantizations schemes  
  - `GGUF` with pre-quantized weights  
  - Switch `GGUF` loader from custom to diffuser native
- **IPEX**: update to IPEX 2.5.10+xpu  
- **OpenVINO**:  
  - update to 2024.6.0  
  - disable model caching by default  
- **Sampler** improvements  
  - UniPC, DEIS, SA, DPM-Multistep: allow FlowMatch sigma method and prediction type  
  - Euler FlowMatch: add sigma methods (*karras/exponential/betas*)  
  - Euler FlowMatch: allow using timestep presets to set sigmas  
  - DPM FlowMatch: update all and add sigma methods  
  - BDIA-DDIM: *experimental* new scheduler  
  - UFOGen: *experimental* new scheduler  

### Fixes  

- add `SD_NO_CACHE=true` env variable to disable file/folder caching  
- add settings -> networks -> embeddings -> enable/disable
- update `diffusers`  
- fix README links  
- fix sdxl controlnet single-file loader  
- relax settings validator  
- improve js progress calls resiliency  
- fix text-to-video pipeline  
- avoid live-preview if vae-decode is running  
- allow xyz-grid with multi-axis s&r  
- fix xyz-grid with lora  
- fix api script callbacks  
- fix gpu memory monitoring  
- simplify img2img/inpaint/sketch canvas handling  
- fix prompt caching  
- fix xyz grid skip final pass  
- fix sd upscale script  
- fix cogvideox-i2v  
- lora auto-apply tags remove duplicates  
- control load model on-demand if not already loaded  
- taesd limit render to 2024px  
- taesd downscale preview to 1024px max: configurable in settings -> live preview  
- uninstall conflicting `wandb` package  
- dont skip diffusers version check if quick is specified  
- notify on torch install  
- detect pipeline fro diffusers folder-style model  
- do not recast flux quants  
- fix xyz-grid with lora none  
- fix svd image2video  
- fix gallery display during generate  
- fix wildcards replacement to be unique  
- fix animatediff-xl  
- fix pag with batch count  

## Update for 2024-11-21

### Highlights for 2024-11-21

Three weeks is a long time in Generative AI world - and we're back with ~140 commits worth of updates!

*What's New?*

First, a massive update to docs including new UI top-level **info** tab with access to [changelog](https://github.com/vladmandic/automatic/blob/master/CHANGELOG.md) and [wiki](https://github.com/vladmandic/automatic/wiki), many updates and new articles AND full **built-in documentation search** capabilities

#### New integrations

- [PuLID](https://github.com/ToTheBeginning/PuLID): Pure and Lightning ID Customization via Contrastive Alignment
- [InstantX InstantIR](https://github.com/instantX-research/InstantIR): Blind Image Restoration with Instant Generative Reference
- [nVidia Labs ConsiStory](https://github.com/NVlabs/consistory): Consistent Image Generation
- [MiaoshouAI PromptGen v2.0](https://huggingface.co/MiaoshouAI/Florence-2-base-PromptGen-v2.0) VQA captioning

#### Workflow Improvements

- Native **Docker** support
- **SD3x & Flux.1**: more ControlNets, all-in-one-safetensors, DPM samplers, skip-layer-guidance, etc.
- **XYZ grid**: benchmarking, video creation, etc.
- Enhanced **prompt** parsing
- **UI** improvements
- **Installer** self-healing `venv`

And quite a few more improvements and fixes since the last update!
For full list and details see changelog...

[README](https://github.com/vladmandic/automatic/blob/master/README.md) | [CHANGELOG](https://github.com/vladmandic/automatic/blob/master/CHANGELOG.md) | [WiKi](https://github.com/vladmandic/automatic/wiki) | [Discord](https://discord.com/invite/sd-next-federal-batch-inspectors-1101998836328697867)

### Details for 2024-11-21

- Docs:  
  - new top-level **info** tab with access to [changelog](https://github.com/vladmandic/automatic/blob/master/CHANGELOG.md) and [wiki](https://github.com/vladmandic/automatic/wiki)  
  - UI built-in [changelog](https://github.com/vladmandic/automatic/blob/master/CHANGELOG.md) search  
    since changelog is the best up-to-date source of info  
    go to info -> changelog and search/highligh/navigate directly in UI!  
  - UI built-in [wiki](https://github.com/vladmandic/automatic/wiki)  
    go to info -> wiki and search wiki pages directly in UI!  
  - major [Wiki](https://github.com/vladmandic/automatic/wiki) and [Home](https://github.com/vladmandic/automatic) updates  
  - updated API swagger docs for at `/docs`  
- Integrations:  
  - [PuLID](https://github.com/ToTheBeginning/PuLID): Pure and Lightning ID Customization via Contrastive Alignment  
    - advanced method of face id transfer with better quality as well as control over identity and appearance  
      try it out, likely the best quality available for sdxl models  
    - select in *scripts -> pulid*  
    - compatible with *sdxl* for text-to-image, image-to-image, inpaint, refine, detailer workflows  
    - can be used in xyz grid  
    - *note*: this module contains several advanced features on top of original implementation  
  - [InstantIR](https://github.com/instantX-research/InstantIR): Blind Image Restoration with Instant Generative Reference  
    - alternative to traditional `img2img` with more control over restoration process  
    - select in *image -> scripts -> instantir*  
    - compatible with *sdxl*  
    - *note*: after used once it cannot be unloaded without reloading base model  
  - [ConsiStory](https://github.com/NVlabs/consistory): Consistent Image Generation  
    - create consistent anchor image and then generate images that are consistent with anchor  
    - select in *scripts -> consistory*  
    - compatible with *sdxl*  
    - *note*: very resource intensive and not compatible with model offloading  
    - *note*: changing default parameters can lead to unexpected results and/or failures  
    - *note*: after used once it cannot be unloaded without reloading base model  
  - [MiaoshouAI PromptGen v2.0](https://huggingface.co/MiaoshouAI/Florence-2-base-PromptGen-v2.0) base and large:  
    - *in process -> visual query*  
    - caption modes:  
      `<GENERATE_TAGS>` generate tags  
      `<CAPTION>`, `<DETAILED_CAPTION>`, `<MORE_DETAILED_CAPTION>` caption image  
      `<ANALYZE>` image composition  
      `<MIXED_CAPTION>`, `<MIXED_CAPTION_PLUS>` detailed caption and tags with optional analyze  

- Model improvements:  
  - SD35: **ControlNets**:  
    - *InstantX Canny, Pose, Depth, Tile*  
    - *Alimama Inpainting, SoftEdge*  
    - *note*: that just like with FLUX.1 or any large model, ControlNet are also large and can push your system over the limit  
      e.g. SD3 controlnets vary from 1GB to over 4GB in size  
  - SD35: **All-in-one** safetensors  
    - *examples*: [large](https://civitai.com/models/882666/sd35-large-google-flan?modelVersionId=1003031), [medium](https://civitai.com/models/900327)  
    - *note*: enable *bnb* on-the-fly quantization for even bigger gains  
  - SD35: **skip-layer-guidance**  
    - enable in *scripts -> slg*
    - allows for granular strength/start/stop control of guidance for each layer of the model  
  - [NoobAI XL ControlNets](https://huggingface.co/collections/Eugeoter/controlnext-673161eae023f413e0432799), thanks @lbeltrame

- Workflow improvements:  
  - Native Docker support with pre-defined [Dockerfile](https://github.com/vladmandic/automatic/blob/dev/Dockerfile)
  - Samplers:
    - **FlowMatch samplers**:
      - Applicable to SD 3.x and Flux.1 models
      - Complete family: *DPM2, DPM2a, DPM2++, DPM2++ 2M, DPM2++ 2S, DPM2++ SDE, DPM2++ 2M SDE, DPM2++ 3M SDE*
    - **Beta and Exponential** sigma method enabled for all samplers
  - **XYZ grid**:  
    - optional time benchmark info to individual images  
    - optional add params to individual images  
    - create video from generated grid images  
      supports all standard video types and interpolation  
  - **Prompt parser**:  
    - support for prompt scheduling  
    - renamed parser options: `native`, `xhinker`, `compel`, `a1111`, `fixed`  
    - parser options are available in xyz grid  
    - improved caching  
  - **UI**:  
    - better gallery and networks sidebar sizing  
    - add additional [hotkeys](https://github.com/vladmandic/automatic/wiki/Hotkeys)  
    - add show networks on startup setting  
    - better mapping of networks previews  
    - optimize networks display load  
  - Image2image:  
    - integrated refine/upscale/hires workflow  
- Other:  
  - **Installer**:  
    - Log `venv` and package search paths  
    - Auto-remove invalid packages from `venv/site-packages`  
      e.g. packages starting with `~` which are left-over due to windows access violation  
    - Requirements: update  
  - Scripts:  
    - More verbose descriptions for all scripts  
  - Model loader:  
    - Report modules included in safetensors when attempting to load a model  
  - CLI:  
    - refactor command line params  
      run `webui.sh`/`webui.bat` with `--help` to see all options  
    - added `cli/model-metadata.py` to display metadata in any safetensors file  
    - added `cli/model-keys.py` to quicky display content of any safetensors file  
  - Internal:  
    - Auto pipeline switching coveres wrapper classes and nested pipelines  
    - Full settings validation on load of `config.json`  
    - Refactor of all params in main processing classes  
    - Improve API scripts usage resiliency  

- Fixes:  
  - custom watermark add alphablending  
  - fix xyz grid include images  
  - fix xyz skip on interrupted  
  - fix vqa models ignoring hfcache folder setting  
  - fix network height in standard vs modern ui  
  - fix k-diff enum on startup  
  - fix text2video scripts  
  - multiple xyz-grid fixes  
  - dont uninstall flash-attn  
  - ui css fixes  

## Update for 2024-11-01

Smaller release just 3 days after the last one, but with some important fixes and improvements.  
This release can be considered an LTS release before we kick off the next round of major updates.  

- Other:
  - Repo: move screenshots to GH pages
  - Update requirements
- Fixes:
  - detailer min/max size as fractions of image size  
  - ipadapter load on-demand  
  - ipadapter face use correct yolo model  
  - list diffusers remove duplicates  
  - fix legacy extensions access to shared objects  
  - fix diffusers load from folder  
  - fix lora enum logging on windows  
  - fix xyz grid with batch count  
  - move dowwloads of some auxillary models to hfcache instead of models folder  

## Update for 2024-10-29

### Highlights for 2024-10-29

- Support for **all SD3.x variants**  
  *SD3.0-Medium, SD3.5-Medium, SD3.5-Large, SD3.0-Large-Turbo*
- Allow quantization using `bitsandbytes` on-the-fly during models load
  Load any variant of SD3.x or FLUX.1 and apply quantization during load without the need for pre-quantized models  
- Allow for custom model URL in standard model selector  
  Can be used to specify any model from *HuggingFace* or *CivitAI*  
- Full support for `torch==2.5.1`
- New wiki articles: [Gated Access](https://github.com/vladmandic/automatic/wiki/Gated), [Quantization](https://github.com/vladmandic/automatic/wiki/Quantization), [Offloading](https://github.com/vladmandic/automatic/wiki/Offload)  

Plus tons of smaller improvements and cumulative fixes reported since last release  

[README](https://github.com/vladmandic/automatic/blob/master/README.md) | [CHANGELOG](https://github.com/vladmandic/automatic/blob/master/CHANGELOG.md) | [WiKi](https://github.com/vladmandic/automatic/wiki) | [Discord](https://discord.com/invite/sd-next-federal-batch-inspectors-1101998836328697867)

### Details for 2024-10-29

- model selector:
  - change-in-behavior
  - when typing, it will auto-load model as soon as exactly one match is found
  - allows entering model that are not on the list which triggers huggingface search  
    e.g. `stabilityai/stable-diffusion-xl-base-1.0`  
    partial search hits are displayed in the log  
    if exact model is found, it will be auto-downloaded and loaded  
  - allows entering civitai direct download link which triggers model download  
    e.g. `https://civitai.com/api/download/models/72396?type=Model&format=SafeTensor&size=full&fp=fp16`  
  - auto-search-and-download can be disabled in settings -> models -> auto-download  
    this also disables reference models as they are auto-downloaded on first use as well  
- sd3 enhancements:  
  - allow on-the-fly bnb quantization during load
  - report when loading incomplete model  
  - handle missing model components during load  
  - handle component preloading  
  - native lora handler  
  - support for all sd35 variants: *medium/large/large-turbo*
  - gguf transformer loader (prototype)  
- flux.1 enhancements:  
  - allow on-the-fly bnb quantization during load
- samplers:
  - support for original k-diffusion samplers  
    select via *scripts -> k-diffusion -> sampler*  
- ipadapter:
  - list available adapters based on loaded model type
  - add adapter `ostris consistency` for sd15/sdxl
- detailer:
  - add `[prompt]` to refine/defailer prompts as placeholder referencing original prompt  
- torch
  - use `torch==2.5.1` by default on supported platforms
  - CUDA set device memory limit
    in *settings -> compute settings -> torch memory limit*  
    default=0 meaning no limit, if set torch will limit memory usage to specified fraction  
    *note*: this is not a hard limit, torch will try to stay under this value  
- compute backends:
  - OpenVINO: add accuracy option  
  - ZLUDA: guess GPU arch  
- major model load refactor
- wiki: new articles
  - [Gated Access Wiki](https://github.com/vladmandic/automatic/wiki/Gated)  
  - [Quantization Wiki](https://github.com/vladmandic/automatic/wiki/Quantization)  
  - [Offloading Wiki](https://github.com/vladmandic/automatic/wiki/Offload)  

fixes:  
- fix send-to-control  
- fix k-diffusion  
- fix sd3 img2img and hires  
- fix ipadapter supported model detection  
- fix t2iadapter auto-download
- fix omnigen dynamic attention  
- handle a1111 prompt scheduling  
- handle omnigen image placeholder in prompt  

## Update for 2024-10-23

### Highlights for 2024-10-23

A month later and with nearly 300 commits, here is the latest [SD.Next](https://github.com/vladmandic/automatic) update!  

#### Workflow highlights for 2024-10-23

- **Reprocess**: New workflow options that allow you to generate at lower quality and then  
  reprocess at higher quality for select images only or generate without hires/refine and then reprocess with hires/refine  
  and you can pick any previous latent from auto-captured history!  
- **Detailer** Fully built-in detailer workflow with support for all standard models  
- Built-in **model analyzer**  
  See all details of your currently loaded model, including components, parameter count, layer count, etc.  
- **Extract LoRA**: load any LoRA(s) and play with generate as usual  
  and once you like the results simply extract combined LoRA for future use!  

#### New models for 2024-10-23

- New fine-tuned [CLiP-ViT-L](https://huggingface.co/zer0int/CLIP-GmP-ViT-L-14) 1st stage **text-encoders** used by most models (SD15/SDXL/SD3/Flux/etc.) brings additional details to your images  
- New models:  
  [Stable Diffusion 3.5 Large](https://huggingface.co/stabilityai/stable-diffusion-3.5-large)  
  [OmniGen](https://arxiv.org/pdf/2409.11340)  
  [CogView 3 Plus](https://huggingface.co/THUDM/CogView3-Plus-3B)  
  [Meissonic](https://github.com/viiika/Meissonic)  
- Additional integration:  
  [Ctrl+X](https://github.com/genforce/ctrl-x) which allows for control of **structure and appearance** without the need for extra models,  
  [APG: Adaptive Projected Guidance](https://arxiv.org/pdf/2410.02416) for optimal **guidance** control,  
  [LinFusion](https://github.com/Huage001/LinFusion) for on-the-fly **distillation** of any sd15/sdxl model  

#### What else for 2024-10-23

- Tons of work on **dynamic quantization** that can be applied *on-the-fly* during model load to any model type (*you do not need to use pre-quantized models*)  
  Supported quantization engines include `BitsAndBytes`, `TorchAO`, `Optimum.quanto`, `NNCF` compression, and more...  
- Auto-detection of best available **device/dtype** settings for your platform and GPU reduces neeed for manual configuration  
  *Note*: This is a breaking change to default settings and its recommended to check your preferred settings after upgrade  
- Full rewrite of **sampler options**, not far more streamlined with tons of new options to tweak scheduler behavior  
- Improved **LoRA** detection and handling for all supported models  
- Several of [Flux.1](https://huggingface.co/black-forest-labs/FLUX.1-dev) optimizations and new quantization types  

Oh, and we've compiled a full table with list of top-30 (*how many have you tried?*) popular text-to-image generative models,  
their respective parameters and architecture overview: [Models Overview](https://github.com/vladmandic/automatic/wiki/Models)  

And there are also other goodies like multiple *XYZ grid* improvements, additional *Flux ControlNets*, additional *Interrogate models*, better *LoRA tags* support, and more...  
[README](https://github.com/vladmandic/automatic/blob/master/README.md) | [CHANGELOG](https://github.com/vladmandic/automatic/blob/master/CHANGELOG.md) | [WiKi](https://github.com/vladmandic/automatic/wiki) | [Discord](https://discord.com/invite/sd-next-federal-batch-inspectors-1101998836328697867)

### Details for 2024-10-23

- **reprocess**
  - new top-level button: reprocess latent from your history of generated image(s)  
  - generate using full-quality:off and then reprocess using *full quality decode*  
  - generate without hires/refine and then *reprocess with hires/refine*  
    *note*: you can change hires/refine settings and run-reprocess again!  
  - reprocess using *detailer*  

- **history**
  - by default, **reprocess** will pick last latent, but you can select any latent from history!  
  - history is under *networks -> history*  
    each history item includes info on operations that were used, timestamp and metadata  
  - any latent operation during workflow automatically adds one or more items to history  
    e.g. generate base + upscale + hires + detailer  
  - history size: *settings -> execution -> latent history size*  
    memory usage is ~130kb of ram for 1mp image  
  - *note* list of latents in history is not auto-refreshed, use refresh button  

- **model analyzer**  
  - see all details of your currently loaded model, including components, parameter count, layer count, etc.  
  - in models -> current -> analyze  

- **text encoder**:  
  - allow loading different custom text encoders: *clip-vit-l, clip-vit-g, t5*  
    will automatically find appropriate encoder in the loaded model and replace it with loaded text encoder  
    download text encoders into folder set in settings -> system paths -> text encoders  
    default `models/Text-encoder` folder is used if no custom path is set  
    finetuned *clip-vit-l* models: [Detailed, Smooth](https://huggingface.co/zer0int/CLIP-GmP-ViT-L-14), [LongCLIP](https://huggingface.co/zer0int/LongCLIP-GmP-ViT-L-14)  
    reference *clip-vit-l* and *clip-vit-g* models: [OpenCLIP-Laion2b](https://huggingface.co/collections/laion/openclip-laion-2b-64fcade42d20ced4e9389b30)  
    *note* sd/sdxl contain heavily distilled versions of reference models, so switching to reference model produces vastly different results  
  - xyz grid support for text encoder  
  - full prompt parser now correctly works with different prompts in batch  

- **detailer**:  
  - replaced *face-hires* with *detailer* which can run any number of standard detailing models  
  - includes *face/hand/person/eyes* predefined detailer models plus support for manually downloaded models  
    set path in *settings -> system paths -> yolo*  
  - select one or more models in detailer menu and thats it!  
  - to avoid duplication of ui elements, detailer will use following values from **refiner**:  
    *sampler, steps, prompts*  
  - when using multiple detailers and prompt is *multi-line*, each line is applied to corresponding detailer  
  - adjustable settings:  
    *strength, max detected objects, edge padding, edge blur, min detection confidence, max detection overlap, min and max size of detected object*  
  - image metadata includes info on used detailer models  
  - *note* detailer defaults are not save in ui settings, they are saved in server settings  
    to apply your defaults, set ui values and apply via *system -> settings -> apply settings*  
  - if using models trained on multiple classes, you can specify which classes you want to detail  
    e.g. original yolo detection model is trained on coco dataset with 80 predefined classes  
    if you leave field blank, it will use any class found in the model  
    you can see classes defined in the model while model itself is loaded for the first time  

- **extract lora**: extract combined lora from current memory state, thanks @AI-Casanova  
  load any LoRA(s) and play with generate as usual and once you like the results simply extract combined LoRA for future use!  
  in *models -> extract lora*  

- **sampler options**: full rewrite  

  *sampler notes*:  
  - pick a sampler and then pick values, all values have "default" as a choice to make it simpler  
  - a lot of options are new, some are old but moved around  
    e.g. karras checkbox is replaced with a choice of different sigma methods  
  - not every combination of settings is valid  
  - some settings are specific to model types  
    e.g. sd15/sdxl typically use epsilon prediction  
  - quite a few well-known schedulers are just variations of settings, for example:  
    - *sampler sgm* is sampler with trailing spacing and sample prediction type  
    - *dpm 2m* or *3m* are *dpm 1s* with orders of 2 or 3  
    - *dpm 2m sde* is *dpm 2m* with *sde* as solver  
    - *sampler simple* is sampler with trailing spacing and linear beta schedule
  - xyz grid support for sampler options  
  - metadata updates for sampler options  
  - modernui updates for sampler options  
  - *note* sampler options defaults are not saved in ui settings, they are saved in server settings  
    to apply your defaults, set ui values and apply via *system -> settings -> apply settings*  

  *sampler options*:  
  - sigma method: *karas, beta, exponential*  
  - timesteps spacing: *linspace, leading, trailing*  
  - beta schedule: *linear, scaled, cosine*  
  - prediction type: *epsilon, sample, v-prediction*  
  - timesteps presents: *none, ays-sd15, ays-sdxl*  
  - timesteps override: <custom>  
  - sampler order: *0=default, 1-5*  
  - options: *dynamic, low order, rescale*  

- [Ctrl+X](https://github.com/genforce/ctrl-x):
  - control **structure** (*similar to controlnet*) and **appearance** (*similar to ipadapter*)  
    without the need for extra models, all via code feed-forwards!
  - can run in structure-only or appearance-only or both modes
  - when providing structure and appearance input images, its best to provide a short prompts describing them  
  - structure image can be *almost anything*: *actual photo, openpose-style stick man, 3d render, sketch, depth-map, etc.*  
    just describe what it is in a structure prompt so it can be de-structured and correctly applied  
  - supports sdxl in both txt2img and img2img, simply select from scripts

- [APG: Adaptive Projected Guidance](https://arxiv.org/pdf/2410.02416)
  - latest algo to provide better guidance for image generation, can be used instead of existing guidance rescale and/or PAG  
  - in addtion to stronger guidance and reduction of burn at high guidance values, it can also increase image details  
  - compatible with *sd15/sdxl/sc*  
  - select in scripts -> apg  
  - for low    cfg scale, use positive momentum: e.g. cfg=2 => momentum=0.6
  - for normal cfg scale, use negative momentum: e.g. cfg=6 => momentum=-0.3
  - for high   cfg scale, use neutral  momentum: e.g. cfg=10 => momentum=0.0

- [LinFusion](https://github.com/Huage001/LinFusion)  
  - apply liner distillation to during load to any sd15/sdxl model  
  - can reduce vram use for high resolutions and increase performance
  - *note*: use lower cfg scales as typical for distilled models  

- [Flux](https://huggingface.co/black-forest-labs/FLUX.1-dev)  
  - see [wiki](https://github.com/vladmandic/automatic/wiki/FLUX#quantization) for details on `gguf`  
  - support for `gguf` binary format for loading unet/transformer component  
  - support for `gguf` binary format for loading t5/text-encoder component: requires transformers pr  
  - additional controlnets: [JasperAI](https://huggingface.co/collections/jasperai/flux1-dev-controlnets-66f27f9459d760dcafa32e08) **Depth**, **Upscaler**, **Surface**, thanks @EnragedAntelope  
  - additional controlnets: [XLabs-AI](https://huggingface.co/XLabs-AI/flux-controlnet-hed-diffusers) **Canny**, **Depth**, **HED**  
  - mark specific unet as unavailable if load failed  
  - fix diffusers local model name parsing  
  - full prompt parser will auto-select `xhinker` for flux models  
  - controlnet support for img2img and inpaint (in addition to previous txt2img controlnet)  
  - allow separate vae load  
  - support for both kohya and onetrainer loras in native load mode for fp16/nf4/fp4, thanks @AI-Casanova  
  - support for differential diffusion  
  - added native load mode for qint8/qint4 models
  - avoid unet load if unchanged  

- [OmniGen](https://arxiv.org/pdf/2409.11340)  
  - Radical new model with pure LLM architecture based on Phi-3  
  - Select from *networks -> models -> reference*  
  - Can be used for text-to-image and image-to-image  
  - Image-to-image is *very* different, you need to specify in prompt what do you want to do  
    and add `|image|` placeholder where input image is used!  
    examples: `in |image| remove glasses from face`, `using depth map from |image|, create new image of a cute robot`  
  - Params used: prompt, steps, guidance scale for prompt guidance, refine guidance scale for image guidance  
    Recommended: guidance=3.0, refine-guidance=1.6  

- [Stable Diffusion 3.5 Large](https://huggingface.co/stabilityai/stable-diffusion-3.5-large)  
  - New/improved variant of Stable Diffusion 3  
  - Select from *networks -> models -> reference*  
  - Available in standard and turbo variations  
  - *Note*: Access to to both variations of SD3.5 model is gated, you must accept the conditions and use HF login  

- [CogView 3 Plus](https://huggingface.co/THUDM/CogView3-Plus-3B)
  - Select from *networks -> models -> reference*  
  - resolution width and height can be from 512px to 2048px and must be divisible by 32  
  - precision: bf16 or fp32  
    fp16 is not supported due to internal model overflows  

- [Meissonic](https://github.com/viiika/Meissonic)  
  - Select from *networks -> models -> reference*  
  - Experimental as upstream implemenation code is unstable
  - Must set scheduler:default, generator:unset

- [SageAttention](https://github.com/thu-ml/SageAttention)  
  - new 8-bit attention implementation on top of SDP that can provide acceleration for some models, thanks @Disty0  
  - enable in *settings -> compute settings -> sdp options -> sage attention*
  - compatible with DiT-based models: e.g. *Flux.1, AuraFlow, CogVideoX*  
  - not compatible with UNet-based models, e.g. *SD15, SDXL*  

- **gpu**
  - previously `cuda_dtype` in settings defaulted to `fp16` if available  
  - now `cuda_type` defaults to **Auto** which executes `bf16` and `fp16` tests on startup and selects best available dtype  
    if you have specific requirements, you can still set to fp32/fp16/bf16 as desired  
    if you have gpu that incorrectly identifies bf16 or fp16 availablity, let us know so we can improve the auto-detection  
  - support for torch **expandable segments**  
    enable in *settings -> compute -> torch expandable segments*  
    can provide significant memory savings for some models  
    not enabled by default as its only supported on latest versions of torch and some gpus  

- **xyz grid** full refactor  
  - multi-mode: *selectable-script* and *alwayson-script*  
  - allow usage combined with other scripts  
  - allow **unet** selection  
  - allow passing **model args** directly:  
    allowed params will be checked against models call signature  
    example: `width=768; height=512, width=512; height=768`  
  - allow passing **processing args** directly:  
    params are set directly on main processing object and can be known or new params  
    example: `steps=10, steps=20; test=unknown`  
  - enable working with different resolutions  
    now you can adjust width/height in the grid just as any other param  
  - renamed options to include section name and adjusted cost of each option  
  - added additional metadata  

- **interrogate**  
  - add additional blip models: *blip-base, blip-large, blip-t5-xl, blip-t5-xxl, opt-2.7b, opt-6.7b*  
  - change default params for better memory utilization  
  - lock commits for miaoshouAI-promptgen  
  - add optional advanced params  
  - update logging  

- **lora** auto-apply tags to prompt  
  - controlled via *settings -> networks -> lora_apply_tags*  
    *0:disable, -1:all-tags, n:top-n-tags*  
  - uses tags from both model embedded data and civitai downloaded data  
  - if lora contains no tags, lora name itself will be used as a tag  
  - if prompt contains `_tags_` it will be used as placeholder for replacement, otherwise tags will be appended  
  - used tags are also logged and registered in image metadata  
  - loras are no longer filtered per detected type vs loaded model type as its unreliable  
  - loras display in networks now shows possible version in top-left corner  
  - correct using of `extra_networks_default_multiplier` if not scale is specified  
  - improve lora base model detection  
  - improve lora error handling and logging  
  - setting `lora_load_gpu` to load LoRA directly to GPU  
    *default*: true unless lovwram  

- **quantization**  
  - new top level settings group as we have quite a few quantization options now!  
    configure in *settings -> quantization*  
  - in addition to existing `optimum.quanto` and `nncf`, we now have `bitsandbytes` and `torchao`  
  - **bitsandbytes**: fp8, fp4, nf4  
    - quantization can be applied on-the-fly during model load  
    - currently supports `transformers` and `t5` in **sd3** and **flux**  
  - **torchao**: int8, int4, fp8, fp4, fpx  
    - configure in settings -> quantization  
    - can be applied to any model on-the-fly during load  

- **huggingface**:  
  - force logout/login on token change  
  - unified handling of cache folder: set via `HF_HUB` or `HF_HUB_CACHE` or via settings -> system paths  

- **cogvideox**:  
  - add support for *image2video* (in addition to previous *text2video* and *video2video*)  
  - *note*: *image2video* requires separate 5b model variant  

- **torch**  
  - due to numerous issues with torch 2.5.0 which was just released as stable, we are sticking with 2.4.1 for now  

- **backend=original** is now marked as in maintenance-only mode  
- **python 3.12** improved compatibility, automatically handle `setuptools`  
- **control**
  - persist/reapply units current state on server restart  
  - better handle size before/after metadata  
- **video** add option `gradio_skip_video` to avoid gradio issues with displaying generated videos  
- add support for manually downloaded diffusers models from huggingface  
- **ui**  
  - move checkboxes `full quality, tiling, hidiffusion` to advanced section  
  - hide token counter until tokens are known  
  - minor ui optimizations  
  - fix update infotext on image select  
  - fix imageviewer exif parser  
  - selectable info view in image viewer, thanks @ZeldaMaster501  
  - setting to enable browser autolaunch, thanks @brknsoul  
- **free-u** check if device/dtype are fft compatible and cast as necessary  
- **rocm**
  - additional gpu detection and auto-config code, thanks @lshqqytiger  
  - experimental triton backend for flash attention, thanks @lshqqytiger  
  - update to rocm 6.2, thanks @Disty0
- **directml**  
  - update `torch` to 2.4.1, thanks @lshqqytiger  
- **extensions**  
  - add mechanism to lock-down extension to specific working commit  
  - added `sd-webui-controlnet` and `adetailer` last-known working commits  
- **upscaling**  
  - interruptible operations
- **refactor**  
  - general lora apply/unapply process  
  - modularize main process loop  
  - massive log cleanup  
  - full lint pass  
  - improve inference mode handling  
  - unify quant lib loading  


## Update for 2024-09-13

### Highlights for 2024-09-13

Major refactor of [FLUX.1](https://blackforestlabs.ai/announcing-black-forest-labs/) support:  
- Full **ControlNet** support, better **LoRA** support, full **prompt attention** implementation  
- Faster execution, more flexible loading, additional quantization options, and more...  
- Added **image-to-image**, **inpaint**, **outpaint**, **hires** modes  
- Added workflow where FLUX can be used as **refiner** for other models  
- Since both *Optimum-Quanto* and *BitsAndBytes* libraries are limited in their platform support matrix,  
  try enabling **NNCF** for quantization/compression on-the-fly!  

Few image related goodies...  
- **Context-aware** resize that allows for *img2img/inpaint* even at massively different aspect ratios without distortions!
- **LUT Color grading** apply professional color grading to your images using industry-standard *.cube* LUTs!
- Auto **HDR** image create for SD and SDXL with both 16ch true-HDR and 8-ch HDR-effect images ;)  

And few video related goodies...  
- [CogVideoX](https://huggingface.co/THUDM/CogVideoX-5b) **2b** and **5b** variants  
  with support for *text-to-video* and *video-to-video*!  
- [AnimateDiff](https://github.com/guoyww/animatediff/) **prompt travel** and **long context windows**!  
  create video which travels between different prompts and at long video lengths!  

Plus tons of other items and fixes - see [changelog](https://github.com/vladmandic/automatic/blob/master/CHANGELOG.md) for details!  
Examples:
- Built-in prompt-enhancer, TAESD optimizations, new DC-Solver scheduler, global XYZ grid management, etc.  
- Updates to ZLUDA, IPEX, OpenVINO...

### Details for 2024-09-13

**Major refactor of FLUX.1 support:**
- allow configuration of individual FLUX.1 model components: *transformer, text-encoder, vae*  
  model load will load selected components first and then initialize model using pre-loaded components  
  components that were not pre-loaded will be downloaded and initialized as needed  
  as usual, components can also be loaded after initial model load  
  *note*: use of transformer/unet is recommended as those are flux.1 finetunes  
  *note*: manually selecting vae and text-encoder is not recommended  
  *note*: mix-and-match of different quantizations for different components can lead to unexpected errors  
  - transformer/unet is list of manually downloaded safetensors  
  - vae is list of manually downloaded safetensors  
  - text-encoder is list of predefined and manually downloaded text-encoders  
- **controlnet** support:
  support for **InstantX/Shakker-Labs** models including [Union-Pro](https://huggingface.co/InstantX/FLUX.1-dev-Controlnet-Union)
  note that flux controlnet models are large, up to 6.6GB on top of already large base model!  
  as such, you may need to use offloading:sequential which is not as fast, but uses far less memory  
  when using union model, you must also select control mode in the control unit  
  flux does not yet support *img2img* so to use controlnet, you need to set contronet input via control unit override  
- model support loading **all-in-one** safetensors  
  not recommended due to massive duplication of components, but added due to popular demand  
  each such model is 20-32GB in size vs ~11GB for typical unet fine-tune  
- improve logging, warn when attempting to load unet as base model  
- **refiner** support  
  FLUX.1 can be used as refiner for other models such as sd/sdxl  
  simply load sd/sdxl model as base and flux model as refiner and use as usual refiner workflow  
- **img2img**, **inpaint** and **outpaint** support  
  *note* flux may require higher denoising strength than typical sd/sdxl models  
  *note*: img2img is not yet supported with controlnet  
- transformer/unet support *fp8/fp4* quantization  
  this brings supported quants to: *nf4/fp8/fp4/qint8/qint4*
- vae support *fp16*  
- **lora** support additional training tools  
- **face-hires** support  
- support **fuse-qkv** projections  
  can speed up generate  
  enable via *settings -> compute -> fused projections*  

**Other improvements & Fixes:**
- [CogVideoX](https://huggingface.co/THUDM/CogVideoX-5b)  
  - support for both **2B** and **5B** variations  
  - support for both **text2video** and **video2video** modes
  - simply select in *scripts -> cogvideox*  
  - as with any video modules, includes additional frame interpolation using RIFE  
  - if init video is used, it will be automatically resized and interpolated to desired number of frames  
- **AnimateDiff**:  
  - **prompt travel**  
     create video which travels between different prompts at different steps!  
     example prompt:
      > 0: dog  
      > 5: cat  
      > 10: bird  
  - support for **v3** model (finally)  
  - support for **LCM** model  
  - support for **free-noise** rolling context window  
    allow for creation of much longer videos, automatically enabled if frames > 16  
- **Context-aware** image resize, thanks @AI-Casanova!  
  based on [seam-carving](https://github.com/li-plus/seam-carving)  
  allows for *img2img/inpaint* even at massively different aspect ratios without distortions!  
  simply select as resize method when using *img2img* or *control* tabs  
- **HDR** high-dynamic-range image create for SD and SDXL  
  create hdr images from in multiple exposures by latent-space modifications during generation  
  use via *scripts -> hdr*  
  option *save hdr images* creates images in standard 8bit/channel (hdr-effect) *and* 16bit/channel (full-hdr) PNG format  
  ui result is always 8bit/channel hdr-effect image plus grid of original images used to create hdr  
  grid image can be disabled via settings -> user interface -> show grid  
  actual full-hdr image is not displayed in ui, only optionally saved to disk  
- new scheduler: [DC Solver](https://github.com/wl-zhao/DC-Solver)  
- **color grading** apply professional color grading to your images  
  using industry-standard *.cube* LUTs!
  enable via *scripts -> color-grading*  
- **hires** workflow now allows for full resize options  
  not just limited width/height/scale  
- **xyz grid** is now availabe as both local and global script!
- **prompt enhance**: improve quality and/or verbosity of your prompts  
  simply select in *scripts -> prompt enhance*
  uses [gokaygokay/Flux-Prompt-Enhance](https://huggingface.co/gokaygokay/Flux-Prompt-Enhance) model  
- **decode**
  - auto-set upcast if first decode fails  
  - restore dtype on upcast  
- **taesd** configurable number of layers  
  can be used to speed-up taesd decoding by reducing number of ops  
  e.g. if generating 1024px image, reducing layers by 1 will result in preview being 512px  
  set via *settings -> live preview -> taesd decode layers*  
- **xhinker** prompt parser handle offloaded models  
- **control** better handle offloading  
- **upscale** will use resize-to if set to non-zero values over resize-by  
  applies to any upscale options, including refine workflow  
- **networks** add option to choose if mouse-over on network should attempt to fetch additional info  
  option:`extra_networks_fetch` enable/disable in *settings -> networks*  
- speed up some garbage collection ops  
- sampler settings add **dynamic shift**  
  used by flow-matching samplers to adjust between structure and details  
- sampler settings force base shift  
  improves quality of the flow-matching samplers  
- **t5** support manually downloaded models  
  applies to all models that use t5 transformer  
- **modern-ui** add override field  
- full **lint** updates  
- use `diffusers` from main branch, no longer tied to release  
- improve diffusers/transformers/huggingface_hub progress reporting  
- use unique identifiers for all ui components  
- **visual query** (a.ka vqa or vlm) added support for several models
  - [MiaoshouAI PromptGen 1.5 Base](https://huggingface.co/MiaoshouAI/Florence-2-base-PromptGen-v1.5)
  - [MiaoshouAI PromptGen 1.5 Large](https://huggingface.co/MiaoshouAI/Florence-2-large-PromptGen-v1.5)
  - [CogFlorence 2.2 Large](https://huggingface.co/thwri/CogFlorence-2.2-Large)
- **modernui** update  
- **zluda** update to 3.8.4, thanks @lshqqytiger!
- **ipex** update to 2.3.110+xpu on linux, thanks @Disty0!
- **openvino** update to 2024.3.0, thanks @Disty0!
- update `requirements`
- fix **AuraFlow**  
- fix handling of model configs if offline config is not available  
- fix vae decode in backend original  
- fix model path typos  
- fix guidance end handler  
- fix script sorting  
- fix vae dtype during load  
- fix all ui labels are unique

## Update for 2024-08-31

### Highlights for 2024-08-31

Summer break is over and we are back with a massive update!  

Support for all of the new models:  
- [Black Forest Labs FLUX.1](https://blackforestlabs.ai/announcing-black-forest-labs/)  
- [AuraFlow 0.3](https://huggingface.co/fal/AuraFlow)  
- [AlphaVLLM Lumina-Next-SFT](https://huggingface.co/Alpha-VLLM/Lumina-Next-SFT-diffusers)  
- [Kwai Kolors](https://huggingface.co/Kwai-Kolors/Kolors)  
- [HunyuanDiT 1.2](https://huggingface.co/Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers)  

What else? Just a bit... ;)  

New **fast-install** mode, new **Optimum Quanto** and **BitsAndBytes** based quantization modes, new **balanced offload** mode that dynamically offloads GPU<->CPU as needed, and more...  
And from previous service-pack: new **ControlNet-Union** *all-in-one* model, support for **DoRA** networks, additional **VLM** models, new **AuraSR** upscaler  

**Breaking Changes...**

Due to internal changes, you'll need to reset your **attention** and **offload** settings!  
But...For a good reason, new *balanced offload* is magic when it comes to memory utilization while sacrificing minimal performance!

### Details for 2024-08-31

**New Models...**

To use and of the new models, simply select model from *Networks -> Reference* and it will be auto-downloaded on first use  

- [Black Forest Labs FLUX.1](https://blackforestlabs.ai/announcing-black-forest-labs/)  
  FLUX.1 models are based on a hybrid architecture of multimodal and parallel diffusion transformer blocks, scaled to 12B parameters and builing on flow matching  
  This is a very large model at ~32GB in size, its recommended to use a) offloading, b) quantization  
  For more information on variations, requirements, options, and how to donwload and use FLUX.1, see [Wiki](https://github.com/vladmandic/automatic/wiki/FLUX)  
  SD.Next supports:  
  - [FLUX.1 Dev](https://huggingface.co/black-forest-labs/FLUX.1-dev) and [FLUX.1 Schnell](https://huggingface.co/black-forest-labs/FLUX.1-schnell) original variations  
  - additional [qint8](https://huggingface.co/Disty0/FLUX.1-dev-qint8) and [qint4](https://huggingface.co/Disty0/FLUX.1-dev-qint4) quantized variations  
  - additional [nf4](https://huggingface.co/sayakpaul/flux.1-dev-nf4) quantized variation  
- [AuraFlow](https://huggingface.co/fal/AuraFlow)  
  AuraFlow v0.3 is the fully open-sourced largest flow-based text-to-image generation model  
  This is a very large model at 6.8B params and nearly 31GB in size, smaller variants are expected in the future  
  Use scheduler: Default or Euler FlowMatch or Heun FlowMatch  
- [AlphaVLLM Lumina-Next-SFT](https://huggingface.co/Alpha-VLLM/Lumina-Next-SFT-diffusers)  
  Lumina-Next-SFT is a Next-DiT model containing 2B parameters, enhanced through high-quality supervised fine-tuning (SFT)  
  This model uses T5 XXL variation of text encoder (previous version of Lumina used Gemma 2B as text encoder)  
  Use scheduler: Default or Euler FlowMatch or Heun FlowMatch  
- [Kwai Kolors](https://huggingface.co/Kwai-Kolors/Kolors)  
  Kolors is a large-scale text-to-image generation model based on latent diffusion  
  This is an SDXL style model that replaces standard CLiP-L and CLiP-G text encoders with a massive `chatglm3-6b` encoder supporting both English and Chinese prompting  
- [HunyuanDiT 1.2](https://huggingface.co/Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers)  
  Hunyuan-DiT is a powerful multi-resolution diffusion transformer (DiT) with fine-grained Chinese understanding  
- [AnimateDiff](https://github.com/guoyww/animatediff/)  
  support for additional models: **SD 1.5 v3** (Sparse), **SD Lightning** (4-step), **SDXL Beta**  

**New Features...**

- support for **Balanced Offload**, thanks @Disty0!  
  balanced offload will dynamically split and offload models from the GPU based on the max configured GPU and CPU memory size  
  model parts that dont fit in the GPU will be dynamically sliced and offloaded to the CPU  
  see *Settings -> Diffusers Settings -> Max GPU memory and Max CPU memory*  
  *note*: recommended value for max GPU memory is ~80% of your total GPU memory  
  *note*: balanced offload will force loading LoRA with Diffusers method  
  *note*: balanced offload is not compatible with Optimum Quanto  
- support for **Optimum Quanto** with 8 bit and 4 bit quantization options, thanks @Disty0 and @Trojaner!  
  to use, go to Settings -> Compute Settings and enable "Quantize Model weights with Optimum Quanto" option  
  *note*: Optimum Quanto requires PyTorch 2.4  
- new prompt attention mode: **xhinker** which brings support for prompt attention to new models such as FLUX.1 and SD3  
  to use, enable in *Settings -> Execution -> Prompt attention*
- use [PEFT](https://huggingface.co/docs/peft/main/en/index) for **LoRA** handling on all models other than SD15/SD21/SDXL  
  this improves LoRA compatibility for SC, SD3, AuraFlow, Flux, etc.  

**Changes & Fixes...**

- default resolution bumped from 512x512 to 1024x1024, time to move on ;)
- convert **Dynamic Attention SDP** into a global SDP option, thanks @Disty0!  
  *note*: requires reset of selected attention option
- update default **CUDA** version from 12.1 to 12.4
- update `requirements`
- samplers now prefers the model defaults over the diffusers defaults, thanks @Disty0!  
- improve xyz grid for lora handling and add lora strength option  
- don't enable Dynamic Attention by default on platforms that support Flash Attention, thanks @Disty0!  
- convert offload options into a single choice list, thanks @Disty0!  
  *note*: requires reset of selected offload option  
- control module allows reszing of indivudual process override images to match input image  
  for example: set size->before->method:nearest, mode:fixed or mode:fill  
- control tab includes superset of txt and img scripts
- automatically offload disabled controlnet units  
- prioritize specified backend if `--use-*` option is used, thanks @lshqqytiger
- ipadapter option to auto-crop input images to faces to improve efficiency of face-transfter ipadapters  
- update **IPEX** to 2.1.40+xpu on Linux, thanks @Disty0!  
- general **ROCm** fixes, thanks @lshqqytiger!  
- support for HIP SDK 6.1 on ZLUDA backend, thanks @lshqqytiger!
- fix full vae previews, thanks @Disty0!  
- fix default scheduler not being applied, thanks @Disty0!  
- fix Stable Cascade with custom schedulers, thanks @Disty0!  
- fix LoRA apply with force-diffusers
- fix LoRA scales with force-diffusers
- fix control API
- fix VAE load refrerencing incorrect configuration
- fix NVML gpu monitoring

## Update for 2024-07-08

This release is primary service release with cumulative fixes and several improvements, but no breaking changes.

**New features...**
- massive updates to [Wiki](https://github.com/vladmandic/automatic/wiki)  
  with over 20 new pages and articles, now includes guides for nearly all major features  
  *note*: this is work-in-progress, if you have any feedback or suggestions, please let us know!
  thanks @GenesisArtemis!  
- support for **DoRA** networks, thanks @AI-Casanova!
- support for [uv](https://pypi.org/project/uv/), extremely fast installer, thanks @Yoinky3000!  
  to use, simply add `--uv` to your command line params  
- [Xinsir ControlNet++ Union](https://huggingface.co/xinsir/controlnet-union-sdxl-1.0)  
  new SDXL *all-in-one* controlnet that can process any kind of preprocessors!
- [CogFlorence 2 Large](https://huggingface.co/thwri/CogFlorence-2-Large-Freeze) VLM model  
  to use, simply select in process -> visual query  
- [AuraSR](https://huggingface.co/fal/AuraSR) high-quality 4x GAN-style upscaling model  
  note: this is a large upscaler at 2.5GB  

**And fixes...**
- enable **Florence VLM**  for all platforms, thanks @lshqqytiger!  
- improve ROCm detection under WSL2, thanks @lshqqytiger!  
- add SD3 with FP16 T5 to list of detected models  
- fix executing extensions with zero params  
- add support for embeddings bundled in LoRA, thanks @AI-Casanova!  
- fix executing extensions with zero params  
- fix nncf for lora, thanks @Disty0!  
- fix diffusers version detection for SD3  
- fix current step for higher order samplers  
- fix control input type video  
- fix reset pipeline at the end of each iteration  
- fix faceswap when no faces detected  
- fix civitai search
- multiple ModernUI fixes

## Update for 2024-06-23

### Highlights for 2024-06-23

Following zero-day **SD3** release, a 10 days later heres a refresh with 10+ improvements  
including full prompt attention, support for compressed weights, additional text-encoder quantization modes.  

But theres more than SD3:  
- support for quantized **T5** text encoder *FP16/FP8/FP4/INT8* in all models that use T5: SD3, PixArt-Σ, etc.  
- support for **PixArt-Sigma** in small/medium/large variants  
- support for **HunyuanDiT 1.1**  
- additional **NNCF weights compression** support: SD3, PixArt, ControlNet, Lora  
- integration of **MS Florence** VLM/VQA *Base* and *Large* models  
- (finally) new release of **Torch-DirectML**  
- additional efficiencies for users with low VRAM GPUs  
- over 20 overall fixes  

### Model Improvements for 2024-06-23

- **SD3**: enable tiny-VAE (TAESD) preview and non-full quality mode  
- SD3: enable base LoRA support  
- SD3: add support for FP4 quantized T5 text encoder  
  simply select in *settings -> model -> text encoder*  
  *note* for SD3 with T5, set SD.Next to use FP16 precision, not BF16 precision  
- SD3: add support for INT8 quantized T5 text encoder, thanks @Disty0!  
- SD3: enable cpu-offloading for T5 text encoder, thanks @Disty0!  
- SD3: simplified loading of model in single-file safetensors format  
  model load can now be performed fully offline  
- SD3: full support for prompt parsing and attention, thanks @AI-Casanova!
- SD3: ability to target different prompts to each of text-encoders, thanks @AI-Casanova!  
  example: `dog TE2: cat TE3: bird`
- SD3: add support for sampler shift for Euler FlowMatch  
  see *settings -> samplers*, also available as param in xyz grid  
  higher shift means model will spend more time on structure and less on details  
- SD3: add support for selecting T5 text encoder variant in XYZ grid
- **Pixart-Σ**: Add *small* (512px) and *large* (2k) variations, in addition to existing *medium* (1k)  
- Pixart-Σ: Add support for 4/8bit quantized t5 text encoder  
  *note* by default pixart-Σ uses full fp16 t5 encoder with large memory footprint  
  simply select in *settings -> model -> text encoder* before or after model load  
- **HunyuanDiT**: support for model version 1.1  
- **MS Florence**: integration of Microsoft Florence VLM/VQA Base and Large models  
  simply select in *process -> visual query*!

### General Improvements for 2024-06-23

- support FP4 quantized T5 text encoder, in addition to existing FP8 and FP16
- support for T5 text-encoder loader in **all** models that use T5  
  *example*: load FP4 or FP8 quantized T5 text-encoder into PixArt Sigma!
- support for `torch-directml` **0.2.2**, thanks @lshqqytiger!  
  *note*: new directml is finally based on modern `torch` 2.3.1!  
- xyz grid: add support for LoRA selector
- vae load: store original vae so it can be restored when set to none
- extra networks: info display now contains link to source url if model if its known  
  works for civitai and huggingface models  
- force gc for lowvram users and improve gc logging
- improved google.colab support
- css tweaks for standardui
- css tweaks for modernui
- additional torch gc checks, thanks @Disty0!

**Improvements: NNCF**, thanks @Disty0!  
- SD3 and PixArt support  
- moved the first compression step to CPU  
- sequential cpu offload (lowvram) support  
- Lora support without reloading the model  
- ControlNet compression support  

### Fixes for 2024-06-23

- fix unsaturated outputs, force apply vae config on model load  
- fix hidiffusion handling of non-square aspect ratios, thanks @ShenZhang-Shin!
- fix control second pass resize  
- fix hunyuandit set attention processor
- fix civitai download without name
- fix compatibility with latest adetailer
- fix invalid sampler warning
- fix starting from non git repo
- fix control api negative prompt handling
- fix saving style without name provided
- fix t2i-color adapter
- fix sdxl "has been incorrectly initialized"
- fix api face-hires
- fix api ip-adapter
- fix memory exceptions with ROCm, thanks @Disty0!
- fix face-hires with lowvram, thanks @Disty0!
- fix pag incorrectly resetting pipeline
- cleanup image metadata
- restructure api examples: `cli/api-*`
- handle theme fallback when invalid theme is specified
- remove obsolete training code leftovers

## Update for 2024-06-13

### Highlights for 2024-06-13

First, yes, it is here and supported: [**StabilityAI Stable Diffusion 3 Medium**](https://stability.ai/news/stable-diffusion-3-medium)  
for details on how to download and use, see [Wiki](https://github.com/vladmandic/automatic/wiki/SD3)

#### What else 2024-06-13?

A lot of work on state-of-the-art multi-lingual models with both [Tenecent HunyuanDiT](https://github.com/Tencent/HunyuanDiT) and [MuLan](https://github.com/mulanai/MuLan)  
Plus tons of minor features such as optimized initial install experience, **T-Gate** and **ResAdapter**, additional ModernUI themes (both light and dark) and fixes since the last release which was only 2 weeks ago!

### Full Changelog for 2024-06-13

#### New Models for 2024-06-23

- [StabilityAI Stable Diffusion 3 Medium](https://stability.ai/news/stable-diffusion-3-medium)  
  yup, supported!  
  quote: *"Stable Diffusion 3 Medium is a multimodal diffusion transformer (MMDiT) model that features improved performance in image quality, typography, complex prompt understanding, and resource-efficiency"*  
  sdnext also supports switching optional T5 text encoder on-the-fly as well as loading model from either diffusers repo or safetensors single-file  
  for details, see [Wiki](https://github.com/vladmandic/automatic/wiki/SD3)
- [Tenecent HunyuanDiT](https://github.com/Tencent/HunyuanDiT) bilingual english/chinese diffusion transformer model  
  note: this is a very large model at ~17GB, but can be used with less VRAM using model offloading  
  simply select from networks -> models -> reference, model will be auto-downloaded on first use  

#### New Functionality for 2024-06-23

- [MuLan](https://github.com/mulanai/MuLan) Multi-language prompts
  write your prompts in ~110 auto-detected languages!  
  compatible with *SD15* and *SDXL*  
  enable in scripts -> MuLan and set encoder to `InternVL-14B-224px` encoder  
  *note*: right now this is more of a proof-of-concept before smaller and/or quantized models are released  
  model will be auto-downloaded on first use: note its huge size of 27GB  
  even executing it in FP16 will require ~16GB of VRAM for text encoder alone  
  examples:  
  - English: photo of a beautiful woman wearing a white bikini on a beach with a city skyline in the background
  - Croatian: fotografija lijepe žene u bijelom bikiniju na plaži s gradskim obzorom u pozadini
  - Italian: Foto di una bella donna che indossa un bikini bianco su una spiaggia con lo skyline di una città sullo sfondo
  - Spanish: Foto de una hermosa mujer con un bikini blanco en una playa con un horizonte de la ciudad en el fondo
  - German: Foto einer schönen Frau in einem weißen Bikini an einem Strand mit einer Skyline der Stadt im Hintergrund
  - Arabic: صورة لامرأة جميلة ترتدي بيكيني أبيض على شاطئ مع أفق المدينة في الخلفية
  - Japanese: 街のスカイラインを背景にビーチで白いビキニを着た美しい女性の写真
  - Chinese: 一个美丽的女人在海滩上穿着白色比基尼的照片, 背景是城市天际线
  - Korean: 도시의 스카이라인을 배경으로 해변에서 흰색 비키니를 입은 아름 다운 여성의 사진
- [T-Gate](https://github.com/HaozheLiu-ST/T-GATE) Speed up generations by gating at which step cross-attention is no longer needed  
  enable via scripts -> t-gate  
  compatible with *SD15*  
- **PCM LoRAs** allow for fast denoising using less steps with standard *SD15* and *SDXL* models  
  download from <https://huggingface.co/Kijai/converted_pcm_loras_fp16/tree/main>
- [ByteDance ResAdapter](https://github.com/bytedance/res-adapter) resolution-free model adapter  
  allows to use resolutions from 0.5 to 2.0 of original model resolution, compatible with *SD15* and *SDXL*
  enable via scripts -> resadapter and select desired model
- **Kohya HiRes Fix** allows for higher resolution generation using standard *SD15* models  
  enable via scripts -> kohya-hires-fix  
  *note*: alternative to regular hidiffusion method, but with different approach to scaling  
- additional built-in 4 great custom trained **ControlNet SDXL** models from Xinsir: OpenPose, Canny, Scribble, AnimePainter  
  thanks @lbeltrame
- add torch **full deterministic mode**
  enable in settings -> compute -> use deterministic mode  
  typical differences are not large and its disabled by default as it does have some performance impact  
- new sampler: **Euler FlowMatch**  

#### Improvements Fixes 2024-06-13

- additional modernui themes
- reintroduce prompt attention normalization, disabled by default, enable in settings -> execution  
  this can drastically help with unbalanced prompts  
- further work on improving python 3.12 functionality and remove experimental flag  
  note: recommended version remains python 3.11 for all users, except if you are using directml in which case its python 3.10  
- improved **installer** for initial installs  
  initial install will do single-pass install of all required packages with correct versions  
  subsequent runs will check package versions as necessary  
- add env variable `SD_PIP_DEBUG` to write `pip.log` for all pip operations  
  also improved installer logging  
- add python version check for `torch-directml`  
- do not install `tensorflow` by default  
- improve metadata/infotext parser  
  add `cli/image-exif.py` that can be used to view/extract metadata from images  
- lower overhead on generate calls  
- auto-synchronize modernui and core branches  
- add option to pad prompt with zeros, thanks @Disty

#### Fixes 2024-06-13

- cumulative fixes since the last release  
- fix apply/unapply hidiffusion for sd15  
- fix controlnet reference enabled check  
- fix face-hires with control batch count  
- install pynvml on-demand  
- apply rollback-vae option to latest torch versions, thanks @Iaotle  
- face hires skip if strength is 0  
- restore all sampler configuration on sampler change  

## Update for 2024-06-02

- fix textual inversion loading
- fix gallery mtime display
- fix extra network scrollable area when using modernui
- fix control prompts list handling
- fix restore variation seed and strength
- fix negative prompt parsing from metadata
- fix stable cascade progress monitoring
- fix variation seed with hires pass
- fix loading models trained with onetrainer
- add variation seed info to metadata
- workaround for scale-by when using modernui
- lock torch-directml version
- improve xformers installer
- improve ultralytics installer (face-hires)
- improve triton installer (compile)
- improve insightface installer (faceip)
- improve mim installer (dwpose)
- add dpm++ 1s and dpm++ 3m aliases for dpm++ 2m scheduler with different orders

## Update for 2024-05-28

### Highlights for 2024-05-28

New [SD.Next](https://github.com/vladmandic/automatic) release has been baking in `dev` for a longer than usual, but changes are massive - about 350 commits for core and 300 for UI...

Starting with the new UI - yup, this version ships with a *preview* of the new [ModernUI](https://github.com/BinaryQuantumSoul/sdnext-modernui)  
For details on how to enable and use it, see [Home](https://github.com/BinaryQuantumSoul/sdnext-modernui) and [WiKi](https://github.com/vladmandic/automatic/wiki/Themes)  
**ModernUI** is still in early development and not all features are available yet, please report [issues and feedback](https://github.com/BinaryQuantumSoul/sdnext-modernui/issues)  
Thanks to @BinaryQuantumSoul for his hard work on this project!  

*What else?*

#### New built-in features

- [PWA](https://developer.mozilla.org/en-US/docs/Web/Progressive_web_apps) SD.Next is now installable as a web-app
- **Gallery**: extremely fast built-in gallery viewer  
  List, preview, search through all your images and videos!  
- **HiDiffusion** allows generating very-high resolution images out-of-the-box using standard models  
- **Perturbed-Attention Guidance** (PAG) enhances sample quality in addition to standard CFG scale  
- **LayerDiffuse** simply create transparent (foreground-only) images  
- **IP adapter masking** allows to use multiple input images for each segment of the input image  
- IP adapter **InstantStyle** implementation  
- **Token Downsampling** (ToDo) provides significant speedups with minimal-to-none quality loss  
- **Samplers optimizations** that allow normal samplers to complete work in 1/3 of the steps!  
  Yup, even popular DPM++2M can now run in 10 steps with quality equaling 30 steps using **AYS** presets  
- Native **wildcards** support  
- Improved built-in **Face HiRes**  
- Better **outpainting**  
- And much more...  
  For details of above features and full list, see [Changelog](https://github.com/vladmandic/automatic/blob/dev/CHANGELOG.md)

#### New models

While still waiting for *Stable Diffusion 3.0*, there have been some significant models released in the meantime:

- [PixArt-Σ](https://pixart-alpha.github.io/PixArt-sigma-project/), high end diffusion transformer model (*DiT*) capable of directly generating images at 4K resolution  
- [SDXS](https://github.com/IDKiro/sdxs), extremely fast 1-step generation consistency model  
- [Hyper-SD](https://huggingface.co/ByteDance/Hyper-SD), 1-step, 2-step, 4-step and 8-step optimized models  

*Note*  
[SD.Next](https://github.com/vladmandic/automatic) is no longer marked as a fork of [A1111](https://github.com/AUTOMATIC1111/stable-diffusion-webui/) and github project has been fully detached  
Given huge number of changes with *+3443/-3342* commits diff (at the time of fork detach) over the past year,  
a completely different backend/engine and a change of focus, it is time to give credit to original [author](https://github.com/auTOMATIC1111),  and move on!  

### Full ChangeLog for 2024-05-28

- **Features**:
  - **ModernUI** preview of the new [ModernUI](https://github.com/BinaryQuantumSoul/sdnext-modernui)  
  - [PWA](https://developer.mozilla.org/en-US/docs/Web/Progressive_web_apps) SD.Next is now installable as a web-app and includes verified manifest  
  - **Gallery
  - **Gallery**: list, preview, search through all your images and videos!  
    Implemented as infinite-scroll with client-side-caching and lazy-loading while being fully async and non-blocking  
    Search or sort by path, name, size, width, height, mtime or any image metadata item, also with extended syntax like *width > 1000*  
    *Settings*: optional additional user-defined folders, thumbnails in fixed or variable aspect-ratio  
  - [HiDiffusion](https://github.com/megvii-research/HiDiffusion):  
    Generate high-resolution images using your standard models without duplicates/distorsions AND improved performance  
    For example, *SD15* can now go up to *2024x2048* and *SDXL* up to *4k* natively
    Simply enable checkbox in advanced menu and set desired resolution  
    Additional settings are available in *settings -> inference settings -> hidiffusion*  
    And can also be set and used via *xyz grid*  
    *Note*: HiDiffusion resolution sensitive, so if you get error, set resolution to be multiples of 128  
  - [Perturbed-Attention Guidance](https://github.com/KU-CVLAB/Perturbed-Attention-Guidance)  
    PAG enhances sample quality by utilizing self-attention in formation of latent in addition to standard CFG scale  
    Simply set *advanced -> attention guidance* and *advanced -> adaptive scaling*  
    Additional options are available in *settings -> inference settings -> pag*  
    *Note*: PAG has replaced SAG as attention guidance method in SD.Next  
  - [LayerDiffuse](https://github.com/rootonchair/diffuser_layerdiffuse)
    Create transparent images with foreground-only being generated  
    Simply select from scripts -> apply to current model  
    All necessary files will be auto-downloaded on first use  
  - **IP Adapter Masking**:  
    Powerful method of using masking with ip-adapters  
    When combined with multiple ip-adapters, it allows for different inputs guidance for each segment of the input image  
    *Hint*: to create masks, you can use manually created masks or control->mask module with auto-segment to create masks and later upload them  
  - **IP Adapter advanced layer configuration**:  
    Allows for more control over how each layer of ip-adapter is applied, requires a valid dict to be passed as input  
    See [InstantStyle](https://github.com/InstantStyle/InstantStyle) for details  
  - **OneDiff**: new optimization/compile engine, thanks @aifartist  
    As with all other compile engines, enable via *settings -> compute settings -> compile*  
  - [ToDo](https://arxiv.org/html/2402.13573v2) Token Downsampling for Efficient Generation of High-Resolution Images  
    Newer alternative method to [ToMe](https://github.com/dbolya/tomesd) that can provide speed-up with minimal quality loss  
    Enable in *settings -> inference settings -> token merging*  
    Also available in XYZ grid  
  - **Outpaint**:  
    New method of outpainting that uses a combination of auto-masking and edge generation to create seamless transitions between original and generated image  
    Use on control tab:
    - *input -> denoising strength: 0.5 or higher*
    - *select image -> outpaint -> expand edges or zoom out to desired size*
    - *size -> mode: outpaint, method: nearest*
    - *mask -> inpaint masked only (if you want to keep original image)*
  - **Wildcards**:
    - native support of standard file-based wildcards in prompt  
    - enabled by default, can be disabled in *settings -> extra networks* if you want to use 3rd party extension  
    - wildcards folder is set in *settings -> system paths* and can be flat-file list or complex folder structure  
    - matches strings `"__*__"` in positive and negative prompts  
    - supports filename and path-based wildcards  
    - supports nested wildcards (wildcard can refer to another wildcard, etc.)  
    - supports wildcards files in one-choice per line or multiple choices per line separated by `|` format  
    - *note*: this is in addition to previously released style-based wildcards  
- **Models**:
  - **Load UNET**: ability to override/load external UNET to a selected model  
    Works similar to how VAE is selected and loaded: Set UNet folder and UNet model in settings  
    Can be replaced on-the-fly, not just during initial model load  
    Enables usage of fine-tunes such as [DPO-SD15](https://huggingface.co/mhdang/dpo-sd1.5-text2image-v1) or [DPO-SDXL](https://huggingface.co/mhdang/dpo-sdxl-text2image-v1)  
    *Note*: if there is a `JSON` file with the same name as the model it will be used as Unet config, otherwise Unet config from currently loaded model will be used  
  - [PixArt-Σ](https://pixart-alpha.github.io/PixArt-sigma-project/)
    pixart-Σ is a high end diffusion Transformer model (DiT) with a T5 encoder/decoder capable of directly generating images at 4K resolution  
    to use, simply select from *networks -> models -> reference -> PixArt-Σ*  
    *note*: this is a very large model at ~22GB  
    set parameters: *sampler: Default*  
  - [SDXS](https://github.com/IDKiro/sdxs)
    sdxs is an extremely fast 1-step generation consistency model that also uses TAESD as quick VAE out-of-the-box  
    to use, simply select from *networks -> models -> reference -> SDXS*  
    set parameters: *sampler: CMSI, steps: 1, cfg_scale: 0.0*
  - [Hyper-SD](https://huggingface.co/ByteDance/Hyper-SD)  
    sd15 and sdxl 1-step, 2-step, 4-step and 8-step optimized models using lora  
    set parameters: *sampler: TCD or LCM, steps: 1/2/4/8, cfg_scale: 0.0*  
- **UI**:
  - Faster **UI** load times
  - Theme types:  
    **Standard** (built-in themes), **Modern** (experimental nextgen ui), **None** (used for Gradio and Huggingface 3rd party themes)  
    Specifying a theme type updates list of available themes  
    For example, *Gradio* themes will not appear as available if theme type is set to *Standard*  
  - Redesign of base txt2img interface  
  - Minor tweaks to styles: refresh/apply/save
  - See details in [WiKi](https://github.com/vladmandic/automatic/wiki/Themes)
- **API**:
  - Add API endpoint `/sdapi/v1/control` and CLI util `cli/simple-control.py`  
    (in addition to previously added `/sdapi/v1/preprocessors` and `/sdapi/v1/masking`)  
    example:
    > simple-control.py --prompt 'woman in the city' --sampler UniPC --steps 20  
    > --input \~/generative/Samples/cutie-512.png --output /tmp/test.png --processed /tmp/proc.png  
    > --control 'Canny:Canny FP16:0.7, OpenPose:OpenPose FP16:0.8' --type controlnet  
    > --ipadapter 'Plus:~/generative/Samples/cutie-512.png:0.5'  
  - Add API endpoint `/sdapi/v1/vqa` and CLI util `cli/simple-vqa.py`
- **Changes**:
  - Due to change in Diffusers model loading  
    initial model load will now fetch config files required for the model  
    from the Huggingface site instead of using predefined YAML files
  - Removed built-in extensions: *ControlNet* and *Image-Browser*  
    as both *image-browser* and *controlnet* have native built-in equivalents  
    both can still be installed by user if desired  
  - Different defaults depending on available GPU, thanks @Disty0
    - 4GB and below: *lowvram*
    - 8GB and below: *medvram*
    - Cross-attention: Dynamic Attention SDP with *medvram* or *lowvram*, otherwise SDP  
    - VAE Tiling enabled with *medvram* and *lowvram*
    - Disable Extract EMA by default
    - Disable forced VAE Slicing for *lowvram*
  - Upscaler compile disabled by default with OpenVINO backend  
  - Hypernetwork support disabled by default, can be enabled in settings  
- **Improvements**:
  - Faster server startup  
  - Styles apply wildcards to params
  - Face HiRes fully configurable and higher quality when using high-resolution models  
  - Extra networks persistent sort order in settings  
  - Add option to make batch generations use fully random seed vs sequential  
  - Make metadata in full screen viewer optional
  - Add VAE civitai scan metadata/preview
  - More efficient in-browser callbacks
  - Updated all system requirements  
  - UI log monitor will auto-reconnect to server on server restart  
  - UI styles includes indicator for active styles  
  - UI reduce load on browser  
  - Secondary sampler add option "same as primary"  
  - Change attention mechanism on-the-fly without model reload, thanks @Disty0  
  - Update stable-fast with support for torch 2.2.2 and 2.3.0, thanks @Aptronymist
  - Add torch *cudaMallocAsync* in compute options  
    Can improve memory utilization on compatible GPUs (RTX and newer)  
  - Torch dynamic profiling  
    You can enable/disable full torch profiling in settings top menu on-the-fly  
  - Prompt caching - if you use the same prompt multiple times, no need to re-parse and encode it  
    Useful for batches as prompt processing is ~0.1sec on each pass  
  - Enhance `SD_PROMPT_DEBUG` to show actual tokens used
  - Support controlnet manually downloads models in both standalone and diffusers format  
    For standalone, simply copy safetensors file to `models/control/controlnet` folder  
    For diffusers format, create folder with model name in `models/control/controlnet/`  
    and copy `model.json` and `diffusion_pytorch_model.safetensors` to that folder  
- **Samplers**
  - Add *Euler SGM* variation (e.g. SGM Uniform), optimized for SDXL-Lightning models  
    *note*: you can use other samplers as well with SDXL-Lightning models  
  - Add *CMSI* sampler, optimized for consistency models  
  - Add option *timestep spacing* to sampler settings and sampler section in main ui
    Note: changing timestep spacing changes behavior of sampler and can help to make any sampler turbo/lightning compatibile
  - Add option *timesteps* to manually set timesteps instead of relying on steps+spacing  
    Additionally, presets from nVidias align-you-steps reasearch are provided  
    Result is that perfectly aligned steps can drastically reduce number of steps needed!  
    For example, **AYS** preset alows DPM++2M to run in ~10 steps with quality equallying ~30 steps!  
- **IPEX**, thanks @Disty0
  - Update to *IPEX 2.1.20* on Linux  
    requires removing the venv folder to update properly  
  - Removed 1024x1024 workaround  
  - Disable ipexrun by default, set `IPEXRUN=True` if you want to use `ipexrun`  
- **ROCm**, thanks @Disty0  
  - Add support for ROCm 6.1 nighthly builds  
  - Switch to stable branch of PyTorch  
  - Compatibility improvenments  
  - Add **MIGraphX** torch compile engine  
- **ZLUDA**, thanks @lshqqytiger
  - Rewrite ZLUDA installer
  - ZLUDA **v3.8** updates: Runtime API support
  - Add `--reinstall-zluda` (to download the latest ZLUDA)
- **Fixes**:
  - Update requirements
  - Installer automatically handle detached git states  
  - Prompt params parser
  - Allowing forcing LoRA loading method for some or all models
  - Image save without metadata
  - API generate save metadata
  - Face/InstantID faults
  - CivitAI update model info for all models
  - FP16/BF16 test on model load
  - Variation seed possible NaNs
  - Enumerate diffusers model with multiple variants
  - Diffusers skip non-models on enum
  - Face-HiRes compatibility with control modules
  - Face-HiRes avoid doule save in some scenarios
  - Loading safetensors embeddings
  - CSS fixes
  - Check if attention processor is compatible with model
  - SD Upscale when used with control module
  - Noise sampler seed, thanks @leppie
  - Control module with ADetailer and active ControlNet
  - Control module restore button full functionality
  - Control improved handling with multiple control units and different init images
  - Control add correct metadata to image
  - Time embeddings load part of model load
  - A1111 update OptionInfo properties
  - MOTD exception handling
  - Notifications not triggering
  - Prompt cropping on copy

## Update for 2024-03-19

### Highlights 2024-03-19

New models:
- [Stable Cascade](https://github.com/Stability-AI/StableCascade) *Full* and *Lite*
- [Playground v2.5](https://huggingface.co/playgroundai/playground-v2.5-1024px-aesthetic)
- [KOALA 700M](https://github.com/youngwanLEE/sdxl-koala)
- [Stable Video Diffusion XT 1.1](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt-1-1)
- [VGen](https://huggingface.co/ali-vilab/i2vgen-xl)  

New pipelines and features:
- Img2img using [LEdit++](https://leditsplusplus-project.static.hf.space/index.html), context aware method with image analysis and positive/negative prompt handling
- Trajectory Consistency Distillation [TCD](https://mhh0318.github.io/tcd) for processing in even less steps
- Visual Query & Answer using [moondream2](https://github.com/vikhyat/moondream) as an addition to standard interrogate methods
- **Face-HiRes**: simple built-in detailer for face refinements
- Even simpler outpaint: when resizing image, simply pick outpaint method and if image has different aspect ratio, blank areas will be outpainted!
- UI aspect-ratio controls and other UI improvements
- User controllable invisibile and visible watermarking
- Native composable LoRA

What else?

- **Reference models**: *Networks -> Models -> Reference*: All reference models now come with recommended settings that can be auto-applied if desired  
- **Styles**: Not just for prompts! Styles can apply *generate parameters* as templates and can be used to *apply wildcards* to prompts  
improvements, Additional API endpoints  
- Given the high interest in [ZLUDA](https://github.com/vosen/ZLUDA) engine introduced in last release weve updated much more flexible/automatic install procedure (see [wiki](https://github.com/vladmandic/automatic/wiki/ZLUDA) for details)  
- Plus Additional Improvements such as: Smooth tiling, Refine/HiRes workflow improvements, Control workflow  

Further details:  
- For basic instructions, see [README](https://github.com/vladmandic/automatic/blob/master/README.md)  
- For more details on all new features see full [CHANGELOG](https://github.com/vladmandic/automatic/blob/master/CHANGELOG.md)  
- For documentation, see [WiKi](https://github.com/vladmandic/automatic/wiki)
- [Discord](https://discord.com/invite/sd-next-federal-batch-inspectors-1101998836328697867) server  

### Full Changelog 2024-03-19

- [Stable Cascade](https://github.com/Stability-AI/StableCascade) *Full* and *Lite*
  - large multi-stage high-quality model from warp-ai/wuerstchen team and released by stabilityai  
  - download using networks -> reference
  - see [wiki](https://github.com/vladmandic/automatic/wiki/Stable-Cascade) for details
- [Playground v2.5](https://huggingface.co/playgroundai/playground-v2.5-1024px-aesthetic)
  - new model version from Playground: based on SDXL, but with some cool new concepts
  - download using networks -> reference
  - set sampler to *DPM++ 2M EDM* or *Euler EDM*
- [KOALA 700M](https://github.com/youngwanLEE/sdxl-koala)
  - another very fast & light sdxl model where original unet was compressed and distilled to 54% of original size  
  - download using networks -> reference
  - *note* to download fp16 variant (recommended), set settings -> diffusers -> preferred model variant  
- [LEdit++](https://leditsplusplus-project.static.hf.space/index.html)
  - context aware img2img method with image analysis and positive/negative prompt handling  
  - enable via img2img -> scripts -> ledit
  - uses following params from standard img2img: cfg scale (recommended ~3), steps (recommended ~50), denoise strength (recommended ~0.7)
  - can use postive and/or negative prompt to guide editing process
    - positive prompt: what to enhance, strength and threshold for auto-masking
    - negative prompt: what to remove, strength and threshold for auto-masking  
  - *note*: not compatible with model offloading
- **Second Pass / Refine**
  - independent upscale and hires options: run hires without upscale or upscale without hires or both
  - upscale can now run 0.1-8.0 scale and will also run if enabled at 1.0 to allow for upscalers that simply improve image quality
  - update ui section to reflect changes
  - *note*: behavior using backend:original is unchanged for backwards compatibilty
- **Visual Query** visual query & answer in process tab  
  - go to process -> visual query  
  - ask your questions, e.g. "describe the image", "what is behind the subject", "what are predominant colors of the image?"
  - primary model is [moondream2](https://github.com/vikhyat/moondream), a *tiny* 1.86B vision language model  
    *note*: its still 3.7GB in size, so not really tiny  
  - additional support for multiple variations of several base models: *GIT, BLIP, ViLT, PIX*, sizes range from 0.3 to 1.7GB  
- **Video**
  - **Image2Video**
    - new module for creating videos from images  
    - simply enable from *img2img -> scripts -> image2video*  
    - model is auto-downloaded on first use
    - based on [VGen](https://huggingface.co/ali-vilab/i2vgen-xl)  
  - **Stable Video Diffusion**
    - updated with *SVD 1.0, SVD XT 1.0 and SVD XT 1.1*
    - models are auto-downloaded on first use
    - simply enable from *img2img -> scripts -> stable video diffusion*  
    - for svd 1.0, use frames=~14, for xt models use frames=~25
- **Composable LoRA**, thanks @AI-Casanova
  - control lora strength for each step
    for example: `<xxx:0.1@0,0.9@1>` means strength=0.1 for step at 0% and intepolate towards strength=0.9 for step at 100%
  - *note*: this is a very experimental feature and may not work as expected
- **Control**
  - added *refiner/hires* workflows
  - added resize methods to before/after/mask: fixed, crop, fill
- **Styles**: styles are not just for prompts!
  - new styles editor: *networks -> styles -> edit*
  - styles can apply generate parameters, for example to have a style that enables and configures hires:  
    parameters=`enable_hr: True, hr_scale: 2, hr_upscaler: Latent Bilinear antialias, hr_sampler_name: DEIS, hr_second_pass_steps: 20, denoising_strength: 0.5`
  - styles can apply wildcards to prompts, for example:  
    wildcards=`movie=mad max, dune, star wars, star trek; intricate=realistic, color sketch, pencil sketch, intricate`
  - as usual, you can apply any number of styles so you can choose which settings are applied and in which order and which wildcards are used
- **UI**
  - *aspect-ratio** add selector and lock to width/height control  
    allowed aspect ration can be configured via *settings -> user interface*  
  - *interrogate* tab is now merged into *process* tab  
  - *image viewer* now displays image metadata
  - *themes* improve on-the-fly switching
  - *log monitor* flag server warnings/errors and overall improve display
  - *control* separate processor settings from unit settings
- **Face HiRes**
  - new *face restore* option, works similar to well-known *adetailer* by running an inpaint on detected faces but with just a checkbox to enable/disable  
  - set as default face restorer in settings -> postprocessing  
  - disabled by default, to enable simply check *face restore* in your generate advanced settings  
  - strength, steps and sampler are set using by hires section in refine menu  
  - strength can be overriden in settings -> postprocessing  
  - will use secondary prompt and secondary negative prompt if present in refine  
- **Watermarking**
  - SD.Next disables all known watermarks in models, but does allow user to set custom watermark  
  - see *settings -> image options -> watermarking*  
  - invisible watermark: using steganogephy  
  - image watermark: overlaid on top of image  
- **Reference models**
  - additional reference models available for single-click download & run:
    *Stable Cascade, Stable Cascade lite, Stable Video Diffusion XT 1.1*  
  - reference models will now download *fp16* variation by default  
  - reference models will print recommended settings to log if present
  - new setting in extra network: *use reference values when available*  
    disabled by default, if enabled will force use of reference settings for models that have them
- **Samplers**
  - [TCD](https://mhh0318.github.io/tcd/): Trajectory Consistency Distillation  
    new sampler that produces consistent results in a very low number of steps (comparable to LCM but without reliance on LoRA)  
    for best results, use with TCD LoRA: <https://huggingface.co/h1t/TCD-SDXL-LoRA>
  - *DPM++ 2M EDM* and *Euler EDM*  
    EDM is a new solver algorithm currently available for DPM++2M and Euler samplers  
    Note that using EDM samplers with non-EDM optimized models will provide just noise and vice-versa  
- **Improvements**
  - **FaceID** extend support for LoRA, HyperTile and FreeU, thanks @Trojaner
  - **Tiling** now extends to both Unet and VAE producing smoother outputs, thanks @AI-Casanova
  - new setting in image options: *include mask in output*
  - improved params parsing from from prompt string and styles
  - default theme updates and additional built-in theme *black-gray*
  - support models with their own YAML model config files
  - support models with their own JSON per-component config files, for example: `playground-v2.5_vae.config`
  - prompt can have comments enclosed with `/*` and `*/`  
    comments are extracted from prompt and added to image metadata  
- **ROCm**  
  - add **ROCm** 6.0 nightly option to installer, thanks @jicka
  - add *flash attention* support for rdna3, thanks @Disty0  
    install flash_attn package for rdna3 manually and enable *flash attention* from *compute settings*  
    to install flash_attn, activate the venv and run `pip install -U git+https://github.com/ROCm/flash-attention@howiejay/navi_support`  
- **IPEX**
  - disabled IPEX Optimize by default  
- **API**
  - add preprocessor api endpoints  
    GET:`/sdapi/v1/preprocessors`, POST:`/sdapi/v1/preprocess`, sample script:`cli/simple-preprocess.py`
  - add masking api endpoints  
    GET:`/sdapi/v1/masking`, POST:`/sdapi/v1/mask`, sample script:`cli/simple-mask.py`
- **Internal**
  - improved vram efficiency for model compile, thanks @Disty0
  - **stable-fast** compatibility with torch 2.2.1  
  - remove obsolete textual inversion training code
  - remove obsolete hypernetworks training code
- **Refiner** validated workflows:
  - Fully functional: SD15 + SD15, SDXL + SDXL, SDXL + SDXL-R
  - Functional, but result is not as good: SD15 + SDXL, SDXL + SD15, SD15 + SDXL-R
- **SDXL Lightning** models just-work, just makes sure to set CFG Scale to 0  
    and choose a best-suited sampler, it may not be the one youre used to (e.g. maybe even basic Euler)  
- **Fixes**
  - improve *model cpu offload* compatibility
  - improve *model sequential offload* compatibility
  - improve *bfloat16* compatibility
  - improve *xformers* installer to match cuda version and install triton
  - fix extra networks refresh
  - fix *sdp memory attention* in backend original
  - fix autodetect sd21 models
  - fix api info endpoint
  - fix *sampler eta* in xyz grid, thanks @AI-Casanova
  - fix *requires_aesthetics_score* errors
  - fix t2i-canny
  - fix *differenital diffusion* for manual mask, thanks @23pennies
  - fix ipadapter apply/unapply on batch runs
  - fix control with multiple units and override images
  - fix control with hires
  - fix control-lllite
  - fix font fallback, thanks @NetroScript
  - update civitai downloader to handler new metadata
  - improve control error handling
  - use default model variant if specified variant doesnt exist
  - use diffusers lora load override for *lcm/tcd/turbo loras*
  - exception handler around vram memory stats gather
  - improve ZLUDA installer with `--use-zluda` cli param, thanks @lshqqytiger

## Update for 2024-02-22

Only 3 weeks since last release, but heres another feature-packed one!
This time release schedule was shorter as we wanted to get some of the fixes out faster.

### Highlights 2024-02-22

- **IP-Adapters** & **FaceID**: multi-adapter and multi-image suport  
- New optimization engines: [DeepCache](https://github.com/horseee/DeepCache), [ZLUDA](https://github.com/vosen/ZLUDA) and **Dynamic Attention Slicing**  
- New built-in pipelines: [Differential diffusion](https://github.com/exx8/differential-diffusion) and [Regional prompting](https://github.com/huggingface/diffusers/blob/main/examples/community/README.md#regional-prompting-pipeline)  
- Big updates to: **Outpainting** (noised-edge-extend), **Clip-skip** (interpolate with non-integrer values!), **CFG end** (prevent overburn on high CFG scales), **Control** module masking functionality  
- All reported issues since the last release are addressed and included in this release  

Further details:  
- For basic instructions, see [README](https://github.com/vladmandic/automatic/blob/master/README.md)  
- For more details on all new features see full [CHANGELOG](https://github.com/vladmandic/automatic/blob/master/CHANGELOG.md)  
- For documentation, see [WiKi](https://github.com/vladmandic/automatic/wiki)
- [Discord](https://discord.com/invite/sd-next-federal-batch-inspectors-1101998836328697867) server  

### Full ChangeLog for 2024-02-22

- **Improvements**:
  - **IP Adapter** major refactor  
    - support for **multiple input images** per each ip adapter  
    - support for **multiple concurrent ip adapters**  
      *note*: you cannot mix & match ip adapters that use different *CLiP* models, for example `Base` and `Base ViT-G`  
    - add **adapter start/end** to settings, thanks @AI-Casanova  
      having adapter start late can help with better control over composition and prompt adherence  
      having adapter end early can help with overal quality and performance  
    - unified interface in txt2img, img2img and control  
    - enhanced xyz grid support  
  - **FaceID** now also works with multiple input images!  
  - [Differential diffusion](https://github.com/exx8/differential-diffusion)  
    img2img generation where you control strength of each pixel or image area  
    can be used with manually created masks or with auto-generated depth-maps
    uses general denoising strength value  
    simply enable from *img2img -> scripts -> differential diffusion*  
    *note*: supports sd15 and sdxl models  
  - [Regional prompting](https://github.com/huggingface/diffusers/blob/main/examples/community/README.md#regional-prompting-pipeline) as a built-in solution  
    usage is same as original implementation from @hako-mikan  
    click on title to open docs and see examples of full syntax on how to use it  
    simply enable from *scripts -> regional prompting*  
    *note*: supports sd15 models only  
  - [DeepCache](https://github.com/horseee/DeepCache) model acceleration  
    it can produce massive speedups (2x-5x) with no overhead, but with some loss of quality  
    *settings -> compute -> model compile -> deep-cache* and *settings -> compute -> model compile -> cache interval*  
  - [ZLUDA](https://github.com/vosen/ZLUDA) experimental support, thanks @lshqqytiger  
    - ZLUDA is CUDA wrapper that can be used for GPUs without native support
    - best use case is *AMD GPUs on Windows*, see [wiki](https://github.com/vladmandic/automatic/wiki/ZLUDA) for details  
  - **Outpaint** control outpaint now uses new alghorithm: noised-edge-extend  
    new method allows for much larger outpaint areas in a single pass, even outpaint 512->1024 works well  
    note that denoise strength should be increased for larger the outpaint areas, for example outpainting 512->1024 works well with denoise 0.75  
    outpaint can run in *img2img* mode (default) and *inpaint* mode where original image is masked (if inpaint masked only is selected)  
  - **Clip-skip** reworked completely, thanks @AI-Casanova & @Disty0  
    now clip-skip range is 0-12 where previously lowest value was 1 (default is still 1)  
    values can also be decimal to interpolate between different layers, for example `clip-skip: 1.5`, thanks @AI-Casanova  
  - **CFG End** new param to control image generation guidance, thanks @AI-Casanova  
    sometimes you want strong control over composition, but you want it to stop at some point  
    for example, when used with ip-adapters or controlnet, high cfg scale can overpower the guided image  
  - **Control**
    - when performing inpainting, you can specify processing resolution using **size->mask**  
    - units now have extra option to re-use current preview image as processor input  
  - **Cross-attention** refactored cross-attention methods, thanks @Disty0  
    - for backend:original, its unchanged: SDP, xFormers, Doggettxs, InvokeAI, Sub-quadratic, Split attention  
    - for backend:diffuers, list is now: SDP, xFormers, Batch matrix-matrix, Split attention, Dynamic Attention BMM, Dynamic Attention SDP  
      note: you may need to update your settings! Attention Slicing is renamed to Split attention  
    - for ROCm, updated default cross-attention to Scaled Dot Product  
  - **Dynamic Attention Slicing**, thanks @Disty0  
    - dynamically slices attention queries in order to keep them under the slice rate  
      slicing gets only triggered if the query size is larger than the slice rate to gain performance  
      *Dynamic Attention Slicing BMM* uses *Batch matrix-matrix*  
      *Dynamic Attention Slicing SDP* uses *Scaled Dot Product*  
    - *settings -> compute settings -> attention -> dynamic attention slicing*  
  - **ONNX**:  
    - allow specify onnx default provider and cpu fallback  
      *settings -> diffusers*  
    - allow manual install of specific onnx flavor  
      *settings -> onnx*  
    - better handling of `fp16` models/vae, thanks @lshqqytiger  
  - **OpenVINO** update to `torch 2.2.0`, thanks @Disty0  
  - **HyperTile** additional options thanks @Disty0  
    - add swap size option  
    - add use only for hires pass option  
  - add `--theme` cli param to force theme on startup  
  - add `--allow-paths` cli param to add additional paths that are allowed to be accessed via web, thanks @OuticNZ  
- **Wiki**:
  - added benchmark notes for IPEX, OpenVINO and Olive  
  - added ZLUDA wiki page  
- **Internal**
  - update dependencies  
  - refactor txt2img/img2img api  
  - enhanced theme loader  
  - add additional debug env variables  
  - enhanced sdp cross-optimization control  
    see *settings -> compute settings*  
  - experimental support for *python 3.12*  
- **Fixes**:  
  - add variation seed to diffusers txt2img, thanks @AI-Casanova  
  - add cmd param `--skip-env` to skip setting of environment parameters during sdnext load  
  - handle extensions that install conflicting versions of packages  
    `onnxruntime`, `opencv2-python`  
  - installer refresh package cache on any install  
  - fix embeddings registration on server startup, thanks @AI-Casanova  
  - ipex handle dependencies, thanks @Disty0  
  - insightface handle dependencies  
  - img2img mask blur and padding  
  - xyz grid handle ip adapter name and scale  
  - lazy loading of image may prevent metadata from being loaded on time  
  - allow startup without valid models folder  
  - fix interrogate api endpoint  
  - control fix resize causing runtime errors  
  - control fix processor override image after processor change  
  - control fix display grid with batch  
  - control restore pipeline before running scripts/extensions  
  - handle pipelines that return dict instead of object  
  - lora use strict name matching if preferred option is by-filename  
  - fix inpaint mask only for diffusers  
  - fix vae dtype mismatch, thanks @Disty0  
  - fix controlnet inpaint mask  
  - fix theme list refresh  
  - fix extensions update information in ui  
  - fix taesd with bfloat16
  - fix model merge manual merge settings, thanks @AI-Casanova  
  - fix gradio instant update issues for textboxes in quicksettings  
  - fix rembg missing dependency  
  - bind controlnet extension to last known working commit, thanks @Aptronymist  
  - prompts-from-file fix resizable prompt area  

## Update for 2024-02-07

Another big release just hit the shelves!

### Highlights 2024-02-07  

- A lot more functionality in the **Control** module:
  - Inpaint and outpaint support, flexible resizing options, optional hires  
  - Built-in support for many new processors and models, all auto-downloaded on first use  
  - Full support for scripts and extensions  
- Complete **Face** module  
  implements all variations of **FaceID**, **FaceSwap** and latest **PhotoMaker** and **InstantID**  
- Much enhanced **IPAdapter** modules  
- Brand new **Intelligent masking**, manual or automatic  
  Using ML models (*LAMA* object removal, *REMBG* background removal, *SAM* segmentation, etc.) and with live previews  
  With granular blur, erode and dilate controls  
- New models and pipelines:  
  **Segmind SegMoE**, **Mixture Tiling**, **InstaFlow**, **SAG**, **BlipDiffusion**  
- Massive work integrating latest advances with [OpenVINO](https://github.com/vladmandic/automatic/wiki/OpenVINO), [IPEX](https://github.com/vladmandic/automatic/wiki/Intel-ARC) and [ONNX Olive](https://github.com/vladmandic/automatic/wiki/ONNX-Runtime-&-Olive)
- Full control over brightness, sharpness and color shifts and color grading during generate process directly in latent space  
- **Documentation**! This was a big one, with a lot of new content and updates in the [WiKi](https://github.com/vladmandic/automatic/wiki)  

Plus welcome additions to **UI performance, usability and accessibility** and flexibility of deployment as well as **API** improvements  
And it also includes fixes for all reported issues so far  

As of this release, default backend is set to **diffusers** as its more feature rich than **original** and supports many additional models (original backend does remain as fully supported)  

Also, previous versions of **SD.Next** were tuned for balance between performance and resource usage.  
With this release, focus is more on performance.  
See [Benchmark](https://github.com/vladmandic/automatic/wiki/Benchmark) notes for details, but as a highlight, we are now hitting **~110-150 it/s** on a standard nVidia RTX4090 in optimal scenarios!  

Further details:  
- For basic instructions, see [README](https://github.com/vladmandic/automatic/blob/master/README.md)  
- For more details on all new features see full [CHANGELOG](https://github.com/vladmandic/automatic/blob/master/CHANGELOG.md)  
- For documentation, see [WiKi](https://github.com/vladmandic/automatic/wiki)

### Full ChangeLog 2024-02-07  

- Heavily updated [Wiki](https://github.com/vladmandic/automatic/wiki)  
- **Control**:  
  - new docs:
    - [Control overview](https://github.com/vladmandic/automatic/wiki/Control)  
    - [Control guide](https://github.com/vladmandic/automatic/wiki/Control-Guide), thanks @Aptronymist  
  - add **inpaint** support  
    applies to both *img2img* and *controlnet* workflows  
  - add **outpaint** support  
    applies to both *img2img* and *controlnet* workflows  
    *note*: increase denoising strength since outpainted area is blank by default  
  - new **mask** module  
    - granular blur (gaussian), erode (reduce or remove noise) and dilate (pad or expand)  
    - optional **live preview**  
    - optional **auto-segmentation** using ml models  
      auto-segmentation can be done using **segment-anything** models or **rembg** models  
      *note*: auto segmentation will automatically expand user-masked area to segments that include current user mask  
    - optional **auto-mask**  
      if you dont provide mask or mask is empty, you can instead use auto-mask to automatically generate mask  
      this is especially useful if you want to use advanced masking on batch or video inputs and dont want to manually mask each image  
      *note*: such auto-created mask is also subject to all other selected settings such as auto-segmentation, blur, erode and dilate  
    - optional **object removal** using LaMA model  
      remove selected objects from images with a single click  
      works best when combined with auto-segmentation to remove smaller objects  
    - masking can be combined with control processors in which case mask is applied before processor  
    - unmasked part of can is optionally applied to final image as overlay, see settings `mask_apply_overlay`  
  - support for many additional controlnet models  
    now built-in models include 30+ SD15 models and 15+ SDXL models  
  - allow **resize** both *before* and *after* generate operation  
    this allows for workflows such as: *image -> upscale or downscale -> generate -> upscale or downscale -> output*  
    providing more flexibility and than standard hires workflow  
    *note*: resizing before generate can be done using standard upscalers or latent
  - implicit **hires**  
    since hires is only used for txt2img, control reuses existing resize functionality
    any image size is used as txt2img target size  
    but if resize scale is also set its used to additionally upscale image after initial txt2img and for hires pass  
  - add support for **scripts** and **extensions**  
    you can now combine control workflow with your favorite script or extension  
    *note* extensions that are hard-coded for txt2img or img2img tabs may not work until they are updated  
  - add **depth-anything** depth map processor and trained controlnet  
  - add **marigold** depth map processor  
    this is state-of-the-art depth estimation model, but its quite heavy on resources  
  - add **openpose xl** controlnet  
  - add blip/booru **interrogate** functionality to both input and output images  
  - configurable output folder in settings  
  - auto-refresh available models on tab activate  
  - add image preview for override images set per-unit  
  - more compact unit layout  
  - reduce usage of temp files  
  - add context menu to action buttons  
  - move ip-adapter implementation to control tabs  
  - resize by now applies to input image or frame individually  
    allows for processing where input images are of different sizes  
  - support controlnets with non-default yaml config files  
  - implement resize modes for override images  
  - allow any selection of units  
  - dynamically install depenencies required by specific processors  
  - fix input image size  
  - fix video color mode  
  - fix correct image mode  
  - fix batch/folder/video modes  
  - fix processor switching within same unit  
  - fix pipeline switching between different modes  
- **Face** module  
  implements all variations of **FaceID**, **FaceSwap** and latest **PhotoMaker** and **InstantID**  
  simply select from scripts and choose your favorite method and model  
  *note*: all models are auto-downloaded on first use  
  - [FaceID](https://huggingface.co/h94/IP-Adapter-FaceID)  
    - faceid guides image generation given the input image  
    - full implementation for *SD15* and *SD-XL*, to use simply select from *Scripts*  
      **Base** (93MB) uses *InsightFace* to generate face embeds and *OpenCLIP-ViT-H-14* (2.5GB) as image encoder  
      **Plus** (150MB) uses *InsightFace* to generate face embeds and *CLIP-ViT-H-14-laion2B* (3.8GB) as image encoder  
      **SDXL** (1022MB) uses *InsightFace* to generate face embeds and *OpenCLIP-ViT-bigG-14* (3.7GB) as image encoder  
  - [FaceSwap](https://github.com/deepinsight/insightface/blob/master/examples/in_swapper/README.md)  
    - face swap performs face swapping at the end of generation  
    - based on InsightFace in-swapper  
  - [PhotoMaker](https://github.com/TencentARC/PhotoMaker)  
    - for *SD-XL* only  
    - new model from TenencentARC using similar concept as IPAdapter, but with different implementation and  
      allowing full concept swaps between input images and generated images using trigger words  
    - note: trigger word must match exactly one term in prompt for model to work  
  - [InstantID](https://github.com/InstantID/InstantID)  
    - for *SD-XL* only  
    - based on custom trained ip-adapter and controlnet combined concepts  
    - note: controlnet appears to be heavily watermarked  
  - enable use via api, thanks @trojaner  
- [IPAdapter](https://huggingface.co/h94/IP-Adapter)  
  - additional models for *SD15* and *SD-XL*, to use simply select from *Scripts*:  
    **SD15**: Base, Base ViT-G, Light, Plus, Plus Face, Full Face  
    **SDXL**: Base SDXL, Base ViT-H SDXL, Plus ViT-H SDXL, Plus Face ViT-H SDXL  
  - enable use via api, thanks @trojaner  
- [Segmind SegMoE](https://github.com/segmind/segmoe)  
  - initial support for reference models  
    download&load via network -> models -> reference -> **SegMoE SD 4x2** (3.7GB), **SegMoE XL 2x1** (10GB), **SegMoE XL 4x2**  
  - note: since segmoe is basically sequential mix of unets from multiple models, it can get large  
    SD 4x2 is ~4GB, XL 2x1 is ~10GB and XL 4x2 is 18GB  
  - supports lora, thanks @AI-Casanova
  - support for create and load custom mixes will be added in the future  
- [Mixture Tiling](https://arxiv.org/abs/2302.02412)  
  - uses multiple prompts to guide different parts of the grid during diffusion process  
  - can be used ot create complex scenes with multiple subjects  
  - simply select from scripts  
- [Self-attention guidance](https://github.com/SusungHong/Self-Attention-Guidance)  
  - simply select scale in advanced menu  
  - can drastically improve image coherence as well as reduce artifacts  
  - note: only compatible with some schedulers  
- [FreeInit](https://tianxingwu.github.io/pages/FreeInit/) for **AnimateDiff**
  - greatly improves temporal consistency of generated outputs  
  - all options are available in animateddiff script  
- [SalesForce BlipDiffusion](https://huggingface.co/docs/diffusers/api/pipelines/blip_diffusion)  
  - model can be used to place subject in a different context  
  - requires input image  
  - last word in prompt and negative prompt will be used as source and target subjects  
  - sampler must be set to default before loading the model  
- [InstaFlow](https://github.com/gnobitab/InstaFlow)  
  - another take on super-fast image generation in a single step  
  - set *sampler:default, steps:1, cfg-scale:0*  
  - load from networks -> models -> reference  
- **Improvements**  
  - **ui**  
    - check version and **update** SD.Next via UI  
      simply go to: settings -> update
    - globally configurable **font size**  
      will dynamically rescale ui depending on settings -> user interface  
    - built-in **themes** can be changed on-the-fly  
      this does not work with gradio-default themes as css is created by gradio itself  
    - two new **themes**: *simple-dark* and *simple-light*  
    - modularized blip/booru interrogate  
      now appears as toolbuttons on image/gallery output  
    - faster browser page load  
    - update hints, thanks @brknsoul  
    - cleanup settings  
  - **server**
    - all move/offload options are disable by default for optimal performance  
      enable manually if low on vram  
  - **server startup**: performance  
    - reduced module imports  
      ldm support is now only loaded when running in backend=original  
    - faster extension load  
    - faster json parsing  
    - faster lora indexing  
    - lazy load optional imports  
    - batch embedding load, thanks @midcoastal and @AI-Casanova  
      10x+ faster embeddings load for large number of embeddings, now works for 1000+ embeddings  
    - file and folder list caching, thanks @midcoastal
      if you have a lot of files and and/or are using slower or non-local storage, this speeds up file access a lot  
    - add `SD_INSTALL_DEBUG` env variable to trace all `git` and `pip` operations
  - **extra networks**  
    - 4x faster civitai metadata and previews lookup  
    - better display and selection of tags & trigger words  
      if hashes are calculated, trigger words will only be displayed for actual model version  
    - better matching of previews  
    - better search, including searching for multiple keywords or using full regex  
      see wiki page for more details on syntax  
      thanks @NetroScript  
    - reduce html overhead  
  - **model compression**, thanks @Disty0  
    - using built-in NNCF model compression, you can reduce the size of your models significantly  
      example: up to 3.4GB of VRAM saved for SD-XL model!  
    - see [wiki](https://github.com/vladmandic/automatic/wiki/Model-Compression-with-NNCF) for details  
  - **embeddings**  
    you can now use sd 1.5 embeddings with your sd-xl models!, thanks @AI-Casanova  
    conversion is done on-the-fly, is completely transparent and result is an approximation of embedding  
    to enable: settings->extra networks->auto-convert embeddings  
  - **offline deployment**: allow deployment without git clone  
    for example, you can now deploy a zip of the sdnext folder  
  - **latent upscale**: updated latent upscalers (some are new)  
    *nearest, nearest-exact, area, bilinear, bicubic, bilinear-antialias, bicubic-antialias*
  - **scheduler**: added `SA Solver`  
  - **model load to gpu**  
    new option in settings->diffusers allowing models to be loaded directly to GPU while keeping RAM free  
    this option is not compatible with any kind of model offloading as model is expected to stay in GPU  
    additionally, all model-moves can now be traced with env variable `SD_MOVE_DEBUG`  
  - **xyz grid**
    - range control  
      example: `5.0-6.0:3` will generate 3 images with values `5.0,5.5,6.0`  
      example: `10-20:4` will generate 4 images with values `10,13,16,20`  
    - continue on error  
      now you can use xyz grid with different params and test which ones work and which dont  
    - correct font scaling, thanks @nCoderGit  
  - **hypertile**  
    - enable vae tiling  
    - add autodetect optimial value  
      set tile size to 0 to use autodetected value  
  - **cli**  
    - `sdapi.py` allow manual api invoke  
      example: `python cli/sdapi.py /sdapi/v1/sd-models`  
    - `image-exif.py` improve metadata parsing  
    - `install-sf` helper script to automatically find best available stable-fast package for the platform  
  - **memory**: add ram usage monitoring in addition to gpu memory usage monitoring  
  - **vae**: enable taesd batch decode  
    enable/disable with settings -> diffusers > vae slicing  
- **compile**
  - new option: **fused projections**  
    pretty much free 5% performance boost for compatible models  
    enable in settings -> compute settings  
  - new option: **dynamic quantization** (experimental)  
    reduces memory usage and increases performance  
    enable in settings -> compute settings  
    best used together with torch compile: *inductor*  
    this feature is highly experimental and will evolve over time  
    requires nightly versions of `torch` and `torchao`  
    > `pip install -U --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121`  
    > `pip install -U git+https://github.com/pytorch-labs/ao`  
  - new option: **compile text encoder** (experimental)  
- **correction**  
  - new section in generate, allows for image corrections during generataion directly in latent space  
  - adds *brightness*, *sharpness* and *color* controls, thanks @AI-Casanova
  - adds *color grading* controls, thanks @AI-Casanova
  - replaces old **hdr** section
- **IPEX**, thanks @disty0  
  - see [wiki](https://github.com/vladmandic/automatic/wiki/Intel-ARC) for details  
  - rewrite ipex hijacks without CondFunc  
    improves compatibilty and performance  
    fixes random memory leaks  
  - out of the box support for Intel Data Center GPU Max Series  
  - remove IPEX / Torch 2.0 specific hijacks  
  - add `IPEX_SDPA_SLICE_TRIGGER_RATE`, `IPEX_ATTENTION_SLICE_RATE` and `IPEX_FORCE_ATTENTION_SLICE` env variables  
  - disable 1024x1024 workaround if the GPU supports 64 bit  
  - fix lock-ups at very high resolutions  
- **OpenVINO**, thanks @disty0  
  - see [wiki](https://github.com/vladmandic/automatic/wiki/OpenVINO) for details  
  - **quantization support with NNCF**  
    run 8 bit directly without autocast  
    enable *OpenVINO Quantize Models with NNCF* from *Compute Settings*  
  - **4-bit support with NNCF**  
    enable *Compress Model weights with NNCF* from *Compute Settings* and set a 4-bit NNCF mode  
    select both CPU and GPU from the device selection if you want to use 4-bit or 8-bit modes on GPU  
  - experimental support for *Text Encoder* compiling  
    OpenVINO is faster than IPEX now  
  - update to OpenVINO 2023.3.0  
  - add device selection to `Compute Settings`  
    selecting multiple devices will use `HETERO` device  
  - remove `OPENVINO_TORCH_BACKEND_DEVICE` env variable  
  - reduce system memory usage after compile  
  - fix cache loading with multiple models  
- **Olive** support, thanks @lshqqytiger
  - fully merged in in [wiki](https://github.com/vladmandic/automatic/wiki/ONNX-Runtime-&-Olive), see wiki for details  
  - as a highlight, 4-5 it/s using DirectML on AMD GPU translates to 23-25 it/s using ONNX/Olive!  
- **fixes**  
  - civitai model download: enable downloads of embeddings
  - ipadapter: allow changing of model/image on-the-fly  
  - ipadapter: fix fallback of cross-attention on unload  
  - rebasin iterations, thanks @AI-Casanova
  - prompt scheduler, thanks @AI-Casanova
  - python: fix python 3.9 compatibility  
  - sdxl: fix positive prompt embeds
  - img2img: clip and blip interrogate  
  - img2img: sampler selection offset  
  - img2img: support variable aspect ratio without explicit resize  
  - cli: add `simple-upscale.py` script  
  - cli: fix cmd args parsing  
  - cli: add `run-benchmark.py` script  
  - api: add `/sdapi/v1/version` endpoint
  - api: add `/sdapi/v1/platform` endpoint
  - api: return current image in progress api if requested  
  - api: sanitize response object  
  - api: cleanup error logging  
  - api: fix api-only errors  
  - api: fix image to base64
  - api: fix upscale  
  - refiner: fix use of sd15 model as refiners in second pass  
  - refiner: enable none as option in xyz grid  
  - sampler: add sampler options info to metadata
  - sampler: guard against invalid sampler index  
  - sampler: add img2img_extra_noise option
  - config: reset default cfg scale to 6.0  
  - hdr: fix math, thanks @AI-Casanova
  - processing: correct display metadata  
  - processing: fix batch file names  
  - live preview: fix when using `bfloat16`  
  - live preview: add thread locking  
  - upscale: fix ldsr
  - huggingface: handle fallback model variant on load  
  - reference: fix links to models and use safetensors where possible  
  - model merge: unbalanced models where not all keys are present, thanks @AI-Casanova
  - better sdxl model detection
  - global crlf->lf switch  
  - model type switch if there is loaded submodels  
  - cleanup samplers use of compute devices, thanks @Disty0  
- **other**  
  - extensions `sd-webui-controlnet` is locked to commit `ecd33eb` due to breaking changes  
  - extension `stable-diffusion-webui-images-browser` is locked to commit `27fe4a7` due to breaking changes  
  - updated core requirements  
  - fully dynamic pipelines  
    pipeline switch is now done on-the-fly and does not require manual initialization of individual components  
    this allows for quick implementation of new pipelines  
    see `modules/sd_models.py:switch_pipe` for details  
  - major internal ui module refactoring  
    this may cause compatibility issues if an extension is doing a direct import from `ui.py`  
    in which case, report it so we can add a compatibility layer  
  - major public api refactoring  
    this may cause compatibility issues if an extension is doing a direct import from `api.py` or `models.py`  
    in which case, report it so we can add a compatibility layer  

## Update for 2023-12-29

To wrap up this amazing year, were releasing a new version of [SD.Next](https://github.com/vladmandic/automatic), this one is absolutely massive!  

### Highlights 2023-12-29

- Brand new Control module for *text, image, batch and video* processing  
  Native implementation of all control methods for both *SD15* and *SD-XL*  
  ▹ **ControlNet | ControlNet XS | Control LLLite | T2I Adapters | IP Adapters**  
  For details, see [Wiki](https://github.com/vladmandic/automatic/wiki/Control) documentation:  
- Support for new models types out-of-the-box  
  This brings number of supported t2i/i2i model families to 13!  
  ▹ **Stable Diffusion 1.5/2.1 | SD-XL | LCM | Segmind | Kandinsky | Pixart-α | Würstchen | aMUSEd | DeepFloyd IF | UniDiffusion | SD-Distilled | BLiP Diffusion | etc.**  
- New video capabilities:  
  ▹ **AnimateDiff | SVD | ModelScope | ZeroScope**  
- Enhanced platform support  
  ▹ **Windows | Linux | MacOS** with **nVidia | AMD | IntelArc | DirectML | OpenVINO | ONNX+Olive** backends  
- Better onboarding experience (first install)  
  with all model types available for single click download & load (networks -> reference)  
- Performance optimizations!
  For comparisment of different processing options and compile backends, see [Wiki](https://github.com/vladmandic/automatic/wiki/Benchmark)  
  As a highlight, were reaching **~100 it/s** (no tricks, this is with full features enabled and end-to-end on a standard nVidia RTX4090)  
- New [custom pipelines](https://github.com/vladmandic/automatic/blob/dev/scripts/example.py) framework for quickly porting any new pipeline  

And others improvements in areas such as: Upscaling (up to 8x now with 40+ available upscalers), Inpainting (better quality), Prompt scheduling, new Sampler options, new LoRA types, additional UI themes, better HDR processing, built-in Video interpolation, parallel Batch processing, etc.  

Plus some nifty new modules such as **FaceID** automatic face guidance using embeds during generation and **Depth 3D** image to 3D scene

### Full ChangeLog 2023-12-29

- **Control**  
  - native implementation of all image control methods:  
    **ControlNet**, **ControlNet XS**, **Control LLLite**, **T2I Adapters** and **IP Adapters**  
  - top-level **Control** next to **Text** and **Image** generate  
  - supports all variations of **SD15** and **SD-XL** models  
  - supports *Text*, *Image*, *Batch* and *Video* processing  
  - for details and list of supported models and workflows, see Wiki documentation:  
    <https://github.com/vladmandic/automatic/wiki/Control>  
- **Diffusers**  
  - [Segmind Vega](https://huggingface.co/segmind/Segmind-Vega) model support  
    - small and fast version of **SDXL**, only 3.1GB in size!  
    - select from *networks -> reference*  
  - [aMUSEd 256](https://huggingface.co/amused/amused-256) and [aMUSEd 512](https://huggingface.co/amused/amused-512) model support  
    - lightweigt models that excel at fast image generation  
    - *note*: must select: settings -> diffusers -> generator device: unset
    - select from *networks -> reference*
  - [Playground v1](https://huggingface.co/playgroundai/playground-v1), [Playground v2 256](https://huggingface.co/playgroundai/playground-v2-256px-base), [Playground v2 512](https://huggingface.co/playgroundai/playground-v2-512px-base), [Playground v2 1024](https://huggingface.co/playgroundai/playground-v2-1024px-aesthetic) model support  
    - comparable to SD15 and SD-XL, trained from scratch for highly aesthetic images  
    - simply select from *networks -> reference* and use as usual  
  - [BLIP-Diffusion](https://dxli94.github.io/BLIP-Diffusion-website/)  
    - img2img model that can replace subjects in images using prompt keywords  
    - download and load by selecting from *networks -> reference -> blip diffusion*
    - in image tab, select `blip diffusion` script
  - [DemoFusion](https://github.com/PRIS-CV/DemoFusion) run your SDXL generations at any resolution!  
    - in **Text** tab select *script* -> *demofusion*  
    - *note*: GPU VRAM limits do not automatically go away so be careful when using it with large resolutions  
      in the future, expect more optimizations, especially related to offloading/slicing/tiling,  
      but at the moment this is pretty much experimental-only  
  - [AnimateDiff](https://github.com/guoyww/animatediff/)  
    - overall improved quality  
    - can now be used with *second pass* - enhance, upscale and hires your videos!  
  - [IP Adapter](https://github.com/tencent-ailab/IP-Adapter)  
    - add support for **ip-adapter-plus_sd15, ip-adapter-plus-face_sd15 and ip-adapter-full-face_sd15**  
    - can now be used in *xyz-grid*  
  - **Text-to-Video**  
    - in text tab, select `text-to-video` script  
    - supported models: **ModelScope v1.7b, ZeroScope v1, ZeroScope v1.1, ZeroScope v2, ZeroScope v2 Dark, Potat v1**  
      *if you know of any other t2v models youd like to see supported, let me know!*  
    - models are auto-downloaded on first use  
    - *note*: current base model will be unloaded to free up resources  
  - **Prompt scheduling** now implemented for Diffusers backend, thanks @AI-Casanova
  - **Custom pipelines** contribute by adding your own custom pipelines!  
    - for details, see fully documented example:  
      <https://github.com/vladmandic/automatic/blob/dev/scripts/example.py>  
  - **Schedulers**  
    - add timesteps range, changing it will make scheduler to be over-complete or under-complete  
    - add rescale betas with zero SNR option (applicable to Euler, Euler a and DDIM, allows for higher dynamic range)  
  - **Inpaint**  
    - improved quality when using mask blur and padding  
  - **UI**  
    - 3 new native UI themes: **orchid-dreams**, **emerald-paradise** and **timeless-beige**, thanks @illu_Zn
    - more dynamic controls depending on the backend (original or diffusers)  
      controls that are not applicable in current mode are now hidden  
    - allow setting of resize method directly in image tab  
      (previously via settings -> upscaler_for_img2img)  
- **Optional**
  - **FaceID** face guidance during generation  
    - also based on IP adapters, but with additional face detection and external embeddings calculation  
    - calculates face embeds based on input image and uses it to guide generation  
    - simply select from *scripts -> faceid*  
    - *experimental module*: requirements must be installed manually:  
        > pip install insightface ip_adapter  
  - **Depth 3D** image to 3D scene
    - delivered as an extension, install from extensions tab  
      <https://github.com/vladmandic/sd-extension-depth3d>  
    - creates fully compatible 3D scene from any image by using depth estimation  
      and creating a fully populated mesh  
    - scene can be freely viewed in 3D in the UI itself or downloaded for use in other applications  
  - [ONNX/Olive](https://github.com/vladmandic/automatic/wiki/ONNX-Olive)  
    - major work continues in olive branch, see wiki for details, thanks @lshqqytiger  
      as a highlight, 4-5 it/s using DirectML on AMD GPU translates to 23-25 it/s using ONNX/Olive!  
- **General**  
  - new **onboarding**  
    - if no models are found during startup, app will no longer ask to download default checkpoint  
      instead, it will show message in UI with options to change model path or download any of the reference checkpoints  
    - *extra networks -> models -> reference* section is now enabled for both original and diffusers backend  
  - support for **Torch 2.1.2** (release) and **Torch 2.3** (dev)  
  - **Process** create videos from batch or folder processing  
      supports *GIF*, *PNG* and *MP4* with full interpolation, scene change detection, etc.  
  - **LoRA**  
    - add support for block weights, thanks @AI-Casanova  
      example `<lora:SDXL_LCM_LoRA:1.0:in=0:mid=1:out=0>`  
    - add support for LyCORIS GLora networks  
    - add support for LoRA PEFT (*Diffusers*) networks  
    - add support for Lora-OFT (*Kohya*) and Lyco-OFT (*Kohaku*) networks  
    - reintroduce alternative loading method in settings: `lora_force_diffusers`  
    - add support for `lora_fuse_diffusers` if using alternative method  
      use if you have multiple complex loras that may be causing performance degradation  
      as it fuses lora with model during load instead of interpreting lora on-the-fly  
  - **CivitAI downloader** allow usage of access tokens for download of gated or private models  
  - **Extra networks** new *settting -> extra networks -> build info on first access*  
    indexes all networks on first access instead of server startup  
  - **IPEX**, thanks @disty0  
    - update to **Torch 2.1**  
      if you get file not found errors, set `DISABLE_IPEXRUN=1` and run the webui with `--reinstall`  
    - built-in *MKL* and *DPCPP* for IPEX, no need to install OneAPI anymore  
    - **StableVideoDiffusion** is now supported with IPEX  
    - **8 bit support with NNCF** on Diffusers backend  
    - fix IPEX Optimize not applying with Diffusers backend  
    - disable 32bit workarounds if the GPU supports 64bit  
    - add `DISABLE_IPEXRUN` and `DISABLE_IPEX_1024_WA` environment variables  
    - performance and compatibility improvements  
  - **OpenVINO**, thanks @disty0  
    - **8 bit support for CPUs**  
    - reduce System RAM usage  
    - update to Torch 2.1.2  
    - add *Directory for OpenVINO cache* option to *System Paths*  
    - remove Intel ARC specific 1024x1024 workaround  
  - **HDR controls**  
    - batch-aware for enhancement of multiple images or video frames  
    - available in image tab  
  - **Logging**
    - additional *TRACE* logging enabled via specific env variables  
      see <https://github.com/vladmandic/automatic/wiki/Debug> for details  
    - improved profiling  
      use with `--debug --profile`  
    - log output file sizes  
  - **Other**  
    - **API** several minor but breaking changes to API behavior to better align response fields, thanks @Trojaner
    - **Inpaint** add option `apply_overlay` to control if inpaint result should be applied as overlay or as-is  
      can remove artifacts and hard edges of inpaint area but also remove some details from original  
    - **chaiNNer** fix `NaN` issues due to autocast  
    - **Upscale** increase limit from 4x to 8x given the quality of some upscalers  
    - **Networks** fix sort  
    - reduced default **CFG scale** from 6 to 4 to be more out-of-the-box compatibile with LCM/Turbo models
    - disable google fonts check on server startup  
    - fix torchvision/basicsr compatibility  
    - fix styles quick save  
    - add hdr settings to metadata  
    - improve handling of long filenames and filenames during batch processing  
    - do not set preview samples when using via api  
    - avoid unnecessary resizes in img2img and inpaint  
    - safe handling of config updates avoid file corruption on I/O errors  
    - updated `cli/simple-txt2img.py` and `cli/simple-img2img.py` scripts  
    - save `params.txt` regardless of image save status  
    - update built-in log monitor in ui, thanks @midcoastal  
    - major CHANGELOG doc cleanup, thanks @JetVarimax  
    - major INSTALL doc cleanup, thanks JetVarimax  

## Update for 2023-12-04

Whats new? Native video in SD.Next via both **AnimateDiff** and **Stable-Video-Diffusion** - and including native MP4 encoding and smooth video outputs out-of-the-box, not just animated-GIFs.  
Also new is support for **SDXL-Turbo** as well as new **Kandinsky 3** models and cool latent correction via **HDR controls** for any *txt2img* workflows, best-of-class **SDXL model merge** using full ReBasin methods and further mobile UI optimizations.  

- **Diffusers**
  - **IP adapter**
    - lightweight native implementation of T2I adapters which can guide generation towards specific image style  
    - supports most T2I models, not limited to SD 1.5  
    - models are auto-downloaded on first use
  - **AnimateDiff**
    - lightweight native implementation of AnimateDiff models:  
      *AnimateDiff 1.4, 1.5 v1, 1.5 v2, AnimateFace*
    - supports SD 1.5 only  
    - models are auto-downloaded on first use  
    - for video saving support, see video support section
    - can be combined with IP-Adapter for even better results!  
  - **HDR latent control**, based on [article](https://huggingface.co/blog/TimothyAlexisVass/explaining-the-sdxl-latent-space#long-prompts-at-high-guidance-scales-becoming-possible)  
    - in *Advanced* params
    - allows control of *latent clamping*, *color centering* and *range maximization*  
    - supported by *XYZ grid*  
  - [SD21 Turbo](https://huggingface.co/stabilityai/sd-turbo) and [SDXL Turbo](https://huggingface.co/stabilityai/sdxl-turbo) support  
    - just set CFG scale (0.0-1.0) and steps (1-3) to a very low value  
    - compatible with original StabilityAI SDXL-Turbo or any of the newer merges
    - download safetensors or select from networks -> reference
  - [Stable Video Diffusion](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid) and [Stable Video Diffusion XT](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt) support  
    - download using built-in model downloader or simply select from *networks -> reference*  
      support for manually downloaded safetensors models will be added later  
    - for video saving support, see video support section
    - go to *image* tab, enter input image and select *script* -> *stable video diffusion*
  - [Kandinsky 3](https://huggingface.co/kandinsky-community/kandinsky-3) support  
    - download using built-in model downloader or simply select from *networks -> reference*  
    - this model is absolutely massive at 27.5GB at fp16, so be patient  
    - model params count is at 11.9B (compared to SD-XL at 3.3B) and its trained on mixed resolutions from 256px to 1024px  
    - use either model offload or sequential cpu offload to be able to use it  
  - better autodetection of *inpaint* and *instruct* pipelines  
  - support long seconary prompt for refiner  
- **Video support**
  - applies to any model that supports video generation, e.g. AnimateDiff and StableVideoDiffusion  
  - support for **animated-GIF**, **animated-PNG** and **MP4**  
  - GIF and PNG can be looped  
  - MP4 can have additional padding at the start/end as well as motion-aware interpolated frames for smooth playback  
    interpolation is done using [RIFE](https://arxiv.org/abs/2011.06294) with native implementation in SD.Next  
    And its fast - interpolation from 16 frames with 10x frames to target 160 frames results takes 2-3sec
  - output folder for videos is in *settings -> image paths -> video*  
- **General**  
  - redesigned built-in profiler  
    - now includes both `python` and `torch` and traces individual functions  
    - use with `--debug --profile`  
  - **model merge** add **SD-XL ReBasin** support, thanks @AI-Casanova  
  - further UI optimizations for **mobile devices**, thanks @iDeNoh  
  - log level defaults to info for console and debug for log file  
  - better prompt display in process tab  
  - increase maximum lora cache values  
  - fix extra networks sorting
  - fix controlnet compatibility issues in original backend  
  - fix img2img/inpaint paste params  
  - fix save text file for manually saved images  
  - fix python 3.9 compatibility issues  

## Update for 2023-11-23

New release, primarily focused around three major new features: full **LCM** support, completely new **Model Merge** functionality and **Stable-fast** compile support  
Also included are several other improvements and large number of hotfixes - see full changelog for details  

- **Diffusers**  
  - **LCM** support for any *SD 1.5* or *SD-XL* model!  
    - download [lcm-lora-sd15](https://huggingface.co/latent-consistency/lcm-lora-sdv1-5/tree/main) and/or [lcm-lora-sdxl](https://huggingface.co/latent-consistency/lcm-lora-sdxl/tree/main)  
    - load for favorite *SD 1.5* or *SD-XL* model *(original LCM was SD 1.5 only, this is both)*  
    - load **lcm lora** *(note: lcm lora is processed differently than any other lora)*  
    - set **sampler** to **LCM**  
    - set number of steps to some low number, for SD-XL 6-7 steps is normally sufficient  
      note: LCM scheduler does not support steps higher than 50  
    - set CFG to between 1 and 2  
  - Add `cli/lcm-convert.py` script to convert any SD 1.5 or SD-XL model to LCM model  
    by baking in LORA and uploading to Huggingface, thanks @Disty0  
  - Support for [Stable Fast](https://github.com/chengzeyi/stable-fast) model compile on *Windows/Linux/WSL2* with *CUDA*  
    See [Wiki:Benchmark](https://github.com/vladmandic/automatic/wiki/Benchmark) for details and comparison  
    of different backends, precision modes, advanced settings and compile modes  
    *Hint*: **70+ it/s** is possible on *RTX4090* with no special tweaks  
  - Add additional pipeline types for manual model loads when loading from `safetensors`  
  - Updated logic for calculating **steps** when using base/hires/refiner workflows  
  - Improve **model offloading** for both model and sequential cpu offload when dealing with meta tensors
  - Safe model offloading for non-standard models  
  - Fix **DPM SDE** scheduler  
  - Better support for SD 1.5 **inpainting** models  
  - Add support for **OpenAI Consistency decoder VAE**
  - Enhance prompt parsing with long prompts and support for *BREAK* keyword  
    Change-in-behavior: new line in prompt now means *BREAK*  
  - Add alternative Lora loading algorithm, triggered if `SD_LORA_DIFFUSERS` is set  
- **Models**
  - **Model merge**
    - completely redesigned, now based on best-of-class `meh` by @s1dlx  
      and heavily modified for additional functionality and fully integrated by @AI-Casanova (thanks!)  
    - merge SD or SD-XL models using *simple merge* (12 methods),  
      using one of *presets* (20 built-in presets) or custom block merge values  
    - merge with ReBasin permutations and/or clipping protection  
    - fully multithreaded for fastest merge possible  
  - **Model update**  
    - under UI -> Models - Update  
    - scan existing models for updated metadata on CivitAI and  
      provide download functionality for models with available  
- **Extra networks**  
  - Use multi-threading for 5x load speedup  
  - Better Lora trigger words support  
  - Auto refresh styles on change  
- **General**  
  - Many **mobile UI** optimizations, thanks @iDeNoh
  - Support for **Torch 2.1.1** with CUDA 12.1 or CUDA 11.8  
  - Configurable location for HF cache folder  
    Default is standard `~/.cache/huggingface/hub`  
  - Reworked parser when pasting previously generated images/prompts  
    includes all `txt2img`, `img2img` and `override` params  
  - Reworked **model compile**
  - Support custom upscalers in subfolders  
  - Add additional image info when loading image in process tab  
  - Better file locking when sharing config and/or models between multiple instances  
  - Handle custom API endpoints when using auth  
  - Show logged in user in log when accessing via UI and/or API  
  - Support `--ckpt none` to skip loading a model  
- **XYZ grid**
  - Add refiner options to XYZ Grid  
  - Add option to create only subgrids in XYZ grid, thanks @midcoastal
  - Allow custom font, background and text color in settings
- **Fixes**  
  - Fix `params.txt` saved before actual image
  - Fix inpaint  
  - Fix manual grid image save  
  - Fix img2img init image save  
  - Fix upscale in txt2img for batch counts when no hires is used  
  - More uniform models paths  
  - Safe scripts callback execution  
  - Improved extension compatibility  
  - Improved BF16 support  
  - Match previews for reference models with downloaded models

## Update for 2023-11-06

Another pretty big release, this time with focus on new models (3 new model types), new backends and optimizations
Plus quite a few fixes  

Also, [Wiki](https://github.com/vladmandic/automatic/wiki) has been updated with new content, so check it out!  
Some highlights: [OpenVINO](https://github.com/vladmandic/automatic/wiki/OpenVINO), [IntelArc](https://github.com/vladmandic/automatic/wiki/Intel-ARC), [DirectML](https://github.com/vladmandic/automatic/wiki/DirectML), [ONNX/Olive](https://github.com/vladmandic/automatic/wiki/ONNX-Olive)

- **Diffusers**
  - since now **SD.Next** supports **12** different model types, weve added reference model for each type in  
    *Extra networks -> Reference* for easier select & auto-download  
    Models can still be downloaded manually, this is just a convenience feature & a showcase for supported models  
  - new model type: [Segmind SSD-1B](https://huggingface.co/segmind/SSD-1B)  
    its a *distilled* model trained at 1024px, this time 50% smaller and faster version of SD-XL!  
    (and quality does not suffer, its just more optimized)  
    test shows batch-size:4 with 1k images at full quality used less than 6.5GB of VRAM  
    and for further optimization, you can use built-in **TAESD** decoder,  
    which results in batch-size:16 with 1k images using 7.9GB of VRAM
    select from extra networks -> reference or download using built-in **Huggingface** downloader: `segmind/SSD-1B`  
  - new model type: [Pixart-α XL 2](https://github.com/PixArt-alpha/PixArt-alpha)  
    in medium/512px and large/1024px variations  
    comparable in quality to SD 1.5 and SD-XL, but with better text encoder and highly optimized training pipeline  
    so finetunes can be done in as little as 10% compared to SD/SD-XL (note that due to much larger text encoder, it is a large model)  
    select from extra networks -> reference or download using built-in **Huggingface** downloader: `PixArt-alpha/PixArt-XL-2-1024-MS`  
  - new model type: [LCM: Latent Consistency Models](https://github.com/openai/consistency_models)  
    trained at 512px, but with near-instant generate in a as little as 3 steps!  
    combined with OpenVINO, generate on CPU takes less than 5-10 seconds: <https://www.youtube.com/watch?v=b90ESUTLsRo>  
    and absolute beast when combined with **HyperTile** and **TAESD** decoder resulting in **28 FPS**  
    (on RTX4090 for batch 16x16 at 512px)  
    note: set sampler to **Default** before loading model as LCM comes with its own *LCMScheduler* sampler  
    select from extra networks -> reference or download using built-in **Huggingface** downloader: `SimianLuo/LCM_Dreamshaper_v7`  
  - support for **Custom pipelines**, thanks @disty0  
    download using built-in **Huggingface** downloader  
    think of them as plugins for diffusers not unlike original extensions that modify behavior of `ldm` backend  
    list of community pipelines: <https://github.com/huggingface/diffusers/blob/main/examples/community/README.md>  
  - new custom pipeline: `Disty0/zero123plus-pipeline`, thanks @disty0  
    generate 4 output images with different camera positions: front, side, top, back!  
    for more details, see <https://github.com/vladmandic/automatic/discussions/2421>  
  - new backend: **ONNX/Olive** *(experimental)*, thanks @lshqqytiger  
    for details, see [WiKi](https://github.com/vladmandic/automatic/wiki/ONNX-Runtime)
  - extend support for [Free-U](https://github.com/ChenyangSi/FreeU)  
    improve generations quality at no cost (other than finding params that work for you)  
- **General**  
  - attempt to auto-fix invalid samples which occur due to math errors in lower precision  
    example: `RuntimeWarning: invalid value encountered in cast: sample = sample.astype(np.uint8)`  
    begone **black images** *(note: if it proves as working, this solution will need to be expanded to cover all scenarios)*  
  - add **Lora OFT** support, thanks @antis0007 and @ai-casanova  
  - **Upscalers**  
    - **compile** option, thanks @disty0  
    - **chaiNNer** add high quality models from [Helaman](https://openmodeldb.info/users/helaman)  
  - redesigned **Progress bar** with full details on current operation  
  - new option: *settings -> images -> keep incomplete*  
    can be used to skip vae decode on aborted/skipped/interrupted image generations  
  - new option: *settings -> system paths -> models*  
    can be used to set custom base path for *all* models (previously only as cli option)  
  - remove external clone of items in `/repositories`  
  - **Interrogator** module has been removed from `extensions-builtin`  
    and fully implemented (and improved) natively  
- **UI**  
  - UI tweaks for default themes  
  - UI switch core font in default theme to **noto-sans**  
    previously default font was simply *system-ui*, but it lead to too much variations between browsers and platforms  
  - UI tweaks for mobile devices, thanks @iDeNoh  
  - updated **Context menu**  
    right-click on any button in action menu (e.g. generate button)  
- **Extra networks**  
  - sort by name, size, date, etc.  
  - switch between *gallery* and *list* views  
  - add tags from user metadata (in addition to tags in model metadata) for **lora**  
  - added **Reference** models for diffusers backend  
  - faster enumeration of all networks on server startup  
- **Packages**
  - updated `diffusers` to 0.22.0, `transformers` to 4.34.1  
  - update **openvino**, thanks @disty0  
  - update **directml**, @lshqqytiger  
- **Compute**  
  - **OpenVINO**:  
    - updated to mainstream `torch` *2.1.0*  
    - support for **ESRGAN** upscalers  
- **Fixes**  
  - fix **freeu** for backend original and add it to xyz grid  
  - fix loading diffuser models in huggingface format from non-standard location  
  - fix default styles looking in wrong location  
  - fix missing upscaler folder on initial startup  
  - fix handling of relative path for models  
  - fix simple live preview device mismatch  
  - fix batch img2img  
  - fix diffusers samplers: dpm++ 2m, dpm++ 1s, deis  
  - fix new style filename template  
  - fix image name template using model name  
  - fix image name sequence  
  - fix model path using relative path  
  - fix safari/webkit layour, thanks @eadnams22
  - fix `torch-rocm` and `tensorflow-rocm` version detection, thanks @xangelix  
  - fix **chainner** upscalers color clipping  
  - fix for base+refiner workflow in diffusers mode: number of steps, diffuser pipe mode  
  - fix for prompt encoder with refiner in diffusers mode  
  - fix prompts-from-file saving incorrect metadata  
  - fix add/remove extra networks to prompt
  - fix before-hires step  
  - fix diffusers switch from invalid model  
  - force second requirements check on startup  
  - remove **lyco**, multiple_tqdm  
  - enhance extension compatibility for extensions directly importing codeformers  
  - enhance extension compatibility for extensions directly accessing processing params  
  - **css** fixes  
  - clearly mark external themes in ui  
  - update `typing-extensions`  

## Update for 2023-10-17

This is a major release, with many changes and new functionality...  

Changelog is massive, but do read through or youll be missing on some very cool new functionality  
or even free speedups and quality improvements (regardless of which workflows youre using)!  

Note that for this release its recommended to perform a clean install (e.g. fresh `git clone`)  
Upgrades are still possible and supported, but clean install is recommended for best experience  

- **UI**  
  - added **change log** to UI  
    see *System -> Changelog*  
  - converted submenus from checkboxes to accordion elements  
    any ui state including state of open/closed menus can be saved as default!  
    see *System -> User interface -> Set menu states*  
  - new built-in theme **invoked**  
    thanks @BinaryQuantumSoul  
  - add **compact view** option in settings -> user interface  
  - small visual indicator bottom right of page showing internal server job state  
- **Extra networks**:  
  - **Details**  
    - new details interface to view and save data about extra networks  
      main ui now has a single button on each en to trigger details view  
    - details view includes model/lora metadata parser!  
    - details view includes civitai model metadata!  
  - **Metadata**:  
    - you can scan [civitai](https://civitai.com/)  
      for missing metadata and previews directly from extra networks  
      simply click on button in top-right corner of extra networks page  
  - **Styles**  
    - save/apply icons moved to extra networks  
    - can be edited in details view  
    - support for single or multiple styles per json  
    - support for embedded previews  
    - large database of art styles included by default  
      can be disabled in *settings -> extra networks -> show built-in*  
    - styles can also be used in a prompt directly: `<style:style_name>`  
      if style if an exact match, it will be used  
      otherwise it will rotate between styles that match the start of the name  
      that way you can use different styles as wildcards when processing batches  
    - styles can have **extra** fields, not just prompt and negative prompt  
      for example: *"Extra: sampler: Euler a, width: 480, height: 640, steps: 30, cfg scale: 10, clip skip: 2"*
  - **VAE**  
    - VAEs are now also listed as part of extra networks  
    - Image preview methods have been redesigned: simple, approximate, taesd, full  
      please set desired preview method in settings  
    - both original and diffusers backend now support "full quality" setting  
      if you desired model or platform does not support FP16 and/or you have a low-end hardware and cannot use FP32  
      you can disable "full quality" in advanced params and it will likely reduce decode errors (infamous black images)  
  - **LoRA**  
    - LoRAs are now automatically filtered based on compatibility with currently loaded model  
      note that if lora type cannot be auto-determined, it will be left in the list  
  - **Refiner**  
    - you can load model from extra networks as base model or as refiner  
      simply select button in top-right of models page  
  - **General**  
    - faster search, ability to show/hide/sort networks  
    - refactored subfolder handling  
      *note*: this will trigger model hash recalculation on first model use  
- **Diffusers**:  
  - better pipeline **auto-detect** when loading from safetensors  
  - **SDXL Inpaint**  
    - although any model can be used for inpainiting, there is a case to be made for  
      dedicated inpainting models as they are tuned to inpaint and not generate  
    - model can be used as base model for **img2img** or refiner model for **txt2img**  
      To download go to *Models -> Huggingface*:  
      - `diffusers/stable-diffusion-xl-1.0-inpainting-0.1` *(6.7GB)*  
  - **SDXL Instruct-Pix2Pix**  
    - model can be used as base model for **img2img** or refiner model for **txt2img**  
      this model is massive and requires a lot of resources!  
      to download go to *Models -> Huggingface*:  
      - `diffusers/sdxl-instructpix2pix-768` *(11.9GB)*  
  - **SD Latent Upscale**  
    - you can use *SD Latent Upscale* models as **refiner models**  
      this is a bit experimental, but it works quite well!  
      to download go to *Models -> Huggingface*:  
      - `stabilityai/sd-x2-latent-upscaler` *(2.2GB)*  
      - `stabilityai/stable-diffusion-x4-upscaler` *(1.7GB)*  
  - better **Prompt attention**  
    should better handle more complex prompts  
    for sdxl, choose which part of prompt goes to second text encoder - just add `TE2:` separator in the prompt  
    for hires and refiner, second pass prompt is used if present, otherwise primary prompt is used  
    new option in *settings -> diffusers -> sdxl pooled embeds*  
    thanks @AI-Casanova  
  - better **Hires** support for SD and SDXL  
  - better **TI embeddings** support for SD and SDXL  
    faster loading, wider compatibility and support for embeddings with multiple vectors  
    information about used embedding is now also added to image metadata  
    thanks @AI-Casanova  
  - better **Lora** handling  
    thanks @AI-Casanova  
  - better **SDXL preview** quality (approx method)  
    thanks @BlueAmulet
  - new setting: *settings -> diffusers -> force inpaint*  
    as some models behave better when in *inpaint* mode even for normal *img2img* tasks  
- **Upscalers**:
  - pretty much a rewrite and tons of new upscalers - built-in list is now at **42**  
  - fix long outstanding memory leak in legacy code, amazing this went undetected for so long  
  - more high quality upscalers available by default  
    **SwinIR** (2), **ESRGAN** (12), **RealESRGAN** (6), **SCUNet** (2)  
  - if that is not enough, there is new **chaiNNer** integration:  
    adds 15 more upscalers from different families out-of-the-box:  
    **HAT** (6), **RealHAT** (2), **DAT** (1), **RRDBNet** (1), **SPSRNet** (1), **SRFormer** (2), **SwiftSR** (2)  
    and yes, you can download and add your own, just place them in `models/chaiNNer`  
  - two additional latent upscalers based on SD upscale models when using Diffusers backend  
    **SD Upscale 2x**, **SD Upscale 4x***  
    note: Recommended usage for *SD Upscale* is by using second pass instead of upscaler  
    as it allows for tuning of prompt, seed, sampler settings which are used to guide upscaler  
  - upscalers are available in **xyz grid**  
  - simplified *settings->postprocessing->upscalers*  
    e.g. all upsamplers share same settings for tiling  
  - allow upscale-only as part of **txt2img** and **img2img** workflows  
    simply set *denoising strength* to 0 so hires does not get triggered  
  - unified init/download/execute/progress code  
  - easier installation  
- **Samplers**:  
  - moved ui options to submenu  
  - default list for new installs is now all samplers, list can be modified in settings  
  - simplified samplers configuration in settings  
    plus added few new ones like sigma min/max which can highly impact sampler behavior  
  - note that list of samplers is now *different* since keeping a flat-list of all possible  
    combinations results in 50+ samplers which is not practical  
    items such as algorithm (e.g. karras) is actually a sampler option, not a sampler itself  
- **CivitAI**:
  - civitai model download is now multithreaded and resumable  
    meaning that you can download multiple models in parallel  
    as well as resume aborted/incomplete downloads  
  - civitai integration in *models -> civitai* can now find most  
    previews AND metadata for most models (checkpoints, loras, embeddings)  
    metadata is now parsed and saved in *[model].json*  
    typical hit rate is >95% for models, loras and embeddings  
  - description from parsed model metadata is used as model description if there is no manual  
    description file present in format of *[model].txt*  
  - to enable search for models, make sure all models have set hash values  
    *Models -> Valida -> Calculate hashes*  
- **LoRA**
  - new unified LoRA handler for all LoRA types (lora, lyco, loha, lokr, locon, ia3, etc.)  
    applies to both original and diffusers backend  
    thanks @AI-Casanova for diffusers port  
  - for *backend:original*, separate lyco handler has been removed  
- **Compute**  
  - **CUDA**:  
    - default updated to `torch` *2.1.0* with cuda *12.1*  
    - testing moved to `torch` *2.2.0-dev/cu122*  
    - check out *generate context menu -> show nvml* for live gpu stats (memory, power, temp, clock, etc.)
  - **Intel Arc/IPEX**:  
    - tons of optimizations, built-in binary wheels for Windows  
      i have to say, intel arc/ipex is getting to be quite a player, especially with openvino  
      thanks @Disty0 @Nuullll  
  - **AMD ROCm**:  
    - updated installer to support detect `ROCm` *5.4/5.5/5.6/5.7*  
    - support for `torch-rocm-5.7`
  - **xFormers**:
    - default updated to *0.0.23*  
    - note that latest xformers are still not compatible with cuda 12.1  
      recommended to use torch 2.1.0 with cuda 11.8  
      if you attempt to use xformers with cuda 12.1, it will force a full xformers rebuild on install  
      which can take a very long time and may/may-not work  
    - added cmd param `--use-xformers` to force usage of exformers  
  - **GC**:  
    - custom garbage collect threshold to reduce vram memory usage, thanks @Disty0  
      see *settings -> compute -> gc*  
- **Inference**  
  - new section in **settings**  
    - [HyperTile](https://github.com/tfernd/HyperTile): new!  
      available for *diffusers* and *original* backends  
      massive (up to 2x) speed-up your generations for free :)  
      *note: hypertile is not compatible with any extension that modifies processing parameters such as resolution*  
      thanks @tfernd
    - [Free-U](https://github.com/ChenyangSi/FreeU): new!  
      available for *diffusers* and *original* backends  
      improve generations quality at no cost (other than finding params that work for you)  
      *note: temporarily disabled for diffusers pending release of diffusers==0.22*  
      thanks @ljleb  
    - [Token Merging](https://github.com/dbolya/tomesd): not new, but updated  
      available for *diffusers* and *original* backends  
      speed-up your generations by merging redundant tokens  
      speed up will depend on how aggressive you want to be with token merging  
    - **Batch mode**  
      new option *settings -> inference -> batch mode*  
      when using img2img process batch, optionally process multiple images in batch in parallel  
      thanks @Symbiomatrix
- **NSFW Detection/Censor**  
  - install extension: [NudeNet](https://github.com/vladmandic/sd-extension-nudenet)  
    body part detection, image metadata, advanced censoring, etc...  
    works for *text*, *image* and *process* workflows  
    more in the extension notes  
- **Extensions**
  - automatic discovery of new extensions on github  
    no more waiting for them to appear in index!
  - new framework for extension validation  
    extensions ui now shows actual status of extensions for reviewed extensions  
    if you want to contribute/flag/update extension status, reach out on github or discord  
  - better overall compatibility with A1111 extensions (up to a point)  
  - [MultiDiffusion](https://github.com/pkuliyi2015/multidiffusion-upscaler-for-automatic1111)  
    has been removed from list of built-in extensions  
    you can still install it manually if desired  
  - [LyCORIS]<https://github.com/KohakuBlueleaf/a1111-sd-webui-lycoris>  
    has been removed from list of built-in extensions  
    it is considered obsolete given that all functionality is now built-in  
- **General**  
  - **Startup**  
    - all main CLI parameters can now be set as environment variable as well  
      for example `--data-dir <path>` can be specified as `SD_DATADIR=<path>` before starting SD.Next  
  - **XYZ Grid**
    - more flexibility to use selection or strings  
  - **Logging**  
    - get browser session info in server log  
    - allow custom log file destination  
      see `webui --log`  
    - when running with `--debug` flag, log is force-rotated  
      so each `sdnext.log.*` represents exactly one server run  
    - internal server job state tracking  
  - **Launcher**  
    - new `webui.ps1` powershell launcher for windows (old `webui.bat` is still valid)  
      thanks @em411  
  - **API**
    - add end-to-end example how to use API: `cli/simple-txt2img.js`  
      covers txt2img, upscale, hires, refiner  
  - **train.py**
    - wrapper script around built-in **kohyas lora** training script  
      see `cli/train.py --help`  
      new support for sd and sdxl, thanks @evshiron  
      new support for full offline mode (without sdnext server running)  
- **Themes**
  - all built-in themes are fully supported:  
    - *black-teal (default), light-teal, black-orange, invoked, amethyst-nightfall, midnight-barbie*  
  - if youre using any **gradio default** themes or a **3rd party** theme or  that are not optimized for SD.Next, you may experience issues  
    default minimal style has been updated for compatibility, but actual styling is completely outside of SD.Next control  

## Update for 2023-09-13

Started as a mostly a service release with quite a few fixes, but then...  
Major changes how **hires** works as well as support for a very interesting new model [Wuerstchen](https://huggingface.co/blog/wuertschen)  

- tons of fixes  
- changes to **hires**  
  - enable non-latent upscale modes (standard upscalers)  
  - when using latent upscale, hires pass is run automatically  
  - when using non-latent upscalers, hires pass is skipped by default  
    enabled using **force hires** option in ui  
    hires was not designed to work with standard upscalers, but i understand this is a common workflow  
  - when using refiner, upscale/hires runs before refiner pass  
  - second pass can now also utilize full/quick vae quality  
  - note that when combining non-latent upscale, hires and refiner output quality is maximum,  
    but operations are really resource intensive as it includes: *base->decode->upscale->encode->hires->refine*
  - all combinations of: decode full/quick + upscale none/latent/non-latent + hires on/off + refiner on/off  
    should be supported, but given the number of combinations, issues are possible  
  - all operations are captured in image metadata
- diffusers:
  - allow loading of sd/sdxl models from safetensors without online connectivity
  - support for new model: [wuerstchen](https://huggingface.co/warp-ai/wuerstchen)  
    its a high-resolution model (1024px+) thats ~40% faster than sd-xl with a bit lower resource requirements  
    go to *models -> huggingface -> search "warp-ai/wuerstchen" -> download*  
    its nearly 12gb in size, so be patient :)
- minor re-layout of the main ui  
- updated **ui hints**  
- updated **models -> civitai**  
  - search and download loras  
  - find previews for already downloaded models or loras  
- new option **inference mode**  
  - default is standard `torch.no_grad`  
    new option is `torch.inference_only` which is slightly faster and uses less vram, but only works on some gpus  
- new cmdline param `--no-metadata`  
  skips reading metadata from models that are not already cached  
- updated **gradio**  
- **styles** support for subfolders  
- **css** optimizations
- clean-up **logging**  
  - capture system info in startup log  
  - better diagnostic output  
  - capture extension output  
  - capture ldm output  
  - cleaner server restart  
  - custom exception handling

## Update for 2023-09-06

One week later, another large update!

- system:  
  - full **python 3.11** support  
    note that changing python version does require reinstall  
    and if youre already on python 3.10, really no need to upgrade  
- themes:  
  - new default theme: **black-teal**  
  - new light theme: **light-teal**  
  - new additional theme: **midnight-barbie**, thanks @nyxia  
- extra networks:  
  - support for **tags**  
    show tags on hover, search by tag, list tags, add to prompt, etc.  
  - **styles** are now also listed as part of extra networks  
    existing `styles.csv` is converted upon startup to individual styles inside `models/style`  
    this is stage one of new styles functionality  
    old styles interface is still available, but will be removed in future  
  - cache file lists for much faster startup  
    speedups are 50+% for large number of extra networks  
  - ui refresh button now refreshes selected page, not all pages  
  - simplified handling of **descriptions**  
    now shows on-mouse-over without the need for user interaction  
  - **metadata** and **info** buttons only show if there is actual content  
- diffusers:  
  - add full support for **textual inversions** (embeddings)  
    this applies to both sd15 and sdxl  
    thanks @ai-casanova for porting compel/sdxl code  
  - mix&match **base** and **refiner** models (*experimental*):  
    most of those are "because why not" and can result in corrupt images, but some are actually useful  
    also note that if youre not using actual refiner model, you need to bump refiner steps  
    as normal models are not designed to work with low step count  
    and if youre having issues, try setting prompt parser to "fixed attention" as majority of problems  
    are due to token mismatches when using prompt attention  
    - any sd15 + any sd15  
    - any sd15 + sdxl-refiner  
    - any sdxl-base + sdxl-refiner  
    - any sdxl-base + any sd15  
    - any sdxl-base + any sdxl-base  
  - ability to **interrupt** (stop/skip) model generate  
  - added **aesthetics score** setting (for sdxl)  
    used to automatically guide unet towards higher pleasing images  
    highly recommended for simple prompts  
  - added **force zeros** setting  
    create zero-tensor for prompt if prompt is empty (positive or negative)  
- general:  
  - `rembg` remove backgrounds support for **is-net** model  
  - **settings** now show markers for all items set to non-default values  
  - **metadata** refactored how/what/when metadata is added to images  
    should result in much cleaner and more complete metadata  
  - pre-create all system folders on startup  
  - handle model load errors gracefully  
  - improved vram reporting in ui  
  - improved script profiling (when running in debug mode)  

## Update for 2023-08-30

Time for a quite a large update that has been leaking bit-by-bit over the past week or so...  
*Note*: due to large changes, it is recommended to reset (delete) your `ui-config.json`  

- diffusers:  
  - support for **distilled** sd models  
    just go to models/huggingface and download a model, for example:  
    `segmind/tiny-sd`, `segmind/small-sd`, `segmind/portrait-finetuned`  
    those are lower quality, but extremely small and fast  
    up to 50% faster than sd 1.5 and execute in as little as 2.1gb of vram  
- general:  
  - redesigned **settings**  
    - new layout with separated sections:  
      *settings, ui config, licenses, system info, benchmark, models*  
    - **system info** tab is now part of settings  
      when running outside of sdnext, system info is shown in main ui  
    - all system and image paths are now relative by default  
    - add settings validation when performing load/save  
    - settings tab in ui now shows settings that are changed from default values  
    - settings tab switch to compact view  
  - update **gradio** major version  
    this may result in some smaller layout changes since its a major version change  
    however, browser page load is now much faster  
  - optimizations:
    - optimize model hashing  
    - add cli param `--skip-all` that skips all installer checks  
      use at personal discretion, but it can be useful for bulk deployments  
    - add model **precompile** option (when model compile is enabled)  
    - **extra network** folder info caching  
      results in much faster startup when you have large number of extra networks  
    - faster **xyz grid** switching  
      especially when using different checkpoints  
  - update **second pass** options for clarity
  - models:
    - civitai download missing model previews
  - add **openvino** (experimental) cpu optimized model compile and inference  
    enable with `--use-openvino`  
    thanks @disty0  
  - enable batch **img2img** scale-by workflows  
    now you can batch process with rescaling based on each individual original image size  
  - fixes:
    - fix extra networks previews  
    - css fixes  
    - improved extensions compatibility (e.g. *sd-cn-animation*)  
    - allow changing **vae** on-the-fly for both original and diffusers backend

## Update for 2023-08-20

Another release thats been baking in dev branch for a while...

- general:
  - caching of extra network information to enable much faster create/refresh operations  
    thanks @midcoastal
- diffusers:
  - add **hires** support (*experimental*)  
    applies to all model types that support img2img, including **sd** and **sd-xl**  
    also supports all hires upscaler types as well as standard params like steps and denoising strength  
    when used with **sd-xl**, it can be used with or without refiner loaded  
    how to enable - there are no explicit checkboxes other than second pass itself:
    - hires: upscaler is set and target resolution is not at default  
    - refiner: if refiner model is loaded  
  - images save options: *before hires*, *before refiner*
  - redo `move model to cpu` logic in settings -> diffusers to be more reliable  
    note that system defaults have also changed, so you may need to tweak to your liking  
  - update dependencies

## Update for 2023-08-17

Smaller update, but with some breaking changes (to prepare for future larger functionality)...

- general:
  - update all metadata saved with images  
    see <https://github.com/vladmandic/automatic/wiki/Metadata> for details  
  - improved **amd** installer with support for **navi 2x & 3x** and **rocm 5.4/5.5/5.6**  
    thanks @evshiron  
  - fix **img2img** resizing (applies to *original, diffusers, hires*)  
  - config change: main `config.json` no longer contains entire configuration  
    but only differences from defaults (similar to recent change performed to `ui-config.json`)  
- diffusers:
  - enable **batch img2img** workflows  
- original:  
  - new samplers: **dpm++ 3M sde** (standard and karras variations)  
    enable in *settings -> samplers -> show samplers*
  - expose always/never discard penultimate sigma  
    enable in *settings -> samplers*  

## Update for 2023-08-11

This is a big one thats been cooking in `dev` for a while now, but finally ready for release...

- diffusers:
  - **pipeline autodetect**
    if pipeline is set to autodetect (default for new installs), app will try to autodetect pipeline based on selected model  
    this should reduce user errors such as loading **sd-xl** model when **sd** pipeline is selected  
  - **quick vae decode** as alternative to full vae decode which is very resource intensive  
    quick decode is based on `taesd` and produces lower quality, but its great for tests or grids as it runs much faster and uses far less vram  
    disabled by default, selectable in *txt2img/img2img -> advanced -> full quality*  
  - **prompt attention** for sd and sd-xl  
    supports both `full parser` and native `compel`  
    thanks @ai-casanova  
  - advanced **lora load/apply** methods  
    in addition to standard lora loading that was recently added to sd-xl using diffusers, now we have  
    - **sequential apply** (load & apply multiple loras in sequential manner) and  
    - **merge and apply** (load multiple loras and merge before applying to model)  
    see *settings -> diffusers -> lora methods*  
    thanks @hameerabbasi and @ai-casanova  
  - **sd-xl vae** from safetensors now applies correct config  
    result is that 3rd party vaes can be used without washed out colors  
  - options for optimized memory handling for lower memory usage  
    see *settings -> diffusers*
- general:
  - new **civitai model search and download**  
    native support for civitai, integrated into ui as *models -> civitai*  
  - updated requirements  
    this time its a bigger change so upgrade may take longer to install new requirements
  - improved **extra networks** performance with large number of networks

## Update for 2023-08-05

Another minor update, but it unlocks some cool new items...

- diffusers:
  - vaesd live preview (sd and sd-xl)  
  - fix inpainting (sd and sd-xl)  
- general:
  - new torch 2.0 with ipex (intel arc)  
  - additional callbacks for extensions  
    enables latest comfyui extension  

## Update for 2023-07-30

Smaller release, but IMO worth a post...

- diffusers:
  - sd-xl loras are now supported!
  - memory optimizations: Enhanced sequential CPU offloading, model CPU offload, FP16 VAE
    - significant impact if running SD-XL (for example, but applies to any model) with only 8GB VRAM
  - update packages
- minor bugfixes

## Update for 2023-07-26

This is a big one, new models, new diffusers, new features and updated UI...

First, **SD-XL 1.0** is released and yes, SD.Next supports it out of the box!

- [SD-XL Base](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/sd_xl_base_1.0.safetensors)
- [SD-XL Refiner](https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0/blob/main/sd_xl_refiner_1.0.safetensors)

Also fresh is new **Kandinsky 2.2** model that does look quite nice:

- [Kandinsky Decoder](https://huggingface.co/kandinsky-community/kandinsky-2-2-decoder)
- [Kandinsky Prior](https://huggingface.co/kandinsky-community/kandinsky-2-2-prior)

Actual changelog is:

- general:
  - new loading screens and artwork
  - major ui simplification for both txt2img and img2img  
    nothing is removed, but you can show/hide individual sections  
    default is very simple interface, but you can enable any sections and save it as default in settings  
  - themes: add additional built-in theme, `amethyst-nightfall`
  - extra networks: add add/remove tags to prompt (e.g. lora activation keywords)
  - extensions: fix couple of compatibility items
  - firefox compatibility improvements
  - minor image viewer improvements
  - add backend and operation info to metadata

- diffusers:
  - were out of experimental phase and diffusers backend is considered stable  
  - sd-xl: support for **sd-xl 1.0** official model
  - sd-xl: loading vae now applies to both base and refiner and saves a bit of vram  
  - sd-xl: denoising_start/denoising_end
  - sd-xl: enable dual prompts  
    dual prompt is used if set regardless if refiner is enabled/loaded  
    if refiner is loaded & enabled, refiner prompt will also be used for refiner pass  
    - primary prompt goes to [OpenAI CLIP-ViT/L-14](https://huggingface.co/openai/clip-vit-large-patch14)
    - refiner prompt goes to [OpenCLIP-ViT/bigG-14](https://huggingface.co/laion/CLIP-ViT-bigG-14-laion2B-39B-b160k)
  - **kandinsky 2.2** support  
    note: kandinsky model must be downloaded using model downloader, not as safetensors due to specific model format  
  - refiner: fix batch processing
  - vae: enable loading of pure-safetensors vae files without config  
    also enable *automatic* selection to work with diffusers  
  - sd-xl: initial lora support  
    right now this applies to official lora released by **stability-ai**, support for **kohyas** lora is expected soon  
  - implement img2img and inpainting (experimental)  
    actual support and quality depends on model  
    it works as expected for sd 1.5, but not so much for sd-xl for now  
  - implement limited stop/interrupt for diffusers
    works between stages, not within steps  
  - add option to save image before refiner pass  
  - option to set vae upcast in settings  
  - enable fp16 vae decode when using optimized vae  
    this pretty much doubles performance of decode step (delay after generate is done)  

- original
  - fix hires secondary sampler  
    this now fully obsoletes `fallback_sampler` and `force_hr_sampler_name`  


## Update for 2023-07-18

While were waiting for official SD-XL release, heres another update with some fixes and enhancements...

- **global**
  - image save: option to add invisible image watermark to all your generated images  
    disabled by default, can be enabled in settings -> image options  
    watermark information will be shown when loading image such as in process image tab  
    also additional cli utility `/cli/image-watermark.py` to read/write/strip watermarks from images  
  - batch processing: fix metadata saving, also allow to drag&drop images for batch processing  
  - ui configuration: you can modify all ui default values from settings as usual,  
    but only values that are non-default will be written to `ui-config.json`  
  - startup: add cmd flag to skip all `torch` checks  
  - startup: force requirements check on each server start  
    there are too many misbehaving extensions that change system requirements  
  - internal: safe handling of all config file read/write operations  
    this allows sdnext to run in fully shared environments and prevents any possible configuration corruptions  
- **diffusers**:
  - sd-xl: remove image watermarks autocreated by 0.9 model  
  - vae: enable loading of external vae, documented in diffusers wiki  
    and mix&match continues, you can even use sd-xl vae with sd 1.5 models!  
  - samplers: add concept of *default* sampler to avoid needing to tweak settings for primary or second pass  
    note that sampler details will be printed in log when running in debug level  
  - samplers: allow overriding of sampler beta values in settings  
  - refiner: fix refiner applying only to first image in batch  
  - refiner: allow using direct latents or processed output in refiner  
  - model: basic support for one more model: [UniDiffuser](https://github.com/thu-ml/unidiffuser)  
    download using model downloader: `thu-ml/unidiffuser-v1`  
    and set resolution to 512x512  

## Update for 2023-07-14

Trying to unify settings for both original and diffusers backend without introducing duplicates...

- renamed **hires fix** to **second pass**  
  as that is what it actually is, name hires fix is misleading to start with  
- actual **hires fix** and **refiner** are now options inside **second pass** section  
- obsoleted settings -> sampler -> **force_hr_sampler_name**  
  it is now part of **second pass** options and it works the same for both original and diffusers backend  
  which means you can use different scheduler settings for txt2img and hires if you want  
- sd-xl refiner will run if its loaded and if second pass is enabled  
  so you can quickly enable/disable refiner by simply enabling/disabling second pass  
- you can mix&match **model** and **refiner**  
  for example, you can generate image using sd 1.5 and still use sd-xl refiner as second pass  
- reorganized settings -> samplers to show which section refers to which backend  
- added diffusers **lmsd** sampler  

## Update for 2023-07-13

Another big one, but now improvements to both **diffusers** and **original** backends as well plus ability to dynamically switch between them!

- swich backend between diffusers and original on-the-fly
  - you can still use `--backend <backend>` and now that only means in which mode app will start,
    but you can change it anytime in ui settings
  - for example, you can even do things like generate image using sd-xl,  
    then switch to original backend and perform inpaint using a different model  
- diffusers backend:
  - separate ui settings for refiner pass with sd-xl  
    you can specify: prompt, negative prompt, steps, denoise start  
  - fix loading from pure safetensors files  
    now you can load sd-xl from safetensors file or from huggingface folder format  
  - fix kandinsky model (2.1 working, 2.2 was just released and will be soon)  
- original backend:
  - improvements to vae/unet handling as well as cross-optimization heads  
    in non-technical terms, this means lower memory usage and higher performance  
    and you should be able to generate higher resolution images without any other changes
- other:
  - major refactoring of the javascript code  
    includes fixes for text selections and navigation  
  - system info tab now reports on nvidia driver version as well  
  - minor fixes in extra-networks  
  - installer handles origin changes for submodules  

big thanks to @huggingface team for great communication, support and fixing all the reported issues asap!


## Update for 2023-07-10

Service release with some fixes and enhancements:

- diffusers:
  - option to move base and/or refiner model to cpu to free up vram  
  - model downloader options to specify model variant / revision / mirror  
  - now you can download `fp16` variant directly for reduced memory footprint  
  - basic **img2img** workflow (*sketch* and *inpaint* are not supported yet)  
    note that **sd-xl** img2img workflows are architecturaly different so it will take longer to implement  
  - updated hints for settings  
- extra networks:
  - fix corrupt display on refesh when new extra network type found  
  - additional ui tweaks  
  - generate thumbnails from previews only if preview resolution is above 1k
- image viewer:
  - fixes for non-chromium browsers and mobile users and add option to download image  
  - option to download image directly from image viewer
- general
  - fix startup issue with incorrect config  
  - installer should always check requirements on upgrades

## Update for 2023-07-08

This is a massive update which has been baking in a `dev` branch for a while now

- merge experimental diffusers support  

*TL;DR*: Yes, you can run **SD-XL** model in **SD.Next** now  
For details, see Wiki page: [Diffusers](https://github.com/vladmandic/automatic/wiki/Diffusers)  
Note this is still experimental, so please follow Wiki  
Additional enhancements and fixes will be provided over the next few days  
*Thanks to @huggingface team for making this possible and our internal @team for all the early testing*

Release also contains number of smaller updates:

- add pan & zoom controls (touch and mouse) to image viewer (lightbox)  
- cache extra networks between tabs  
  this should result in neat 2x speedup on building extra networks  
- add settings -> extra networks -> do not automatically build extra network pages  
  speeds up app start if you have a lot of extra networks and you want to build them manually when needed  
- extra network ui tweaks  

## Update for 2023-07-01

Small quality-of-life updates and bugfixes:

- add option to disallow usage of ckpt checkpoints
- change lora and lyco dir without server restart
- additional filename template fields: `uuid`, `seq`, `image_hash`  
- image toolbar is now shown only when image is present
- image `Zip` button gone and its not optional setting that applies to standard `Save` button
- folder `Show` button is present only when working on localhost,  
  otherwise its replaced with `Copy` that places image URLs on clipboard so they can be used in other apps

## Update for 2023-06-30

A bit bigger update this time, but contained to specific areas...

- change in behavior  
  extensions no longer auto-update on startup  
  using `--upgrade` flag upgrades core app as well as all submodules and extensions  
- **live server log monitoring** in ui  
  configurable via settings -> live preview  
- new **extra networks interface**  
  *note: if youre using a 3rd party ui extension for extra networks, it will likely need to be updated to work with new interface*
  - display in front of main ui, inline with main ui or as a sidebar  
  - lazy load thumbnails  
    drastically reduces load times for large number of extra networks  
  - auto-create thumbnails from preview images in extra networks in a background thread  
    significant load time saving on subsequent restarts  
  - support for info files in addition to description files  
  - support for variable aspect-ratio thumbnails  
  - new folder view  
- **extensions sort** by trending  
- add requirements check for training  

## Update for 2023-06-26

- new training tab interface  
  - redesigned preprocess, train embedding, train hypernetwork  
- new models tab interface  
  - new model convert functionality, thanks @akegarasu  
  - new model verify functionality  
- lot of ipex specific fixes/optimizations, thanks @disty0  

## Update for 2023-06-20

This one is less relevant for standard users, but pretty major if youre running an actual server  
But even if not, it still includes bunch of cumulative fixes since last release - and going by number of new issues, this is probably the most stable release so far...
(next one is not going to be as stable, but it will be fun :) )

- minor improvements to extra networks ui  
- more hints/tooltips integrated into ui  
- new dedicated api server  
  - but highly promising for high throughput server  
- improve server logging and monitoring with  
  - server log file rotation  
  - ring buffer with api endpoint `/sdapi/v1/log`  
  - real-time status and load endpoint `/sdapi/v1/system-info/status`

## Update for 2023-06-14

Second stage of a jumbo merge from upstream plus few minor changes...

- simplify token merging  
- reorganize some settings  
- all updates from upstream: **A1111** v1.3.2 [df004be] *(latest release)*  
  pretty much nothing major that i havent released in previous versions, but its still a long list of tiny changes  
  - skipped/did-not-port:  
    add separate hires prompt: unnecessarily complicated and spread over large number of commits due to many regressions  
    allow external scripts to add cross-optimization methods: dangerous and i dont see a use case for it so far  
    load extension info in threads: unnecessary as other optimizations ive already put place perform equally good  
  - broken/reverted:  
    sub-quadratic optimization changes  

## Update for 2023-06-13

Just a day later and one *bigger update*...
Both some **new functionality** as well as **massive merges** from upstream  

- new cache for models/lora/lyco metadata: `metadata.json`  
  drastically reduces disk access on app startup  
- allow saving/resetting of **ui default values**  
  settings -> ui defaults
- ability to run server without loaded model  
  default is to auto-load model on startup, can be changed in settings -> stable diffusion  
  if disabled, model will be loaded on first request, e.g. when you click generate  
  useful when you want to start server to perform other tasks like upscaling which do not rely on model  
- updated `accelerate` and `xformers`
- huge nubmer of changes ported from **A1111** upstream  
  this was a massive merge, hopefully this does not cause any regressions  
  and still a bit more pending...

## Update for 2023-06-12

- updated ui labels and hints to improve clarity and provide some extra info  
  this is 1st stage of the process, more to come...  
  if you want to join the effort, see <https://github.com/vladmandic/automatic/discussions/1246>
- new localization and hints engine  
  how hints are displayed can be selected in settings -> ui  
- reworked **installer** sequence  
  as some extensions are loading packages directly from their preload sequence  
  which was preventing some optimizations to take effect  
- updated **settings** tab functionality, thanks @gegell  
  with real-time monitor for all new and/or updated settings  
- **launcher** will now warn if application owned files are modified  
  you are free to add any user files, but do not modify app files unless youre sure in what youre doing  
- add more profiling for scripts/extensions so you can see what takes time  
  this applies both to initial load as well as execution  
- experimental `sd_model_dict` setting which allows you to load model dictionary  
  from one model and apply weights from another model specified in `sd_model_checkpoint`  
  results? who am i to judge :)


## Update for 2023-06-05

Few new features and extra handling for broken extensions  
that caused my phone to go crazy with notifications over the weekend...

- added extra networks to **xyz grid** options  
  now you can have more fun with all your embeddings and loras :)  
- new **vae decode** method to help with larger batch sizes, thanks @bigdog  
- new setting -> lora -> **use lycoris to handle all lora types**  
  this is still experimental, but the goal is to obsolete old built-in lora module  
  as it doesnt understand many new loras and built-in lyco module can handle it all  
- somewhat optimize browser page loading  
  still slower than id want, but gradio is pretty bad at this  
- profiling of scripts/extensions callbacks  
  you can now see how much or pre/post processing is done, not just how long generate takes  
- additional exception handling so bad exception does not crash main app  
- additional background removal models  
- some work on bfloat16 which nobody really should be using, but why not 🙂


## Update for 2023-06-02

Some quality-of-life improvements while working on larger stuff in the background...

- redesign action box to be uniform across all themes  
- add **pause** option next to stop/skip  
- redesigned progress bar  
- add new built-in extension: **agent-scheduler**  
  very elegant way to getting full queueing capabilities, thank @artventurdev  
- enable more image formats  
  note: not all are understood by browser so previews and images may appear as blank  
  unless you have some browser extensions that can handle them  
  but they are saved correctly. and cant beat raw quality of 32-bit `tiff` or `psd` :)  
- change in behavior: `xformers` will be uninstalled on startup if they are not active  
  if you do have `xformers` selected as your desired cross-optimization method, then they will be used  
  reason is that a lot of libaries try to blindly import xformers even if they are not selected or not functional  

## Update for 2023-05-30

Another bigger one...And more to come in the next few days...

- new live preview mode: taesd  
  i really like this one, so its enabled as default for new installs  
- settings search feature  
- new sampler: dpm++ 2m sde  
- fully common save/zip/delete (new) options in all tabs  
  which (again) meant rework of process image tab  
- system info tab: live gpu utilization/memory graphs for nvidia gpus  
- updated controlnet interface  
- minor style changes  
- updated lora, swinir, scunet and ldsr code from upstream  
- start of merge from a1111 v1.3  

## Update for 2023-05-26

Some quality-of-life improvements...

- updated [README](https://github.com/vladmandic/automatic/blob/master/README.md)
- created [CHANGELOG](https://github.com/vladmandic/automatic/blob/master/CHANGELOG.md)  
  this will be the source for all info about new things moving forward  
  and cross-posted to [Discussions#99](https://github.com/vladmandic/automatic/discussions/99) as well as discord [announcements](https://discord.com/channels/1101998836328697867/1109953953396957286)
- optimize model loading on startup  
  this should reduce startup time significantly  
- set default cross-optimization method for each platform backend  
  applicable for new installs only  
  - `cuda` => Scaled-Dot-Product
  - `rocm` => Sub-quadratic
  - `directml` => Sub-quadratic
  - `ipex` => invokeais
  - `mps` => Doggettxs
  - `cpu` => Doggettxs
- optimize logging  
- optimize profiling  
  now includes startup profiling as well as `cuda` profiling during generate  
- minor lightbox improvements  
- bugfixes...i dont recall when was a release with at least several of those  

other than that - first stage of [Diffusers](https://github.com/huggingface/diffusers) integration is now in master branch  
i dont recommend anyone to try it (and dont even think reporting issues for it)  
but if anyone wants to contribute, take a look at [project page](https://github.com/users/vladmandic/projects/1/views/1)

## Update for 2023-05-23

Major internal work with perhaps not that much user-facing to show for it ;)

- update core repos: **stability-ai**, **taming-transformers**, **k-diffusion, blip**, **codeformer**  
  note: to avoid disruptions, this is applicable for new installs only
- tested with **torch 2.1**, **cuda 12.1**, **cudnn 8.9**  
  (production remains on torch2.0.1+cuda11.8+cudnn8.8)  
- fully extend support of `--data-dir`  
  allows multiple installations to share pretty much everything, not just models  
  especially useful if you want to run in a stateless container or cloud instance  
- redo api authentication  
  now api authentication will use same user/pwd (if specified) for ui and strictly enforce it using httpbasicauth  
  new authentication is also fully supported in combination with ssl for both sync and async calls  
  if you want to use api programatically, see examples in `cli/sdapi.py`  
- add dark/light theme mode toggle  
- redo some `clip-skip` functionality  
- better matching for vae vs model  
- update to `xyz grid` to allow creation of large number of images without creating grid itself  
- update `gradio` (again)  
- more prompt parser optimizations  
- better error handling when importing image settings which are not compatible with current install  
  for example, when upscaler or sampler originally used is not available  
- fixes...amazing how many issues were introduced by porting a1111 v1.20 code without adding almost no new functionality  
  next one is v1.30 (still in dev) which does bring a lot of new features  

## Update for 2023-05-17

This is a massive one due to huge number of changes,  
but hopefully it will go ok...

- new **prompt parsers**  
  select in UI -> Settings -> Stable Diffusion  
  - **Full**: my new implementation  
  - **A1111**: for backward compatibility  
  - **Compel**: as used in ComfyUI and InvokeAI (a.k.a *Temporal Weighting*)  
  - **Fixed**: for really old backward compatibility  
- monitor **extensions** install/startup and  
  log if they modify any packages/requirements  
  this is a *deep-experimental* python hack, but i think its worth it as extensions modifying requirements  
  is one of most common causes of issues
- added `--safe` command line flag mode which skips loading user extensions  
  please try to use it before opening new issue  
- reintroduce `--api-only` mode to start server without ui  
- port *all* upstream changes from [A1111](https://github.com/AUTOMATIC1111/stable-diffusion-webui)  
  up to today - commit hash `89f9faa`  

## Update for 2023-05-15

- major work on **prompt parsing**
  this can cause some differences in results compared to what youre used to, but its all about fixes & improvements
  - prompt parser was adding commas and spaces as separate words and tokens and/or prefixes
  - negative prompt weight using `[word:weight]` was ignored, it was always `0.909`
  - bracket matching was anything but correct. complex nested attention brackets are now working.
  - btw, if you run with `--debug` flag, youll now actually see parsed prompt & schedule
- updated all scripts in `/cli`  
- add option in settings to force different **latent sampler** instead of using primary only
- add **interrupt/skip** capabilities to process images

## Update for 2023-05-13

This is mostly about optimizations...

- improved `torch-directml` support  
  especially interesting for **amd** users on **windows**  where **torch+rocm** is not yet available  
  dont forget to run using `--use-directml` or default is **cpu**  
- improved compatibility with **nvidia** rtx 1xxx/2xxx series gpus  
- fully working `torch.compile` with **torch 2.0.1**  
  using `inductor` compile takes a while on first run, but does result in 5-10% performance increase  
- improved memory handling  
  for highest performance, you can also disable aggressive **gc** in settings  
- improved performance  
  especially *after* generate as image handling has been moved to separate thread  
- allow per-extension updates in extension manager  
- option to reset configuration in settings  

## Update for 2023-05-11

- brand new **extension manager**  
  this is pretty much a complete rewrite, so new issues are possible
- support for `torch` 2.0.1  
  note that if you are experiencing frequent hangs, this may be a worth a try  
- updated `gradio` to 3.29.0
- added `--reinstall` flag to force reinstall of all packages  
- auto-recover & re-attempt when `--upgrade` is requested but fails
- check for duplicate extensions  

## Update for 2023-05-08

Back online with few updates:

- bugfixes. yup, quite a lot of those  
- auto-detect some cpu/gpu capabilities on startup  
  this should reduce need to tweak and tune settings like no-half, no-half-vae, fp16 vs fp32, etc  
- configurable order of top level tabs  
- configurable order of scripts in txt2img and img2img  
  for both, see sections in ui-> settings -> user interface

## Update for 2023-05-04

Again, few days later...

- reviewed/ported **all** commits from **A1111** upstream  
  some a few are not applicable as i already have alternative implementations  
  and very few i choose not to implement (save/restore last-known-good-config is a bad hack)  
  otherwise, were fully up to date (it doesnt show on fork status as code merges were mostly manual due to conflicts)  
  but...due to sheer size of the updates, this may introduce some temporary issues  
- redesigned server restart function  
  now available and working in ui  
  actually, since server restart is now a true restart and not ui restart, it can be used much more flexibly  
- faster model load  
  plus support for slower devices via stream-load function (in ui settings)  
- better logging  
  this includes new `--debug` flag for more verbose logging when troubleshooting  

## Update for 2023-05-01

Been a bit quieter for last few days as changes were quite significant, but finally here we are...

- Updated core libraries: Gradio, Diffusers, Transformers
- Added support for **Intel ARC** GPUs via Intel OneAPI IPEX (auto-detected)
- Added support for **TorchML** (set by default when running on non-compatible GPU or on CPU)
- Enhanced support for AMD GPUs with **ROCm**
- Enhanced support for Apple **M1/M2**
- Redesigned command params: run `webui --help` for details
- Redesigned API and script processing
- Experimental support for multiple **Torch compile** options
- Improved sampler support
- Google Colab: <https://colab.research.google.com/drive/126cDNwHfifCyUpCCQF9IHpEdiXRfHrLN>  
  Maintained by <https://github.com/Linaqruf/sd-notebook-collection>
- Fixes, fixes, fixes...

To take advantage of new out-of-the-box tunings, its recommended to delete your `config.json` so new defaults are applied. its not necessary, but otherwise you may need to play with UI Settings to get the best of Intel ARC, TorchML, ROCm or Apple M1/M2.

## Update for 2023-04-27

a bit shorter list as:

- ive been busy with bugfixing  
  there are a lot of them, not going to list each here.  
  but seems like critical issues backlog is quieting down and soon i can focus on new features development.  
- ive started collaboration with couple of major projects,
  hopefully this will accelerate future development.

whats new:

- ability to view/add/edit model description shown in extra networks cards  
- add option to specify fallback sampler if primary sampler is not compatible with desired operation  
- make clip skip a local parameter  
- remove obsolete items from UI settings  
- set defaults for AMD ROCm  
  if you have issues, you may want to start with a fresh install so configuration can be created from scratch
- set defaults for Apple M1/M2  
  if you have issues, you may want to start with a fresh install so configuration can be created from scratch

## Update for 2023-04-25

- update process image -> info
- add VAE info to metadata
- update GPU utility search paths for better GPU type detection
- update git flags for wider compatibility
- update environment tuning
- update ti training defaults
- update VAE search paths
- add compatibility opts for some old extensions
- validate script args for always-on scripts  
  fixes: deforum with controlnet  

## Update for 2023-04-24

- identify race condition where generate locks up while fetching preview
- add pulldowns to x/y/z script
- add VAE rollback feature in case of NaNs
- use samples format for live preview
- add token merging
- use **Approx NN** for live preview
- create default `styles.csv`
- fix setup not installing `tensorflow` dependencies
- update default git flags to reduce number of warnings

## Update for 2023-04-23

- fix VAE dtype  
  should fix most issues with NaN or black images  
- add built-in Gradio themes  
- reduce requirements  
- more AMD specific work
- initial work on Apple platform support
- additional PR merges
- handle torch cuda crashing in setup
- fix setup race conditions
- fix ui lightbox
- mark tensorflow as optional
- add additional image name templates

## Update for 2023-04-22

- autodetect which system libs should be installed  
  this is a first pass of autoconfig for **nVidia** vs **AMD** environments  
- fix parse cmd line args from extensions  
- only install `xformers` if actually selected as desired cross-attention method
- do not attempt to use `xformers` or `sdp` if running on cpu
- merge tomesd token merging  
- merge 23 PRs pending from a1111 backlog (!!)

*expect shorter updates for the next few days as ill be partially ooo*

## Update for 2023-04-20

- full CUDA tuning section in UI Settings
- improve exif/pnginfo metadata parsing  
  it can now handle 3rd party images or images edited in external software
- optimized setup performance and logging
- improve compatibility with some 3rd party extensions
  for example handle extensions that install packages directly from github urls
- fix initial model download if no models found
- fix vae not found issues
- fix multiple git issues

note: if you previously had command line optimizations such as --no-half, those are now ignored and moved to ui settings

## Update for 2023-04-19

- fix live preview
- fix model merge
- fix handling of user-defined temp folders
- fix submit benchmark
- option to override `torch` and `xformers` installer
- separate benchmark data for system-info extension
- minor css fixes
- created initial merge backlog from pending prs on a1111 repo  
  see #258 for details

## Update for 2023-04-18

- reconnect ui to active session on browser restart  
  this is one of most frequently asked for items, finally figured it out  
  works for text and image generation, but not for process as there is no progress bar reported there to start with  
- force unload `xformers` when not used  
  improves compatibility with AMD/M1 platforms  
- add `styles.csv` to UI settings to allow customizing path  
- add `--skip-git` to cmd flags for power users that want  
  to skip all git checks and operations and perform manual updates
- add `--disable-queue` to cmd flags that disables Gradio queues (experimental)
  this forces it to use HTTP instead of WebSockets and can help on unreliable network connections  
- set scripts & extensions loading priority and allow custom priorities  
  fixes random extension issues:  
  `ScuNet` upscaler disappearing, `Additional Networks` not showing up on XYZ axis, etc.
- improve html loading order
- remove some `asserts` causing runtime errors and replace with user-friendly messages
- update README.md

## Update for 2023-04-17

- **themes** are now dynamic and discovered from list of available gradio themes on huggingface  
  its quite a list of 30+ supported themes so far  
- added option to see **theme preview** without the need to apply it or restart server
- integrated **image info** functionality into **process image** tab and removed separate **image info** tab
- more installer improvements
- fix urls
- updated github integration
- make model download as optional if no models found

## Update for 2023-04-16

- support for ui themes! to to *settings* -> *user interface* -> "ui theme*
  includes 12 predefined themes
- ability to restart server from ui
- updated requirements
- removed `styles.csv` from repo, its now fully under user control
- removed model-keyword extension as overly aggressive
- rewrite of the fastapi middleware handlers
- install bugfixes, hopefully new installer is now ok  \
  i really want to focus on features and not troubleshooting installer

## Update for 2023-04-15

- update default values
- remove `ui-config.json` from repo, its now fully under user control
- updated extensions manager
- updated locon/lycoris plugin
- enable quick launch by default
- add multidiffusion upscaler extensions
- add model keyword extension
- enable strong linting
- fix circular imports
- fix extensions updated
- fix git update issues
- update github templates

## Update for 2023-04-14

- handle duplicate extensions
- redo exception handler
- fix generate forever
- enable cmdflags compatibility
- change default css font
- fix ti previews on initial start
- enhance tracebacks
- pin transformers version to last known good version
- fix extension loader

## Update for 2023-04-12

This has been pending for a while, but finally uploaded some massive changes

- New launcher
  - `webui.bat` and `webui.sh`:  
    Platform specific wrapper scripts that starts `launch.py` in Python virtual environment  
    *Note*: Server can run without virtual environment, but it is recommended to use it  
    This is carry-over from original repo  
    **If youre unsure which launcher to use, this is the one you want**  
  - `launch.py`:  
    Main startup script  
    Can be used directly to start server in manually activated `venv` or to run it without `venv`  
  - `installer.py`:  
    Main installer, used by `launch.py`  
  - `webui.py`:  
    Main server script  
- New logger
- New exception handler
- Built-in performance profiler
- New requirements handling
- Move of most of command line flags into UI Settings
