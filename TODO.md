# TODO

## Internal

- Update: `transformers==5.0.0`
- Deploy: Create executable for SD.Next
- Deploy: Lite vs Expert mode
- Engine: [mmgp](https://github.com/deepbeepmeep/mmgp)
- Engine: [sharpfin](https://github.com/drhead/sharpfin) instead of `torchvision`
- Engine: `TensorRT` acceleration
- Feature: Auto handle scheduler `prediction_type`
- Feature: Cache models in memory
- Feature: Control tab add overrides handling
- Feature: Integrate natural language image search
  [ImageDB](https://github.com/vladmandic/imagedb)
- Feature: LoRA add OMI format support for SD35/FLUX.1
- Feature: Multi-user support
- Feature: Remote Text-Encoder support
- Feature: Settings profile manager
- Feature: Video tab add full API support
- Refactor: Unify *huggingface* and *diffusers* model folders
- Refactor: Move `nunchaku` models to refernce instead of internal decision  
- Refactor: [GGUF](https://huggingface.co/docs/diffusers/main/en/quantization/gguf)
- Refactor: move sampler options to settings to config
- Refactor: remove `CodeFormer`
- Refactor: remove `GFPGAN`
- Reimplement `llama` remover for Kanvas

## Modular

*Pending finalization of modular pipelines implementation and development of compatibility layer*

- Switch to modular pipelines
- Feature: Transformers unified cache handler
- Refactor: [Modular pipelines and guiders](https://github.com/huggingface/diffusers/issues/11915)
- [MagCache](https://github.com/huggingface/diffusers/pull/12744)
- [SmoothCache](https://github.com/huggingface/diffusers/issues/11135)
- [STG](https://github.com/huggingface/diffusers/blob/main/examples/community/README.md#spatiotemporal-skip-guidance)

## New models / Pipelines

TODO: Investigate which models are diffusers-compatible and prioritize!

### Image-Base
- [Bria FIBO](https://huggingface.co/briaai/FIBO)
- [Chroma Zeta](https://huggingface.co/lodestones/Zeta-Chroma)
- [Chroma Radiance](https://huggingface.co/lodestones/Chroma1-Radiance)
- [Liquid](https://github.com/FoundationVision/Liquid)
- [Lumina-DiMOO](https://huggingface.co/Alpha-VLLM/Lumina-DiMOO)

### Image-Edit
- [Meituan LongCat-Image-Edit-Turbo](https://huggingface.co/meituan-longcat/LongCat-Image-Edit-Turbo)
- [VIBE Image-Edit](https://huggingface.co/iitolstykh/VIBE-Image-Edit)
- [Bria FiboEdit](https://github.com/huggingface/diffusers/commit/d7a1c31f4f85bae5a9e01cdce49bd7346bd8ccd6)
- [LucyEdit](https://github.com/huggingface/diffusers/pull/12340)
- [SD3 UltraEdit](https://github.com/HaozheZhao/UltraEdit)
- [Step1X-Edit](https://github.com/stepfun-ai/Step1X-Edit)
- [OneReward](https://github.com/bytedance/OneReward)

### Video
- [OpenMOSS MOVA](https://huggingface.co/OpenMOSS-Team/MOVA-720p)
- [Wan family (Wan2.1 / Wan2.2 variants)](https://huggingface.co/Wan-AI/Wan2.2-Animate-14B)
  - example: [Wan2.1-T2V-14B-CausVid](https://huggingface.co/lightx2v/Wan2.1-T2V-14B-CausVid)
  - distill / step-distill examples: [Wan2.1-StepDistill-CfgDistill](https://huggingface.co/lightx2v/Wan2.1-T2V-14B-StepDistill-CfgDistill)
- [Krea Realtime Video](https://huggingface.co/krea/krea-realtime-video)
- [MAGI-1 (autoregressive video)](https://github.com/SandAI-org/MAGI-1)
- [MUG-V 10B (video generation)](https://huggingface.co/MUG-V/MUG-V-inference)
- [Ovi (audio/video generation)](https://github.com/character-ai/Ovi)
- [MUG-V 10B](https://huggingface.co/MUG-V/MUG-V-inference)
- [HunyuanVideo-Avatar / HunyuanCustom](https://huggingface.co/tencent/HunyuanVideo-Avatar)
- [Sana Image→Video (Sana-I2V)](https://github.com/huggingface/diffusers/pull/12634#issuecomment-3540534268)
- [Wan-2.2 S2V (diffusers PR)](https://github.com/huggingface/diffusers/pull/12258)
- [LongCat-Video](https://huggingface.co/meituan-longcat/LongCat-Video)
- [LTXVideo / LTXVideo LongMulti (diffusers PR)](https://github.com/huggingface/diffusers/pull/12614)
- [DiffSynth-Studio (ModelScope)](https://github.com/modelscope/DiffSynth-Studio)
- [Phantom (Phantom HuMo)](https://github.com/Phantom-video/Phantom)
- [CausVid-Plus / WAN-CausVid-Plus](https://github.com/goatWu/CausVid-Plus/)
- [Wan2GP (workflow/GUI for Wan)](https://github.com/deepbeepmeep/Wan2GP)

### Multimodal
- [Cosmos-Predict-2.5 (NVIDIA)](https://huggingface.co/nvidia/Cosmos-Predict2.5-2B)
- [Liquid (unified multimodal generator)](https://github.com/FoundationVision/Liquid)
- [Lumina-DiMOO](https://huggingface.co/Alpha-VLLM/Lumina-DiMOO)
- [Ming (inclusionAI)](https://github.com/inclusionAI/Ming)
- [Magi (SandAI)](https://github.com/SandAI-org/MAGI-1)
- [DreamO (ByteDance)](https://huggingface.co/ByteDance/DreamO)

### Other/Unsorted
- [DiffusionForcing](https://github.com/kwsong0113/diffusion-forcing-transformer)
- [Self-Forcing](https://github.com/guandeh17/Self-Forcing)
- [SEVA](https://github.com/huggingface/diffusers/pull/11440)
- [ByteDance USO](https://github.com/bytedance/USO)
- [ByteDance Lynx](https://github.com/bytedance/lynx)
- [LanDiff](https://github.com/landiff/landiff)
- [Video Inpaint Pipeline](https://github.com/huggingface/diffusers/pull/12506)
- [Sonic Inpaint](https://github.com/ubc-vision/sonic)
- [BoxDiff](https://github.com/huggingface/diffusers/pull/7947)
- [Make-It-Count](https://github.com/Litalby1/make-it-count)
- [FreeCustom](https://github.com/aim-uofa/FreeCustom)
- [ControlNeXt](https://github.com/dvlab-research/ControlNeXt/)
- [MS-Diffusion](https://github.com/MS-Diffusion/MS-Diffusion)
- [UniRef](https://github.com/FoundationVision/UniRef)
- [AnyDoor](https://github.com/ali-vilab/AnyDoor)
- [AnyText2](https://github.com/tyxsspa/AnyText2)
- [DragonDiffusion](https://github.com/MC-E/DragonDiffusion)
- [DenseDiffusion](https://github.com/naver-ai/DenseDiffusion)
- [FlashFace](https://github.com/ali-vilab/FlashFace)
- [PowerPaint](https://github.com/open-mmlab/PowerPaint)
- [IC-Light](https://github.com/lllyasviel/IC-Light)
- [ReNO](https://github.com/ExplainableML/ReNO)
- [LoRAdapter](https://github.com/CompVis/LoRAdapter)
- [LivePortrait](https://github.com/KwaiVGI/LivePortrait)

## Migration

### Asyncio

- Policy system is deprecated and will be removed in **Python 3.16**
  - [Python 3.14 removals - asyncio](https://docs.python.org/3.14/whatsnew/3.14.html#id10)
  - https://docs.python.org/3.14/library/asyncio-policy.html
  - Affected files:
    - [`webui.py`](webui.py)
    - [`cli/sdapi.py`](cli/sdapi.py)
  - Migration:
    - [asyncio.run](https://docs.python.org/3.14/library/asyncio-runner.html#asyncio.run)
    - [asyncio.Runner](https://docs.python.org/3.14/library/asyncio-runner.html#asyncio.Runner)

### rmtree

- `onerror` deprecated and replaced with `onexc` in **Python 3.12**
``` python
    def excRemoveReadonly(func, path, exc: BaseException):
        import stat
        shared.log.debug(f'Exception during cleanup: {func} {path} {type(exc).__name__}')
        if func in (os.rmdir, os.remove, os.unlink) and isinstance(exc, PermissionError):
            shared.log.debug(f'Retrying cleanup: {path}')
            os.chmod(path, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
            func(path)
    # ...
      try:
          shutil.rmtree(found.path, ignore_errors=False, onexc=excRemoveReadonly)
```

## Code TODO

> npm run todo
 
- fc: autodetect distilled based on model
- fc: autodetect tensor format based on model
- hypertile: vae breaks when using non-standard sizes
- install: switch to pytorch source when it becomes available
- loader: load receipe
- loader: save receipe
- lora: add other quantization types
- lora: add t5 key support for sd35/f1
- lora: maybe force imediate quantization
- model load: force-reloading entire model as loading transformers only leads to massive memory usage
- model load: implement model in-memory caching
- modernui: monkey-patch for missing tabs.select event
- modules/lora/lora_extract.py:188:9: W0511: TODO: lora: support pre-quantized flux
- modules/modular_guiders.py:65:58: W0511: TODO: guiders
- processing: remove duplicate mask params
- resize image: enable full VAE mode for resize-latent
