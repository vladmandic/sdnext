# TODO

## Project Board

- <https://github.com/users/vladmandic/projects>

## Internal

- Feature: Move `nunchaku` models to refernce instead of internal decision  
- Update: `transformers==5.0.0`
- Feature: Unify *huggingface* and *diffusers* model folders
- Reimplement `llama` remover for Kanvas
- Deploy: Create executable for SD.Next
- Feature: Integrate natural language image search
  [ImageDB](https://github.com/vladmandic/imagedb)
- Feature: Remote Text-Encoder support
- Refactor: move sampler options to settings to config
- Refactor: [GGUF](https://huggingface.co/docs/diffusers/main/en/quantization/gguf)
- Feature: LoRA add OMI format support for SD35/FLUX.1
- Refactor: remove `CodeFormer`
- Refactor: remove `GFPGAN`
- UI: Lite vs Expert mode
- Video tab: add full API support
- Control tab: add overrides handling
- Engine: `TensorRT` acceleration
- Engine: [mmgp](https://github.com/deepbeepmeep/mmgp)
- Engine: [sharpfin](https://github.com/drhead/sharpfin) instead of `torchvision`

## Modular

- Switch to modular pipelines
- Feature: Transformers unified cache handler
- Refactor: [Modular pipelines and guiders](https://github.com/huggingface/diffusers/issues/11915)
- [MagCache](https://github.com/lllyasviel/FramePack/pull/673/files)
- [SmoothCache](https://github.com/huggingface/diffusers/issues/11135)

## Features

- [Flux.2 TinyVAE](https://huggingface.co/fal/FLUX.2-Tiny-AutoEncoder)
- [IPAdapter composition](https://huggingface.co/ostris/ip-composition-adapter)
- [IPAdapter negative guidance](https://github.com/huggingface/diffusers/discussions/7167)
- [STG](https://github.com/huggingface/diffusers/blob/main/examples/community/README.md#spatiotemporal-skip-guidance)
- [Video Inpaint Pipeline](https://github.com/huggingface/diffusers/pull/12506)
- [Sonic Inpaint](https://github.com/ubc-vision/sonic)

### New models / Pipelines

TODO: Investigate which models are diffusers-compatible and prioritize!

- [Bria FiboEdit](https://github.com/huggingface/diffusers/commit/d7a1c31f4f85bae5a9e01cdce49bd7346bd8ccd6)
- [LTXVideo 0.98 LongMulti](https://github.com/huggingface/diffusers/pull/12614)
- [Cosmos-Predict-2.5](https://huggingface.co/nvidia/Cosmos-Predict2.5-2B)
- [NewBie Image Exp0.1](https://github.com/huggingface/diffusers/pull/12803)
- [Sana-I2V](https://github.com/huggingface/diffusers/pull/12634#issuecomment-3540534268)
- [Bria FIBO](https://huggingface.co/briaai/FIBO)
- [Bytedance Lynx](https://github.com/bytedance/lynx)
- [ByteDance OneReward](https://github.com/bytedance/OneReward)
- [ByteDance USO](https://github.com/bytedance/USO)
- [Chroma Radiance](https://huggingface.co/lodestones/Chroma1-Radiance)
- [Chroma Zeta](https://huggingface.co/lodestones/Zeta-Chroma)
- [DiffSynth Studio](https://github.com/modelscope/DiffSynth-Studio)
- [DiffusionForcing](https://github.com/kwsong0113/diffusion-forcing-transformer)  
- [Dream0 guidance](https://huggingface.co/ByteDance/DreamO)  
- [HunyuanAvatar](https://huggingface.co/tencent/HunyuanVideo-Avatar)  
- [HunyuanCustom](https://github.com/Tencent-Hunyuan/HunyuanCustom)  
- [Inf-DiT](https://github.com/zai-org/Inf-DiT)
- [Krea Realtime Video](https://huggingface.co/krea/krea-realtime-video)
- [LanDiff](https://github.com/landiff/landiff)  
- [Liquid](https://github.com/FoundationVision/Liquid)  
- [LongCat-Video](https://huggingface.co/meituan-longcat/LongCat-Video)
- [LucyEdit](https://github.com/huggingface/diffusers/pull/12340)
- [Lumina-DiMOO](https://huggingface.co/Alpha-VLLM/Lumina-DiMOO)
- [Magi](https://github.com/SandAI-org/MAGI-1)(https://github.com/huggingface/diffusers/pull/11713)  
- [Ming](https://github.com/inclusionAI/Ming)  
- [MUG-V 10B](https://huggingface.co/MUG-V/MUG-V-inference)
- [Ovi](https://github.com/character-ai/Ovi)
- [Phantom HuMo](https://github.com/Phantom-video/Phantom)
- [SD3 UltraEdit](https://github.com/HaozheZhao/UltraEdit)  
- [SelfForcing](https://github.com/guandeh17/Self-Forcing)  
- [SEVA](https://github.com/huggingface/diffusers/pull/11440)  
- [Step1X](https://github.com/stepfun-ai/Step1X-Edit)  
- [Wan-2.2 Animate](https://github.com/huggingface/diffusers/pull/12526)
- [Wan-2.2 S2V](https://github.com/huggingface/diffusers/pull/12258)
- [WAN-CausVid-Plus t2v](https://github.com/goatWu/CausVid-Plus/)  
- [WAN-CausVid](https://huggingface.co/lightx2v/Wan2.1-T2V-14B-CausVid)  
- [WAN-StepDistill](https://huggingface.co/lightx2v/Wan2.1-T2V-14B-StepDistill-CfgDistill)  
- [Wan2.2-Animate-14B](https://huggingface.co/Wan-AI/Wan2.2-Animate-14B)
- [WAN2GP](https://github.com/deepbeepmeep/Wan2GP)  

### Migration

#### Asyncio

- Policy system is deprecated and will be removed in **Python 3.16**
  - [Python 3.14 removals - asyncio](https://docs.python.org/3.14/whatsnew/3.14.html#id10)
  - https://docs.python.org/3.14/library/asyncio-policy.html
  - Affected files:
    - [`webui.py`](webui.py)
    - [`cli/sdapi.py`](cli/sdapi.py)
  - Migration:
    - [asyncio.run](https://docs.python.org/3.14/library/asyncio-runner.html#asyncio.run)
    - [asyncio.Runner](https://docs.python.org/3.14/library/asyncio-runner.html#asyncio.Runner)

#### rmtree

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
