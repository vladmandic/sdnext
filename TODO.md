# TODO

## Internal

- Update: `transformers==5.0.0`, owner @CalamitousFelicitousness
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
- Feature: LoRA add OMI format support for SD35/FLUX.1, on-hold
- Feature: Multi-user support
- Feature: Remote Text-Encoder support, sidelined for the moment
- Feature: Settings profile manager
- Feature: Video tab add full API support
- Refactor: Unify *huggingface* and *diffusers* model folders
- Refactor: Move `nunchaku` models to refernce instead of internal decision, owner @CalamitousFelicitousness
- Refactor: [GGUF](https://huggingface.co/docs/diffusers/main/en/quantization/gguf)
- Refactor: move sampler options to settings to config
- Refactor: remove `CodeFormer`, owner @CalamitousFelicitousness
- Refactor: remove `GFPGAN`, owner @CalamitousFelicitousness
- Reimplement `llama` remover for Kanvas, pending end-to-end review of `Kanvas`

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
- [Chroma Zeta](https://huggingface.co/lodestones/Zeta-Chroma): Image and video generator for creative effects and professional filters
- [Chroma Radiance](https://huggingface.co/lodestones/Chroma1-Radiance): Pixel-space model eliminating VAE artifacts for high visual fidelity
- [Liquid](https://github.com/FoundationVision/Liquid): Unified vision-language auto-regressive generation paradigm
- [Lumina-DiMOO](https://huggingface.co/Alpha-VLLM/Lumina-DiMOO): Foundational multi-modal generation and understanding via discrete diffusion
- [nVidia Cosmos-Predict-2.5](https://huggingface.co/nvidia/Cosmos-Predict2.5-2B): Physics-aware world foundation model for consistent scene prediction
- [Liquid (unified multimodal generator)](https://github.com/FoundationVision/Liquid): Auto-regressive generation paradigm across vision and language
- [Lumina-DiMOO](https://huggingface.co/Alpha-VLLM/Lumina-DiMOO): foundational multi-modal multi-task generation and understanding

### Image-Edit
- [Meituan LongCat-Image-Edit-Turbo](https://huggingface.co/meituan-longcat/LongCat-Image-Edit-Turbo):6B instruction-following image editing with high visual consistency
- [VIBE Image-Edit](https://huggingface.co/iitolstykh/VIBE-Image-Edit): (Sana+Qwen-VL)Fast visual instruction-based image editing framework
- [LucyEdit](https://github.com/huggingface/diffusers/pull/12340):Instruction-guided video editing while preserving motion and identity
- [Step1X-Edit](https://github.com/stepfun-ai/Step1X-Edit):Multimodal image editing decoding MLLM tokens via DiT
- [OneReward](https://github.com/bytedance/OneReward):Reinforcement learning grounded generative reward model for image editing
- [ByteDance DreamO](https://huggingface.co/ByteDance/DreamO): image customization framework for IP adaptation and virtual try-on

### Video
- [OpenMOSS MOVA](https://huggingface.co/OpenMOSS-Team/MOVA-720p): Unified foundation model for synchronized high-fidelity video and audio
- [Wan family (Wan2.1 / Wan2.2 variants)](https://huggingface.co/Wan-AI/Wan2.2-Animate-14B): MoE-based foundational tools for cinematic T2V/I2V/TI2V
 example: [Wan2.1-T2V-14B-CausVid](https://huggingface.co/lightx2v/Wan2.1-T2V-14B-CausVid)
 distill / step-distill examples: [Wan2.1-StepDistill-CfgDistill](https://huggingface.co/lightx2v/Wan2.1-T2V-14B-StepDistill-CfgDistill)
- [Krea Realtime Video](https://huggingface.co/krea/krea-realtime-video): (Wan2.1)Distilled real-time video diffusion using self-forcing techniques
- [MAGI-1 (autoregressive video)](https://github.com/SandAI-org/MAGI-1): Autoregressive video generation allowing infinite and timeline control
- [MUG-V 10B (video generation)](https://huggingface.co/MUG-V/MUG-V-inference): large-scale DiT-based video generation system trained via flow-matching
- [Ovi (audio/video generation)](https://github.com/character-ai/Ovi): (Wan2.2)Speech-to-video with synchronized sound effects and music
- [HunyuanVideo-Avatar / HunyuanCustom](https://huggingface.co/tencent/HunyuanVideo-Avatar): (HunyuanVideo)MM-DiT based dynamic emotion-controllable dialogue generation
- [Sana Image→Video (Sana-I2V)](https://github.com/huggingface/diffusers/pull/12634#issuecomment-3540534268): (Sana)Compact Linear DiT framework for efficient high-resolution video
- [Wan-2.2 S2V (diffusers PR)](https://github.com/huggingface/diffusers/pull/12258): (Wan2.2)Audio-driven cinematic speech-to-video generation
- [LongCat-Video](https://huggingface.co/meituan-longcat/LongCat-Video): Unified framework for minutes-long coherent video generation via Block Sparse Attention
- [LTXVideo / LTXVideo LongMulti (diffusers PR)](https://github.com/huggingface/diffusers/pull/12614): Real-time DiT-based generation with production-ready camera controls
- [DiffSynth-Studio (ModelScope)](https://github.com/modelscope/DiffSynth-Studio): (Wan2.2)Comprehensive training and quantization tools for Wan video models
- [Phantom (Phantom HuMo)](https://github.com/Phantom-video/Phantom): Human-centric video generation framework focus on subject ID consistency
- [CausVid-Plus / WAN-CausVid-Plus](https://github.com/goatWu/CausVid-Plus/): (Wan2.1)Causal diffusion for high-quality temporally consistent long videos
- [Wan2GP (workflow/GUI for Wan)](https://github.com/deepbeepmeep/Wan2GP): (Wan)Web-based UI focused on running complex video models for GPU-poor setups
- [LivePortrait](https://github.com/KwaiVGI/LivePortrait): Efficient portrait animation system with high stitching and retargeting control
- [Magi (SandAI)](https://github.com/SandAI-org/MAGI-1): High-quality autoregressive video generation framework
- [Ming (inclusionAI)](https://github.com/inclusionAI/Ming): Unified multimodal model for processing text, audio, image, and video

### Other/Unsorted
- [DiffusionForcing](https://github.com/kwsong0113/diffusion-forcing-transformer): Full-sequence diffusion with autoregressive next-token prediction
- [Self-Forcing](https://github.com/guandeh17/Self-Forcing): Framework for improving temporal consistency in long-horizon video generation
- [SEVA](https://github.com/huggingface/diffusers/pull/11440): Stable Virtual Camera for novel view synthesis and 3D-consistent video
- [ByteDance USO](https://github.com/bytedance/USO): Unified Style-Subject Optimized framework for personalized image generation
- [ByteDance Lynx](https://github.com/bytedance/lynx): State-of-the-art high-fidelity personalized video generation based on DiT
- [LanDiff](https://github.com/landiff/landiff): Coarse-to-fine text-to-video integrating Language and Diffusion Models
- [Video Inpaint Pipeline](https://github.com/huggingface/diffusers/pull/12506): Unified inpainting pipeline implementation within Diffusers library
- [Sonic Inpaint](https://github.com/ubc-vision/sonic): Audio-driven portrait animation system focus on global audio perception
- [Make-It-Count](https://github.com/Litalby1/make-it-count): CountGen method for precise numerical control of objects via object identity features
- [ControlNeXt](https://github.com/dvlab-research/ControlNeXt/): Lightweight architecture for efficient controllable image and video generation
- [MS-Diffusion](https://github.com/MS-Diffusion/MS-Diffusion): Layout-guided multi-subject image personalization framework
- [UniRef](https://github.com/FoundationVision/UniRef): Unified model for segmentation tasks designed as foundation model plug-in
- [FlashFace](https://github.com/ali-vilab/FlashFace): High-fidelity human image customization and face swapping framework
- [ReNO](https://github.com/ExplainableML/ReNO): Reward-based Noise Optimization to improve text-to-image quality during inference

### Not Planned
- [Bria FIBO](https://huggingface.co/briaai/FIBO): Fully JSON based
- [Bria FiboEdit](https://github.com/huggingface/diffusers/commit/d7a1c31f4f85bae5a9e01cdce49bd7346bd8ccd6): Fully JSON based
- [LoRAdapter](https://github.com/CompVis/LoRAdapter): Not recently updated
- [SD3 UltraEdit](https://github.com/HaozheZhao/UltraEdit): Based on SD3
- [PowerPaint](https://github.com/open-mmlab/PowerPaint): Based on SD15
- [FreeCustom](https://github.com/aim-uofa/FreeCustom): Based on SD15
- [AnyDoor](https://github.com/ali-vilab/AnyDoor): Based on SD21
- [AnyText2](https://github.com/tyxsspa/AnyText2): Based on SD15
- [DragonDiffusion](https://github.com/MC-E/DragonDiffusion): Based on SD15
- [DenseDiffusion](https://github.com/naver-ai/DenseDiffusion): Based on SD15
- [IC-Light](https://github.com/lllyasviel/IC-Light): Based on SD15

## Migration

### Asyncio

- Policy system is deprecated and will be removed in Python 3.16
 [Python 3.14 removalsasyncio](https://docs.python.org/3.14/whatsnew/3.14.html#id10)
 https://docs.python.org/3.14/library/asyncio-policy.html
 Affected files:
   [`webui.py`](webui.py)
   [`cli/sdapi.py`](cli/sdapi.py)
 Migration:
   [asyncio.run](https://docs.python.org/3.14/library/asyncio-runner.html#asyncio.run)
   [asyncio.Runner](https://docs.python.org/3.14/library/asyncio-runner.html#asyncio.Runner)

### rmtree

- `onerror` deprecated and replaced with `onexc` in Python 3.12
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
