# TODO

## Release

- Review: feat/tag-dictionaries
- Review: er-sde-solver-scheduler
- Test: Zeta-Chroma
- Test: Anima-Preview-3
- Test: SDXS-1B
- Test: VIBE-Image-Edit
- Test: Bria-FIBO
- Test: Lumina-DiMOO
- Test: Step1X-Edit
- Code: prompt encode for Bria-FIBO: <https://github.com/Bria-AI/Fibo-Edit/blob/master/src/fibo_edit/fibo_edit_vlm.py>
- Port: ERNIE-Image (merged, unpublished)
- Port: NucleusMoE-Image (merged, unpublished)
- Port: JoyAI-Image-Edit (in-progress, published)

## Internal

- Feature: implement `unload_auxiliary_models`
- Feature: RIFE update
- Feature: RIFE in processing
- Feature: SeedVR2 in processing
- Feature: Add video models to `Reference`
- Feature: Add <https://huggingface.co/briaai/RMBG-2.0> to REMBG
- Deploy: Lite vs Expert mode
- Engine: [mmgp](https://github.com/deepbeepmeep/mmgp)
- Engine: `TensorRT` acceleration
- Feature: Auto handle scheduler `prediction_type`
- Feature: Cache models in memory
- Feature: JSON image metadata
- Validate: Control tab add overrides handling
- Feature: Integrate natural language image search
  [ImageDB](https://github.com/vladmandic/imagedb)
- Feature: Multi-user support
- Feature: Settings profile manager
- Feature: Video tab add full API support
- Refactor: Unify *huggingface* and *diffusers* model folders
- Refactor: [GGUF](https://huggingface.co/docs/diffusers/main/en/quantization/gguf)
- Reimplement `llama` remover for Kanvas
- Integrate: [Depth3D](https://github.com/vladmandic/sd-extension-depth3d)

## OnHold

- Feature: LoRA add OMI format support for SD35/FLUX.1, on-hold
- Feature: Remote Text-Encoder support, sidelined for the moment

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

### Image

- [Mugen](https://huggingface.co/CabalResearch/Mugen)
- [NucleusMoe](https://github.com/huggingface/diffusers/pull/13317)
- [Liquid](https://github.com/FoundationVision/Liquid)
- [nVidia Cosmos-Predict-2.5](https://huggingface.co/nvidia/Cosmos-Predict2.5-2B)
- [Liquid (unified multimodal generator)](https://github.com/FoundationVision/Liquid)
- [Tencent HY-WU](https://huggingface.co/tencent/HY-WU)
- [nVidia Cosmos-Transfer-2.5](https://github.com/huggingface/diffusers/pull/13066)

### Video

- [HY-OmniWeaving](https://huggingface.co/tencent/HY-OmniWeaving)
- [LTX-Condition](https://github.com/huggingface/diffusers/pull/13058)
- [LTX-Distilled](https://github.com/huggingface/diffusers/pull/12934)
- [OpenMOSS MOVA](https://huggingface.co/OpenMOSS-Team/MOVA-720p): Unified foundation model for synchronized high-fidelity video and audio
- [Wan family (Wan2.1 / Wan2.2 variants)](https://huggingface.co/Wan-AI/Wan2.2-Animate-14B): MoE-based foundational tools for cinematic T2V/I2V/TI2V
 example: [Wan2.1-T2V-14B-CausVid](https://huggingface.co/lightx2v/Wan2.1-T2V-14B-CausVid)
 distill / step-distill examples: [Wan2.1-StepDistill-CfgDistill](https://huggingface.co/lightx2v/Wan2.1-T2V-14B-StepDistill-CfgDistill)
- [Krea Realtime Video](https://huggingface.co/krea/krea-realtime-video): (Wan2.1)Distilled real-time video diffusion using self-forcing techniques
- [MAGI-1 (autoregressive video)](https://github.com/SandAI-org/MAGI-1): Autoregressive video generation allowing infinite and timeline control
- [MUG-V 10B (video generation)](https://huggingface.co/MUG-V/MUG-V-inference): large-scale DiT-based video generation system trained via flow-matching
- [Ovi (audio/video generation)](https://github.com/character-ai/Ovi): (Wan2.2)Speech-to-video with synchronized sound effects and music
- [LucyEdit](https://github.com/huggingface/diffusers/pull/12340):Instruction-guided video editing while preserving motion and identity
- [HunyuanVideo-Avatar / HunyuanCustom](https://huggingface.co/tencent/HunyuanVideo-Avatar): (HunyuanVideo)MM-DiT based dynamic emotion-controllable dialogue generation
- [Sana Image→Video (Sana-I2V)](https://github.com/huggingface/diffusers/pull/12634#issuecomment-3540534268): (Sana)Compact Linear DiT framework for efficient high-resolution video
- [Wan-2.2 S2V (diffusers PR)](https://github.com/huggingface/diffusers/pull/12258): (Wan2.2)Audio-driven cinematic speech-to-video generation
- [Meituan LongCat-Video](https://huggingface.co/meituan-longcat/LongCat-Video): Unified framework for minutes-long coherent video generation via Block Sparse Attention
- [LTXVideo / LTXVideo LongMulti (diffusers PR)](https://github.com/huggingface/diffusers/pull/12614): Real-time DiT-based generation with production-ready camera controls
- [DiffSynth-Studio (ModelScope)](https://github.com/modelscope/DiffSynth-Studio): (Wan2.2)Comprehensive training and quantization tools for Wan video models
- [Phantom (Phantom HuMo)](https://github.com/Phantom-video/Phantom): Human-centric video generation framework focus on subject ID consistency
- [CausVid-Plus / WAN-CausVid-Plus](https://github.com/goatWu/CausVid-Plus/): (Wan2.1)Causal diffusion for high-quality temporally consistent long videos
- [Wan2GP (workflow/GUI for Wan)](https://github.com/deepbeepmeep/Wan2GP): (Wan)Web-based UI focused on running complex video models for GPU-poor setups
- [LivePortrait](https://github.com/KwaiVGI/LivePortrait): Efficient portrait animation system with high stitching and retargeting control
- [Magi (SandAI)](https://github.com/SandAI-org/MAGI-1): High-quality autoregressive video generation framework
- [Ming (inclusionAI)](https://github.com/inclusionAI/Ming): Unified multimodal model for processing text, audio, image, and video

### Other/Unsorted

- [OneReward](https://github.com/bytedance/OneReward)
- [ByteDance DreamO](https://huggingface.co/ByteDance/DreamO)
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

## Code TODO

> npm run todo

```code
installer.py:642:15: W0511: TODO rocm: switch to pytorch source when it becomes available (fixme)
modules/transformer_cache.py:29:61: W0511: TODO fc: autodetect tensor format based on model (fixme)
modules/transformer_cache.py:30:50: W0511: TODO fc: autodetect distilled based on model (fixme)
modules/processing_class.py:404:32: W0511: TODO processing: remove duplicate mask params (fixme)
modules/sd_samplers_diffusers.py:355:31: W0511: TODO enso-required (fixme)
modules/sd_models.py:1356:5: W0511: TODO model load: implement model in-memory caching (fixme)
modules/ui_models_load.py:257:5: W0511: TODO loader: load receipe (fixme)
modules/ui_models_load.py:264:5: W0511: TODO loader: save receipe (fixme)
modules/sd_hijack_hypertile.py:123:17: W0511: TODO hypertile: vae breaks when using non-standard sizes (fixme)
modules/sd_unet.py:77:39: W0511: TODO model load: force-reloading entire model as loading transformers only leads to massive memory usage (fixme)
modules/modular_guiders.py:66:51: W0511: TODO: guiders (fixme)
```
