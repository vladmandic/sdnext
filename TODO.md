# TODO

## Internal

- Feature: implement `unload_auxiliary_models`
- Feature: RIFE update
- Feature: RIFE in processing
- Feature: SeedVR2 in processing
- Feature: Add video models to `Reference`
- Feature: Add <https://huggingface.co/briaai/RMBG-2.0> to REMBG
- Deploy: Lite vs Expert mode
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

- Port: ERNIE-Image (merged, unpublished)
- Port: NucleusMoE-Image (merged, unpublished)
- Port: JoyAI-Image-Edit (pr in-progress)
- Port: Lumina-DiMOO (pr in-progress)
- Port: Step1X-Edit (pr in-progress)
- [VIBE Image Edit](https://huggingface.co/iitolstykh/VIBE-Image-Edit)
- [UltraFlux](https://huggingface.co/Owen777/UltraFlux-v1)
- [Mugen](https://huggingface.co/CabalResearch/Mugen)
- [Liquid](https://github.com/FoundationVision/Liquid)
- [nVidia Cosmos-Predict-2.5](https://huggingface.co/nvidia/Cosmos-Predict2.5-2B)
- [Liquid](https://github.com/FoundationVision/Liquid)
- [Tencent HY-WU](https://huggingface.co/tencent/HY-WU)
- [nVidia Cosmos-Transfer-2.5](https://github.com/huggingface/diffusers/pull/13066)

### Video

- [HY-OmniWeaving](https://huggingface.co/tencent/HY-OmniWeaving)
- [LTX-Condition](https://huggingface.co/Lightricks/LTX-2)
- [LTX-Distilled](https://huggingface.co/Lightricks/LTX-2)
- [OpenMOSS MOVA](https://huggingface.co/OpenMOSS-Team/MOVA-720p)
- [Wan2.2-Animate](https://huggingface.co/Wan-AI/Wan2.2-Animate-14B)
- [Wan2.1-T2V-14B-CausVid](https://huggingface.co/lightx2v/Wan2.1-T2V-14B-CausVid)
- [Wan2.1-StepDistill-CfgDistill](https://huggingface.co/lightx2v/Wan2.1-T2V-14B-StepDistill-CfgDistill)
- [Krea Realtime Video](https://huggingface.co/krea/krea-realtime-video)
- [MAGI-1](https://github.com/SandAI-org/MAGI-1)
- [MUG-V 10B](https://huggingface.co/MUG-V/MUG-V-inference)
- [Ovi](https://github.com/character-ai/Ovi)
- [LucyEdit](https://huggingface.co/decart-ai/Lucy-Edit-1.1-Dev)
- [HunyuanVideo-Avatar](https://huggingface.co/tencent/HunyuanVideo-Avatar)
- [Sana I2V](https://huggingface.co/Efficient-Large-Model/SANA-Video_2B_480p_diffusers)
- [Wan-2.2 S2V](https://huggingface.co/Wan-AI/Wan2.2-S2V-14B)
- [Meituan LongCat-Video](https://huggingface.co/meituan-longcat/LongCat-Video)
- [LTXVideo LongMulti](https://huggingface.co/Lightricks/LTX-Video-0.9.8-13B-distilled)
- [Phantom HuMo](https://github.com/Phantom-video/Phantom)
- [CausVid-Plus](https://github.com/goatWu/CausVid-Plus/)
- [LivePortrait](https://github.com/KwaiVGI/LivePortrait)
- [Magi (SandAI)](https://github.com/SandAI-org/MAGI-1)
- [Ming (inclusionAI)](https://github.com/inclusionAI/Ming)
- [HummingbirdXT](https://huggingface.co/amd/HummingbirdXT)
- [DiffusionForcing](https://github.com/kwsong0113/diffusion-forcing-transformer)
- [ByteDance Lynx](https://github.com/bytedance/lynx)
- [LanDiff](https://github.com/landiff/landiff)

### Other/Unsorted

- [ByteDance DreamO](https://github.com/bytedance/DreamO)
  - Unified image customization framework combining face identity preservation, virtual try-on, style transfer, etc.
  - Created: 2025-05 | Updated: 2025-08 | Stars: 1,700
- [ControlNeXt](https://github.com/dvlab-research/ControlNeXt/)
  - Lightweight controllable generation framework for images and videos (SD1.5, SDXL, SVD) that uses up to 90% fewer trainable parameters than ControlNet
  - Created: 2024-08 | Updated: 2024-08 | Stars: 1,600
- [ByteDance USO](https://github.com/bytedance/USO)
  - Unified model for both style-transfer and subject-driven image generation from one or two reference images
  - Created: 2025-08 | Updated: 2025-09 | Stars: 1,200
- [TwinFlow](https://github.com/inclusionAI/TwinFlow)
  - Distillation technique that converts large image generation models into 1–2 step generators without requiring a separate teacher model
  - Created: 2025-12 | Updated: 2026-02 | Stars: 506
- [FlashFace](https://github.com/ali-vilab/FlashFace)
  - Zero-shot face personalization method that generates images of a specific person from one or a few reference photos
  - Created: 2024-03 | Updated: 2024-05 | Stars: 436
- [DiffSynth-Engine](https://github.com/modelscope/DiffSynth-Engine)
  - Alternative to diffusers library that unlocks some diffsynth specific capabilities
  - Created: 2024-05 | Updated: 2026-03 | Stars: 393
- [MS-Diffusion](https://github.com/MS-Diffusion/MS-Diffusion)
  - Multi-subject image personalization framework that uses layout guidance to place multiple reference subjects in a single generated image without identity confusion
  - Created: 2024-04 | Updated: 2025-07 | Stars: 309
- [RamTorch](https://github.com/lodestone-rock/ramtorch)
  - Alternative memory management and offloading library
  - Created: 2025-09 | Updated: 2026-04 | Stars: 266
- [UniRef](https://github.com/FoundationVision/UniRef)
  - Unified segmentation model that handles referring image segmentation and few-shot segmentation
  - Created: 2023-04 | Updated: 2025-04 | Stars: 238
- [FreeFuse](https://github.com/yaoliliu/FreeFuse)
  - Training-free method to combine multiple subject LoRAs in one image generation without conflicts, by automatically routing each LoRA's influence to its target spatial region.
  - Created: 2026-01 | Updated: 2026-03 | Stars: 178
- [mmgp](https://github.com/deepbeepmeep/mmgp)
  - Alternative memory management and offloading library
  - Created: 2024-03 | Updated: 2026-02 | Stars: 175
- [ReNO](https://github.com/ExplainableML/ReNO)
  - Inference-time technique that improves one-step text-to-image models by iteratively optimizing the initial noise using reward model signals, boosting prompt accuracy in 20–50 seconds
  - Created: 2024-06 | Updated: 2025-09 | Stars: 166
- [RegionE](https://github.com/Peyton-Chen/RegionE)
  - Speeds up instruction-based image editing by skipping redundant computation in image regions that are not being changed.
  - Created: 2025-10 | Updated: 2026-02 | Stars: 98
- [Make-It-Count](https://github.com/Litalby1/make-it-count)
  - Method that reliably generates the exact number of objects requested by tracking instance identities during denoising
  - Created: 2024-04 | Updated: 2025-04 | Stars: 96
- [FaceClip](https://huggingface.co/ByteDance/FaceCLIP)
  - Identity-preserving image generation model that jointly encodes a face and a text prompt into a shared embedding to produce portraits matching both the subject's appearance and the scene description
  - Created: 2025-04 | Updated: 2025-04 | Likes: 88
- [T5Gemma Adapter](https://github.com/NeuroSenko/ComfyUI_LLM_SDXL_Adapter)
  - Experiment that replaces the SDXL text encoder with a T5Gemma LLM via a trained adapter for richer prompt understanding
  - Created: 2025-07 | Updated: 2025-10 | Stars: 67
- [Sonic Inpaint](https://github.com/ubc-vision/sonic)
  - Image inpainting method that optimizes for better masked-region filling
  - Created: 2025-11 | Updated: 2026-01 | Stars: 23
- [SEVA](https://github.com/Stability-AI/stable-virtual-camera)
  - Model that generates novel-view images of a scene from a single input photo.
  - Created: 2025-04 | Updated: 2025-06 | Stars: N/A (draft PR)

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
