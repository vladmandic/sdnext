# TODO

## Issues

None

## Features

### Roadmap

- OpenAI API interface for image generation, @vladmandic
- Integrate natural language image search: [ImageDB](https://github.com/vladmandic/imagedb), @vladmandic
- Support cloud providers, @CalamitousFelicitousness
- Automated testing and integrate models repo, @CalamitousFelicitousness
- Video: unify execution path for ui and api
- Video: support finetunes
- Video: use Networks/Reference instead of custom, @vladmandic
- Video: implement LTX2DFRPipeline
- Video: upscaling LTX-Upscaler
- Video: upscaling [MiniMax-Upscaler](https://huggingface.co/LBH-123-AI/Minimax_h3_latent_Upscaler)
- Video: capabilities to processing tab, add RIFE, upscaling (once available)
- UI Lite vs Expert mode, @vladmandic
- Incorporate [prompting guides](https://github.com/CalamitousFelicitousness/ai-prompting-guides)

### OnHold

- LoRA add OMI format support for SD35/FLUX.1
- Remote Text-Encoder support, sidelined for the moment
- Auto handle scheduler `prediction_type`
- Multi-user support
- Settings profile manager
- TensorRT acceleration
- Cache models in memory
- Unify *huggingface* and *diffusers* model folders
- JSON image metadata

## New models / Pipelines

### Image

- [Nemotron-Labs-Image-30B-A3B](https://huggingface.co/nvidia/Nemotron-Labs-Image-30B-A3B)

### Video

- [Kandinski 6.0](https://huggingface.co/collections/kandinskylab/kandinsky-60-diffusers)
- [Bernini-R](https://huggingface.co/ByteDance/Bernini-R-Diffusers)
- [Bernini-R 1.3B](https://huggingface.co/ByteDance/Bernini-R-1.3B-Diffusers)
- [PAN2](https://huggingface.co/IFM/PAN2)
- [ByteDance Lance](https://github.com/bytedance/Lance)
- [HY-OmniWeaving](https://huggingface.co/tencent/HY-OmniWeaving)
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
- [Phantom HuMo](https://github.com/Phantom-video/Phantom)
- [CausVid-Plus](https://github.com/goatWu/CausVid-Plus/)
- [LivePortrait](https://github.com/KwaiVGI/LivePortrait)
- [SandAI Magi](https://github.com/SandAI-org/MAGI-1)
- [inclusionAI Ming](https://github.com/inclusionAI/Ming)
- [HummingbirdXT](https://huggingface.co/amd/HummingbirdXT)
- [DiffusionForcing](https://github.com/kwsong0113/diffusion-forcing-transformer)
- [ByteDance Lynx](https://github.com/bytedance/lynx)
- [LanDiff](https://github.com/landiff/landiff)

### Other/Unsorted

- [GPEN Face Restoration](https://github.com/yangxy/GPEN)
- [ByteDance DreamO](https://github.com/bytedance/DreamO)
  - Unified image customization framework combining face identity preservation, virtual try-on, style transfer, etc.
  - Created: 2025-05 | Updated: 2025-08 | Stars: 1,700
- [ByteDance USO](https://github.com/bytedance/USO)
  - Unified model for both style-transfer and subject-driven image generation from one or two reference images
  - Created: 2025-08 | Updated: 2025-09 | Stars: 1,200
- [TwinFlow](https://github.com/inclusionAI/TwinFlow)
  - Distillation technique that converts large image generation models into 1–2 step generators without requiring a separate teacher model
  - Created: 2025-12 | Updated: 2026-02 | Stars: 506
- [FreeFuse](https://github.com/yaoliliu/FreeFuse)
  - Training-free method to combine multiple subject LoRAs in one image generation without conflicts, by automatically routing each LoRA's influence to its target spatial region.
  - Created: 2026-01 | Updated: 2026-03 | Stars: 178
- [RegionE](https://github.com/Peyton-Chen/RegionE)
  - Speeds up instruction-based image editing by skipping redundant computation in image regions that are not being changed.
  - Created: 2025-10 | Updated: 2026-02 | Stars: 98
- [FaceClip](https://huggingface.co/ByteDance/FaceCLIP)
  - Identity-preserving image generation model that jointly encodes a face and a text prompt into a shared embedding to produce portraits matching both the subject's appearance and the scene description
  - Created: 2025-04 | Updated: 2025-04 | Likes: 88
- [Sonic Inpaint](https://github.com/ubc-vision/sonic)
  - Image inpainting method that optimizes for better masked-region filling
  - Created: 2025-11 | Updated: 2026-01 | Stars: 23
- [SEVA](https://github.com/Stability-AI/stable-virtual-camera)
  - Model that generates novel-view images of a scene from a single input photo.
  - Created: 2025-04 | Updated: 2025-06 | Stars: N/A (draft PR)

### Tech

- [DiffSynth-Engine](https://github.com/modelscope/DiffSynth-Engine) Alternative to diffusers library that unlocks some diffsynth specific capabilities
- [RamTorch](https://github.com/lodestone-rock/ramtorch)
- [mmgp](https://github.com/deepbeepmeep/mmgp)
- [DDO](https://github.com/DEVAIEXP/diffusers-dynamic-offloader)
- [QuantFunc](https://github.com/QuantFunc/ComfyUI-QuantFunc)

### Rejected

- [Mugen](https://huggingface.co/CabalResearch/Mugen) (sdxl with flux vae experiment, not clean)
- [Liquid](https://github.com/FoundationVision/Liquid) (autoregressive, not clean)
- [SenseNova-U1.5](https://huggingface.co/sensenova/SenseNova-U1.5-8B-MoT) (custom multi-modal)
- [Iris-3B](https://huggingface.co/speridlabs/iris-3b)

## Code TODO

> pnpm run todo

```code
modules/caption/vqa.py:TODO vqa: implement handling of custom args # pylint: disable=unused-argument
modules/control/run.py:TODO modernui: monkey-patch for missing tabs.select event
modules/history.py:TODO: apply metadata, preview, load/save
modules/processing_class.py:TODO processing: remove duplicate mask params
modules/sd_hijack_hypertile.py:TODO hypertile: vae breaks when using non-standard sizes
modules/sd_samplers_diffusers.py:TODO enso-required
modules/ui_models_load.py:TODO loader: load receipe
modules/ui_models_load.py:TODO loader: save receipe
pipelines/objectclear/pipeline_objectclear.py:TODO: Logic should ideally just be moved out of the pipeline
```
