# TODO

Main ToDo list can be found at [GitHub projects](https://github.com/users/vladmandic/projects)

## Future Candidates

- Remote TE  
- [Canvas](https://konvajs.org/)  
- Refactor: [Modular pipelines and guiders](https://github.com/huggingface/diffusers/issues/11915)  
- Refactor: Sampler options  
- Refactor: [GGUF](https://huggingface.co/docs/diffusers/main/en/quantization/gguf)  
- Feature: LoRA add OMI format support for SD35/FLUX.1  
- Video Core: API  
- Video LTX: TeaCache and others, API, Conditioning preprocess Video: LTX API  

### Under Consideration

- [Inf-DiT](https://github.com/zai-org/Inf-DiT)
- [X-Omni](https://github.com/X-Omni-Team/X-Omni/blob/main/README.md)
- [DiffSynth Studio](https://github.com/modelscope/DiffSynth-Studio)
- [IPAdapter negative guidance](https://github.com/huggingface/diffusers/discussions/7167)  
- [IPAdapter composition](https://huggingface.co/ostris/ip-composition-adapter)  
- [STG](https://github.com/huggingface/diffusers/blob/main/examples/community/README.md#spatiotemporal-skip-guidance)  
- [SmoothCache](https://github.com/huggingface/diffusers/issues/11135)  
- [MagCache](https://github.com/lllyasviel/FramePack/pull/673/files)  
- [Nunchaku PulID](https://github.com/mit-han-lab/nunchaku/pull/274)  
- [Dream0 guidance](https://huggingface.co/ByteDance/DreamO)  
- [SUPIR upscaler](https://github.com/Fanghua-Yu/SUPIR)  
- [ByteDance OneReward](https://github.com/bytedance/OneReward)
- [ByteDance USO](https://github.com/bytedance/USO)
- Remove: `Agent Scheduler`  
- Remove: `CodeFormer`
- Remove: `GFPGAN`  
- ModernUI: Lite vs Expert mode  
- Engine: TensorRT acceleration

### New models

- [HunyuanImage](https://huggingface.co/tencent/HunyuanImage-2.1)
- [Phantom HuMo](https://github.com/Phantom-video/Phantom)
- [Lumina-DiMOO](https://huggingface.co/Alpha-VLLM/Lumina-DiMOO)
- [Magi](https://github.com/SandAI-org/MAGI-1)(https://github.com/huggingface/diffusers/pull/11713)  
- [SEVA](https://github.com/huggingface/diffusers/pull/11440)  
- [Ming](https://github.com/inclusionAI/Ming)  
- [Liquid](https://github.com/FoundationVision/Liquid)  
- [Step1X](https://github.com/stepfun-ai/Step1X-Edit)  
- [LucyEdit](https://github.com/huggingface/diffusers/pull/12340)
- [SD3 UltraEdit](https://github.com/HaozheZhao/UltraEdit)  
- [WAN2GP](https://github.com/deepbeepmeep/Wan2GP)  
- [SelfForcing](https://github.com/guandeh17/Self-Forcing)  
- [DiffusionForcing](https://github.com/kwsong0113/diffusion-forcing-transformer)  
- [LanDiff](https://github.com/landiff/landiff)  
- [HunyuanCustom](https://github.com/Tencent-Hunyuan/HunyuanCustom)  
- [HunyuanAvatar](https://huggingface.co/tencent/HunyuanVideo-Avatar)  
- [WAN-CausVid](https://huggingface.co/lightx2v/Wan2.1-T2V-14B-CausVid)  
- [WAN-CausVid-Plus t2v](https://github.com/goatWu/CausVid-Plus/)  
- [WAN-StepDistill](https://huggingface.co/lightx2v/Wan2.1-T2V-14B-StepDistill-CfgDistill)  

## Code TODO

> pnpm lint | grep W0511 | awk -F'TODO ' '{print "- "$NF}' | sed 's/ (fixme)//g' | sort
 
- control: support scripts via api
- fc: autodetect distilled based on model
- fc: autodetect tensor format based on model
- hypertile: vae breaks when using non-standard sizes
- install: enable ROCm for windows when available
- loader: load receipe
- loader: save receipe
- lora: add other quantization types
- lora: add t5 key support for sd35/f1
- lora: maybe force imediate quantization
- model load: force-reloading entire model as loading transformers only leads to massive memory usage
- model load: implement model in-memory caching
- modernui: monkey-patch for missing tabs.select event
- modules/lora/lora_extract.py:188:9: W0511: TODO: lora: support pre-quantized flux
- processing: remove duplicate mask params
- resize image: enable full VAE mode for resize-latent
