# TODO

Main ToDo list can be found at [GitHub projects](https://github.com/users/vladmandic/projects)

## Current

## Future Candidates

### Features

- Model receipe: load/save
- Python 3.13 improved support
- Control: API enhance scripts compatibility  
- Video: API support  

### Enhancements

- [IPAdapter negative guidance](https://github.com/huggingface/diffusers/discussions/7167)  
- [STG](https://github.com/huggingface/diffusers/blob/main/examples/community/README.md#spatiotemporal-skip-guidance)  
- [LBM](https://github.com/gojasper/LBM)  
- [SmoothCache](https://github.com/huggingface/diffusers/issues/11135)  
- [HiDream GGUF](https://github.com/huggingface/diffusers/pull/11550)  
- [Diffusers guiders](https://github.com/huggingface/diffusers/pull/11311)  
- [Nunchaku PulID](https://github.com/mit-han-lab/nunchaku/pull/274)  
- [Pydantic changes](https://github.com/Cschlaefli/automatic)  
- [Dream0 guidance](https://huggingface.co/ByteDance/DreamO)  

### Models

- [AniSora t2v](https://github.com/bilibili/Index-anisora)  
- [Ming t2i](https://github.com/inclusionAI/Ming)
- [Magi t2v](https://github.com/SandAI-org/MAGI-1)  
- [SEVA t2v](https://github.com/huggingface/diffusers/pull/11440)  
- [WanAI-2.1 VACE t2v](https://huggingface.co/Wan-AI/Wan2.1-VACE-14B)  
- [Bagel t2i](https://github.com/bytedance-seed/bagel)
- [OmniGen2 t2i](https://huggingface.co/OmniGen2/OmniGen2)
- [Step1X i2i](https://github.com/stepfun-ai/Step1X-Edit)
- [LTXVideo t2v](https://github.com/Lightricks/LTX-Video?tab=readme-ov-file#diffusers-integration)
- [LTXVideo t2v](https://github.com/huggingface/diffusers/pull/11516)  
- [SkyReels-v2 t2v](https://github.com/SkyworkAI/SkyReels-V2)
- [SkyReels-v2 t2v](https://github.com/huggingface/diffusers/pull/11518)  
- [Cosmos2 i2v](https://huggingface.co/nvidia/Cosmos-Predict2-2B-Video2World)
- [WAN2GP t2v](https://github.com/deepbeepmeep/Wan2GP)
- [SelfForcing t2v](https://github.com/guandeh17/Self-Forcing)
- [DiffusionForcing t2v](https://github.com/kwsong0113/diffusion-forcing-transformer)
- [LanDiff t2v](https://github.com/landiff/landiff)
- [HunyuanCustom t2v](https://github.com/Tencent-Hunyuan/HunyuanCustom)
- [WAN-CausVid t2v](https://huggingface.co/lightx2v/Wan2.1-T2V-14B-CausVid)
- [WAN-StepDistill t2v](https://huggingface.co/lightx2v/Wan2.1-T2V-14B-StepDistill-CfgDistill)
- [WAN-CausVid-Plus t2v](https://github.com/goatWu/CausVid-Plus/)  
- [HunyuanAvatar t2v](https://huggingface.co/tencent/HunyuanVideo-Avatar)

## Code TODO

> pnpm lint | grep W0511 | awk -F'TODO ' '{print "- "$NF}' | sed 's/ (fixme)//g'
 
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
- lora: support pre-quantized flux
- model load: force-reloading entire model as loading transformers only leads to massive memory usage
- model loader: implement model in-memory caching
- modernui: monkey-patch for missing tabs.select event
- nunchaku: batch support
- nunchaku: cache-dir for transformer and t5 loader
- processing: remove duplicate mask params
- resize image: enable full VAE mode for resize-latent
