# TODO

Main ToDo list can be found at [GitHub projects](https://github.com/users/vladmandic/projects)

## Current

### Issues/Limitations

- Control: API enhance scripts compatibility  
- Video: API support  

## Future Candidates

- [IPAdapter negative guidance](https://github.com/huggingface/diffusers/discussions/7167)  
- [STG](https://github.com/huggingface/diffusers/blob/main/examples/community/README.md#spatiotemporal-skip-guidance)  
- [LBM](https://github.com/gojasper/LBM)  
- [SmoothCache](https://github.com/huggingface/diffusers/issues/11135)  
- [Magi](https://github.com/SandAI-org/MAGI-1)  
- [SkyReels-v2](https://github.com/huggingface/diffusers/pull/11518)  
- [WanAI-2.1 VACE](https://huggingface.co/Wan-AI/Wan2.1-VACE-14B)  
- [LTXVideo-0.9.7](https://github.com/huggingface/diffusers/pull/11516)  
- [VisualClose](https://github.com/huggingface/diffusers/pull/11377)  
- [SEVA](https://github.com/huggingface/diffusers/pull/11440)  
- [CausVid-Plus](https://github.com/goatWu/CausVid-Plus/)  
- [Index-AniSora](https://github.com/bilibili/Index-anisora)  
- [HiDream GGUF](https://github.com/huggingface/diffusers/pull/11550)  
- [JoyCaption-Beta-One](https://huggingface.co/fancyfeast/llama-joycaption-beta-one-hf-llava)  
- [Diffusers guiders](https://github.com/huggingface/diffusers/pull/11311)  
- [Nunchaku PulID](https://github.com/mit-han-lab/nunchaku/pull/274)  
- [Dream0](https://huggingface.co/ByteDance/DreamO)  
- [Pydantic changes](https://github.com/Cschlaefli/automatic)  

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
- model load: force-reloading entire model as loading transformers only leads to massive memory usage
- model loader: implement model in-memory caching
- modernui: monkey-patch for missing tabs.select event
- modules/lora/lora_extract.py:185:9: W0511: TODO: lora: support pre-quantized flux
- nunchaku: batch support
- nunchaku: cache-dir for transformer and t5 loader
- processing: remove duplicate mask params
- resize image: enable full VAE mode for resize-latent
