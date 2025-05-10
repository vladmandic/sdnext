# TODO

Main ToDo list can be found at [GitHub projects](https://github.com/users/vladmandic/projects)

## Current

- [Diffusers guiders](https://github.com/huggingface/diffusers/pull/11311)
- [Nunchaku PulID](https://github.com/mit-han-lab/nunchaku/pull/274)
- Video: API support  
- ModernUI for Custom model loader  
- ModernUI for History tab  
- ModernUI for FramePack

### Issues/Limitations

N/A

## Future Candidates

- Control: API enhance scripts compatibility  
- IPAdapter: negative guidance: <https://github.com/huggingface/diffusers/discussions/7167>  
- Video: STG: <https://github.com/huggingface/diffusers/blob/main/examples/community/README.md#spatiotemporal-skip-guidance>  
- Video: SmoothCache: https://github.com/huggingface/diffusers/issues/11135  

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
- lora: maybe force imediate quantization
- lora: add t5 key support for sd35/f16
- lora: support pre-quantized flux
- model load: force-reloading entire model as loading transformers only leads to massive memory usage
- model loader: implement model in-memory caching
- modernui: monkey-patch for missing tabs.select event
- nunchaku: batch support
- nunchaku: cache-dir for transformer and t5 loader
- processing: remove duplicate mask params
- resize image: enable full VAE mode for resize-latent
  