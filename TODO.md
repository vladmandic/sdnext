# TODO

Main ToDo list can be found at [GitHub projects](https://github.com/users/vladmandic/projects)

## Current

- ModernUI for custom model loader  
- ModernUI for history tab  

### Issues/Limitations

N/A

## Future Candidates

- IPAdapter: negative guidance: <https://github.com/huggingface/diffusers/discussions/7167>  
- Control: API enhance scripts compatibility  
- Video: add generate context menu  
- Video: API support  
- Video: STG: <https://github.com/huggingface/diffusers/blob/main/examples/community/README.md#spatiotemporal-skip-guidance>  
- Video: SmoothCache: https://github.com/huggingface/diffusers/issues/11135  

## Code TODO

> pnpm lint | grep W0511 | awk -F'TODO ' '{print "- "$NF}' | sed 's/ (fixme)//g'
 
- install: enable ROCm for windows when available
- resize image: enable full VAE mode for resize-latent
- infotext: handle using regex instead
- fc: autodetect tensor format based on model
- fc: autodetect distilled based on model
- processing: remove duplicate mask params
- model loader: implement model in-memory caching
- custom: load receipe
- custom: save receipe
- hypertile: vae breaks when using non-standard sizes
- model load: force-reloading entire model as loading transformers only leads to massive memory usage
- lora: add other quantization types
- lora: maybe force imediate quantization
- modules/lora/lora_extract.py:185:9: W0511: TODO: lora support pre-quantized flux
- control: support scripts via api
- modernui: monkey-patch for missing tabs.select event
- nunchaku: cache-dir for transformer and t5 loader
- nunchaku: batch support
- nunchaku: LoRA support
