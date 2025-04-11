# TODO

Main ToDo list can be found at [GitHub projects](https://github.com/users/vladmandic/projects)

## Current

- HiDream: configurable LLM

### Issues/Limitations

N/A

## Future Candidates

- Flux: NF4 loader: <https://github.com/huggingface/diffusers/issues/9996>  
- IPAdapter: negative guidance: <https://github.com/huggingface/diffusers/discussions/7167>  
- Control: API enhance scripts compatibility  
- Video: add generate context menu  
- Video: API support  
- Video: STG: <https://github.com/huggingface/diffusers/blob/main/examples/community/README.md#spatiotemporal-skip-guidance>  
- Video: SmoothCache: https://github.com/huggingface/diffusers/issues/11135  
- TeaCache: https://github.com/ali-vilab/TeaCache

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
