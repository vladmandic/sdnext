# TODO

Main ToDo list can be found at [GitHub projects](https://github.com/users/vladmandic/projects)

## Current

### Issues/Limitations

- Video: Hunyuan Video I2V: requires `transformers==4.47.1` <https://github.com/huggingface/diffusers/issues/11118>  
- Video: Latte 1 T2V: dtype mismatch <https://github.com/huggingface/diffusers/issues/11137>  
- Video: CogVideoX 1.5 5B T2V/I2V: all-gray output  
- Video: Allegro T2V: all-gray output

## Future Candidates

- Flux: NF4 loader: <https://github.com/huggingface/diffusers/issues/9996>  
- IPAdapter: negative guidance: <https://github.com/huggingface/diffusers/discussions/7167>  
- Control: API enhance scripts compatibility  
- Video: add generate context menu  
- Video: API support  
- Video: STG: <https://github.com/huggingface/diffusers/blob/main/examples/community/README.md#spatiotemporal-skip-guidance>  
- Video: SmoothCache: https://github.com/huggingface/diffusers/issues/11135  
- SoftFill: https://github.com/zacheryvaughn/softfill-pipelines

## Code TODO

- control: support scripts via api
- enable ROCm for windows when available
- fc: autodetect distilled based on model
- fc: autodetect tensor format based on model
- hypertile: vae breaks when using non-standard sizes
- infotext: handle using regex instead
- lora: add other quantization types
- lora: force-reloading entire model as loading transformers only leads to massive memory usage
- lora: required for flux to reapply offload after lora has been applied, but fails with oom
- lora: support pre-quantized flux
- model loader: implement model in-memory caching
- modernui: monkey-patch for missing tabs.select event
- processing: remove duplicate mask params
- resize image: enable full VAE mode for resize-latent
