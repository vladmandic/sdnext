# TODO

Main ToDo list can be found at [GitHub projects](https://github.com/users/vladmandic/projects)

## Current

### Issues/Limitations

- Video: Hunyuan Video I2V: requires `transformers==4.47.1` <https://github.com/huggingface/diffusers/issues/11118>  
- Video: Latte 1 T2V: dtype mismatch <https://github.com/huggingface/diffusers/issues/11137>  
- Video: WAN 2.1 14B I2V 480p/720p: broken offload <https://github.com/huggingface/diffusers/issues/11162>  
- Video: WAN 2.1 14B I2V 480p/720p: custom number of frames <https://github.com/huggingface/diffusers/issues/11163>  
- Video: CogVideoX 1.5 5B T2V/I2V: all-gray output  
- Video: Allegro T2V: all-gray output

## Future Candidates

- Flux: NF4 loader: <https://github.com/huggingface/diffusers/issues/9996>  
- IPAdapter: negative guidance: <https://github.com/huggingface/diffusers/discussions/7167>  
- Control: API enhance scripts compatibility  
- Video: add generate context menu
- Video: FasterCache and PyramidAttentionBroadcast granular config  
- Video: FasterCache and PyramidAttentionBroadcast for LTX and WAN <https://github.com/huggingface/diffusers/issues/11134>  
- Video: API support
- Video: STG: <https://github.com/huggingface/diffusers/blob/main/examples/community/README.md#spatiotemporal-skip-guidance>
- Video SmoothCache: https://github.com/huggingface/diffusers/issues/11135
- FasterCache, PyramidAttentionBroadcast, SmoothCache general support

## Code TODO

- enable ROCm for windows when available
- resize image: enable full VAE mode for resize-latent
- infotext: handle using regex instead
- processing: remove duplicate mask params
- model loader: implement model in-memory caching
- hypertile: vae breaks when using non-standard sizes
- force-reloading entire model as loading transformers only leads to massive memory usage
- add other quantization types
- lora make support quantized flux
- control: support scripts via api
- modernui: monkey-patch for missing tabs.select event
