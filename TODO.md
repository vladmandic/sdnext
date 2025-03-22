# TODO

Main ToDo list can be found at [GitHub projects](https://github.com/users/vladmandic/projects)

## Current

### Issues/Limitations

- VLM Gemma3: requires `transformers==git+https://github.com/huggingface/transformers@v4.49.0-Gemma-3`  
- VAE Remote encode: SD15 and Flux.1 issues: <https://github.com/huggingface/diffusers/issues/11069>  
- Video: API support is TBD  
- Video: Hunyuan Video I2V: transformers incompatibility <https://github.com/huggingface/diffusers/issues/11118>  
- Video: Hunyuan Video I2V: add 16ch vs 33ch processing <https://github.com/huggingface/diffusers/pull/11066>  
- Video: WAN 2.1 14B I2V 480p/720p: broken offload  
- Video: CogVideoX 1.5 5B T2V/I2V: requires pipeline update  
- Video: LTXVideo 0.9.5 T2V/I2V: broken offload  
- Video: LTXVideo 0.9.5 T2V/I2V: requires different params  
- Video: LTXVideo 0.9.5 T2V/I2V: add support for conditioned input  
- Video: LTXVideo 0.9.1 I2V: generator list mismatch  
- Video: Latte 1 T2V: dtype mismatch <https://github.com/huggingface/diffusers/issues/11137>  
- Video: Allegro T2V: all-gray output, requires vae-fp32  
- Video: FasterCache and PyramidAttentionBroadcast granular config  
- Video: FasterCache and PyramidAttentionBroadcast for LTX and WAN <https://github.com/huggingface/diffusers/issues/11134>  

## Future Candidates

- Flux NF4 loader: <https://github.com/huggingface/diffusers/issues/9996>  
- IPAdapter negative: <https://github.com/huggingface/diffusers/discussions/7167>  
- Control API enhance scripts compatibility  

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
