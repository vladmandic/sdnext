# TODO

Main ToDo list can be found at [GitHub projects](https://github.com/users/vladmandic/projects)

## Current

## Future Candidates

- Redesign postprocessing  
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
