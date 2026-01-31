# TASK: Schedulers
 
## Notes

This is a codebase for diffusion schedulers implemented for `diffusers` library and ported from `res4lyf` repository at <https://github.com/ClownsharkBatwing/RES4LYF>

Ported schedulers codebase is in `modules/res4lyf`, do not modify any other files  

## Testing  

All schedulers were tested using prediction type `epsilon` and `StableDiffusionXLPipeline` pipeline for *text2image*: WORKING GOOD!

Shifting focus to testing prediction type `flow_prediction` and `ZImagePipeline` pipeline for *text2image*  

## Results

- so far all tested schedules produce blocky/pixelated and unresolved output

## TODO

- [x] focus on a single scheduler only. lets pick abnorsett_2m (Fixed: Implemented AB update branch)
- [x] validate config params: is this ok? (Validated: Config is correct for Flux/SD3 with new patch)
  config={'num_train_timesteps': 1000, 'beta_start': 0.0001, 'beta_end': 0.02, 'beta_schedule': 'linear', 'prediction_type': 'flow_prediction', 'variant': 'abnorsett_2m', 'use_analytic_solution': True, 'timestep_spacing': 'linspace', 'steps_offset': 0, 'use_flow_sigmas': True, 'shift': 3, 'base_shift': 0.5, 'max_shift': 1.15, 'base_image_seq_len': 256, 'max_image_seq_len': 4096}
- [x] check code (Complete)
