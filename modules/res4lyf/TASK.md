# TASK: Schedulers

## Notes

This is a codebase for diffusion schedulers implemented for `diffusers` library and ported from `res4lyf` repository at <https://github.com/ClownsharkBatwing/RES4LYF>

Ported schedulers codebase is in `modules/res4lyf`, do not modify any other files  

## Testing  

Current focus is on following code-paths:
- using `epsilon` prediction type  
- using `StableDiffusionXLPipeline` pipeline for *text2image*  

## Results

- *ETDRKScheduler, LawsonScheduler, ABNorsettScheduler, RESSinglestepScheduler, RESSinglestepSDEScheduler, PECScheduler, etc.*:  
  do NOT modify behavior and codebase for these schedulers as they produce good outputs under all circumstances  
  if needed, you can use them as gold-standard references to compare other schedulers against  
- *RESUnifiedScheduler*, *DEISMultistepScheduler, RESMultistepScheduler*
  work fine with `rk_type=res_2s`, `rk_type=deis_1s` and similar single-step params,  
  but with `rk_type=res_2m`, `rk_type=deis_2m` and similar multi-step params  
  image looks fine in early steps, but then degrages at the final steps with what looks like too much noise
