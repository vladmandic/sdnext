# RES4LYF SCHEDULERS

- Schedulers codebase is in `modules/res4lyf`  
  do not modify any other files  
- Testing notes:  
  - using `epsilon` prediction type  
  - using `StableDiffusionXLPipeline` pipeline for text2image  
  - using `StableDiffusionXLInpaintPipeline` for inpainting and image2image  
- *ETDRKScheduler, LawsonScheduler, ABNorsettScheduler, RESSinglestepSDEScheduler, PECScheduler*:  
  produce good outputs under all circumstances  
- *LinearRKScheduler, LobattoScheduler, RadauIIAScheduler, GaussLegendreScheduler, SpecializedRKScheduler, RungeKuttaScheduler*:  
  work well for text2image, but then in image2image it produces pure black image  
- *RESUnifiedScheduler*:  
  works fine with `rk_type=res_2s` and similar single-step params,  
  but produces too much noise with `rk_type=res_2m` and similar multi-step params  
- *DEISMultistepScheduler, RESMultistepScheduler* have the same problem  
  while *RESSinglestepScheduler* works fine  
- *CommonSigmaScheduler, LangevinDynamicsScheduler*:  
  do not work with `epsilon` prediction type, results in pure noise  
