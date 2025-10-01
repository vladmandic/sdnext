const guiders = {
  None: '',
  'LSC: LayerSkipConfig': 'https://github.com/huggingface/diffusers/blob/041501aea92919c9c7f36e189fc9cf7d865ebb96/src/diffusers/hooks/layer_skip.py#L41',
  'CFG: ClassifierFreeGuidance': 'https://huggingface.co/docs/diffusers/v0.35.1/en/api/modular_diffusers/guiders#diffusers.ClassifierFreeGuidance',
  'Auto: AutoGuidance': 'https://huggingface.co/docs/diffusers/v0.35.1/en/api/modular_diffusers/guiders#diffusers.AutoGuidance',
  'Zero: ClassifierFreeZeroStar': 'https://huggingface.co/docs/diffusers/v0.35.1/en/api/modular_diffusers/guiders#diffusers.ClassifierFreeZeroStarGuidance',
  'PAG: PerturbedAttentionGuidance': 'https://huggingface.co/docs/diffusers/v0.35.1/en/api/modular_diffusers/guiders#diffusers.PerturbedAttentionGuidance',
  'APG: AdaptiveProjectedGuidance': 'https://huggingface.co/docs/diffusers/v0.35.1/en/api/modular_diffusers/guiders#diffusers.AdaptiveProjectedGuidance',
  'SLG: SkipLayerGuidance': 'https://huggingface.co/docs/diffusers/v0.35.1/en/api/modular_diffusers/guiders#diffusers.SkipLayerGuidance',
  'SEG: SmoothedEnergyGuidance': 'https://huggingface.co/docs/diffusers/v0.35.1/en/api/modular_diffusers/guiders#diffusers.SmoothedEnergyGuidance',
  'TCFG: TangentialClassifierFreeGuidance': 'https://huggingface.co/docs/diffusers/v0.35.1/en/api/modular_diffusers/guiders#diffusers.TangentialClassifierFreeGuidance',
  'FDG: FrequencyDecoupledGuidance': 'https://huggingface.co/docs/diffusers/v0.35.1/en/api/modular_diffusers/guiders#diffusers.FrequencyDecoupledGuidance',
};

function getGuidanceDocs(guider) {
  if (guider.label) guider = guider.label;
  const url = guiders[guider];
  log('getGuidanceDocs', guider, url);
  if (url) window.open(url, '_blank');
}
