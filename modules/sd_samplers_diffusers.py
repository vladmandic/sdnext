import os
import re
import copy
import inspect
import diffusers
from modules import shared, errors
from modules.sd_samplers_common import SamplerData, flow_models


debug = os.environ.get('SD_SAMPLER_DEBUG', None) is not None
debug_log = shared.log.trace if debug else lambda *args, **kwargs: None

try:
    from diffusers import (
        CMStochasticIterativeScheduler,
        CosineDPMSolverMultistepScheduler,
        DDIMScheduler,
        DDPMScheduler,
        DEISMultistepScheduler,
        DPMSolverMultistepInverseScheduler,
        DPMSolverMultistepScheduler,
        DPMSolverSDEScheduler,
        DPMSolverSinglestepScheduler,
        EDMDPMSolverMultistepScheduler,
        EDMEulerScheduler,
        EulerAncestralDiscreteScheduler,
        EulerDiscreteScheduler,
        FlowMatchEulerDiscreteScheduler,
        FlowMatchHeunDiscreteScheduler,
        FlowMatchLCMScheduler,
        HeunDiscreteScheduler,
        IPNDMScheduler,
        KDPM2AncestralDiscreteScheduler,
        KDPM2DiscreteScheduler,
        LCMScheduler,
        LMSDiscreteScheduler,
        PNDMScheduler,
        SASolverScheduler,
        UniPCMultistepScheduler,
    )
except Exception as e:
    shared.log.error(f'Sampler import: version={diffusers.__version__} error: {e}')
    if os.environ.get('SD_SAMPLER_DEBUG', None) is not None:
        errors.display(e, 'Samplers')
try:
    from modules.schedulers.scheduler_tcd import TCDScheduler # pylint: disable=ungrouped-imports
    from modules.schedulers.scheduler_tdd import TDDScheduler # pylint: disable=ungrouped-imports
    from modules.schedulers.scheduler_dc import DCSolverMultistepScheduler # pylint: disable=ungrouped-imports
    from modules.schedulers.scheduler_vdm import VDMScheduler # pylint: disable=ungrouped-imports
    from modules.schedulers.scheduler_dpm_flowmatch import FlowMatchDPMSolverMultistepScheduler # pylint: disable=ungrouped-imports
    from modules.schedulers.scheduler_bdia import BDIA_DDIMScheduler # pylint: disable=ungrouped-imports
    from modules.schedulers.scheduler_ufogen import UFOGenScheduler # pylint: disable=ungrouped-imports
    from modules.schedulers.scheduler_unipc_flowmatch import FlowUniPCMultistepScheduler # pylint: disable=ungrouped-imports
    from modules.schedulers.scheduler_flashflow import FlashFlowMatchEulerDiscreteScheduler # pylint: disable=ungrouped-imports
    from modules.perflow import PeRFlowScheduler # pylint: disable=ungrouped-imports
except Exception as e:
    shared.log.error(f'Sampler import: version={diffusers.__version__} error: {e}')
    if os.environ.get('SD_SAMPLER_DEBUG', None) is not None:
        errors.display(e, 'Samplers')

config = {
    # beta_start, beta_end are typically per-scheduler, but we don't want them as they should be taken from the model itself as those are values model was trained on
    # prediction_type is ideally set in model as well, but it maybe needed that we do auto-detect of model type in the future
    'All': { 'num_train_timesteps': 1000, 'beta_start': 0.0001, 'beta_end': 0.02, 'beta_schedule': 'linear', 'prediction_type': 'epsilon' },

    'UniPC': { 'flow_shift': 1, 'predict_x0': True, 'sample_max_value': 1.0, 'solver_order': 2, 'solver_type': 'bh2', 'thresholding': False, 'use_beta_sigmas': False, 'use_exponential_sigmas': False, 'use_flow_sigmas': False, 'use_karras_sigmas': False, 'lower_order_final': True, 'timestep_spacing': 'linspace', 'final_sigmas_type': 'zero', 'rescale_betas_zero_snr': False },
    'DDIM': { 'clip_sample': False, 'set_alpha_to_one': True, 'steps_offset': 0, 'clip_sample_range': 1.0, 'sample_max_value': 1.0, 'timestep_spacing': 'leading', 'rescale_betas_zero_snr': False, 'thresholding': False },

    'Euler': { 'steps_offset': 0, 'interpolation_type': "linear", 'rescale_betas_zero_snr': False, 'final_sigmas_type': 'zero', 'timestep_spacing': 'linspace', 'use_beta_sigmas': False, 'use_exponential_sigmas': False, 'use_karras_sigmas': False },
    'Euler a': { 'steps_offset': 0, 'rescale_betas_zero_snr': False, 'timestep_spacing': 'linspace' },
    'Euler SGM': { 'steps_offset': 0, 'interpolation_type': "linear", 'rescale_betas_zero_snr': False, 'final_sigmas_type': 'zero', 'timestep_spacing': 'trailing', 'use_beta_sigmas': False, 'use_exponential_sigmas': False, 'use_karras_sigmas': False, 'prediction_type': "sample" },
    'Euler EDM': { 'sigma_schedule': "karras" },
    'Euler FlowMatch': { 'timestep_spacing': "linspace", 'shift': 1, 'use_dynamic_shifting': False, 'use_karras_sigmas': False, 'use_exponential_sigmas': False, 'use_beta_sigmas': False },

    'DPM++': { 'thresholding': False, 'sample_max_value': 1.0, 'algorithm_type': "dpmsolver++", 'solver_type': "midpoint", 'lower_order_final': True, 'use_karras_sigmas': False, 'use_exponential_sigmas': False, 'use_flow_sigmas': False, 'use_beta_sigmas': False, 'use_lu_lambdas': False, 'final_sigmas_type': 'zero', 'timestep_spacing': 'linspace', 'solver_order': 1 },
    'DPM++ 2M': { 'thresholding': False, 'sample_max_value': 1.0, 'algorithm_type': "dpmsolver++", 'solver_type': "midpoint", 'lower_order_final': True, 'use_karras_sigmas': False, 'use_exponential_sigmas': False, 'use_flow_sigmas': False, 'use_beta_sigmas': False, 'use_lu_lambdas': False, 'final_sigmas_type': 'zero', 'timestep_spacing': 'linspace', 'solver_order': 2 },
    'DPM++ 3M': { 'thresholding': False, 'sample_max_value': 1.0, 'algorithm_type': "dpmsolver++", 'solver_type': "midpoint", 'lower_order_final': True, 'use_karras_sigmas': False, 'use_exponential_sigmas': False, 'use_flow_sigmas': False, 'use_beta_sigmas': False, 'use_lu_lambdas': False, 'final_sigmas_type': 'zero', 'timestep_spacing': 'linspace', 'solver_order': 3 },
    'DPM++ 1S': { 'solver_order': 2, 'thresholding': False, 'sample_max_value': 1.0, 'algorithm_type': "dpmsolver++", 'solver_type': "midpoint", 'lower_order_final': True, 'use_karras_sigmas': False, 'use_exponential_sigmas': False, 'use_flow_sigmas': False, 'use_beta_sigmas': False, 'final_sigmas_type': 'sigma_min' },
    'DPM++ SDE': { 'thresholding': False, 'sample_max_value': 1.0, 'algorithm_type': "sde-dpmsolver++", 'solver_type': "midpoint", 'lower_order_final': True, 'use_karras_sigmas': False, 'use_exponential_sigmas': False, 'use_flow_sigmas': False, 'use_beta_sigmas': False, 'use_lu_lambdas': False, 'final_sigmas_type': 'zero', 'timestep_spacing': 'linspace', 'solver_order': 1 },
    'DPM++ 2M SDE': { 'thresholding': False, 'sample_max_value': 1.0, 'algorithm_type': "sde-dpmsolver++", 'solver_type': "midpoint", 'lower_order_final': True, 'use_karras_sigmas': False, 'use_exponential_sigmas': False, 'use_flow_sigmas': False, 'use_beta_sigmas': False, 'use_lu_lambdas': False, 'final_sigmas_type': 'zero', 'timestep_spacing': 'linspace', 'solver_order': 2 },
    'DPM++ 2M EDM': { 'solver_order': 2, 'solver_type': 'midpoint', 'final_sigmas_type': 'zero', 'algorithm_type': 'dpmsolver++' },
    'DPM++ Cosine': { 'solver_order': 2, 'sigma_schedule': "exponential", 'prediction_type': "v-prediction" },
    'DPM SDE': { 'use_karras_sigmas': False, 'use_exponential_sigmas': False, 'use_beta_sigmas': False, 'noise_sampler_seed': None, 'timestep_spacing': 'linspace', 'steps_offset': 0,  },

    'DPM++ Inverse': { 'thresholding': False, 'sample_max_value': 1.0, 'algorithm_type': "dpmsolver++", 'solver_type': "midpoint", 'lower_order_final': True, 'use_karras_sigmas': False, 'use_exponential_sigmas': False, 'use_flow_sigmas': False, 'use_beta_sigmas': False, 'use_lu_lambdas': False, 'final_sigmas_type': 'zero', 'timestep_spacing': 'linspace', 'solver_order': 1 },
    'DPM++ 2M Inverse': { 'thresholding': False, 'sample_max_value': 1.0, 'algorithm_type': "dpmsolver++", 'solver_type': "midpoint", 'lower_order_final': True, 'use_karras_sigmas': False, 'use_exponential_sigmas': False, 'use_flow_sigmas': False, 'use_beta_sigmas': False, 'use_lu_lambdas': False, 'final_sigmas_type': 'zero', 'timestep_spacing': 'linspace', 'solver_order': 2 },
    'DPM++ 3M Inverse': { 'thresholding': False, 'sample_max_value': 1.0, 'algorithm_type': "dpmsolver++", 'solver_type': "midpoint", 'lower_order_final': True, 'use_karras_sigmas': False, 'use_exponential_sigmas': False, 'use_flow_sigmas': False, 'use_beta_sigmas': False, 'use_lu_lambdas': False, 'final_sigmas_type': 'zero', 'timestep_spacing': 'linspace', 'solver_order': 3 },

    'UniPC FlowMatch': { 'predict_x0': True, 'sample_max_value': 1.0, 'solver_order': 2, 'solver_type': 'bh2', 'thresholding': False, 'use_beta_sigmas': False, 'use_exponential_sigmas': False, 'use_flow_sigmas': False, 'use_karras_sigmas': False, 'lower_order_final': True, 'timestep_spacing': 'linspace', 'final_sigmas_type': 'zero', 'rescale_betas_zero_snr': False },
    'DPM2 FlowMatch': { 'shift': 1, 'use_dynamic_shifting': False, 'solver_order': 2, 'sigma_schedule': None, 'use_beta_sigmas': False, 'algorithm_type': 'dpmsolver2', 'use_noise_sampler': True, 'beta_start': 0.00085, 'beta_end': 0.012 },
    'DPM2a FlowMatch': { 'shift': 1, 'use_dynamic_shifting': False, 'solver_order': 2, 'sigma_schedule': None, 'use_beta_sigmas': False, 'algorithm_type': 'dpmsolver2A', 'use_noise_sampler': True, 'beta_start': 0.00085, 'beta_end': 0.012 },
    'DPM2++ 2M FlowMatch': { 'shift': 1, 'use_dynamic_shifting': False, 'solver_order': 2, 'sigma_schedule': None, 'use_beta_sigmas': False, 'algorithm_type': 'dpmsolver++2M', 'use_noise_sampler': True, 'beta_start': 0.00085, 'beta_end': 0.012 },
    'DPM2++ 2S FlowMatch': { 'shift': 1, 'use_dynamic_shifting': False, 'solver_order': 2, 'sigma_schedule': None, 'use_beta_sigmas': False, 'algorithm_type': 'dpmsolver++2S', 'use_noise_sampler': True, 'beta_start': 0.00085, 'beta_end': 0.012 },
    'DPM2++ SDE FlowMatch': { 'shift': 1, 'use_dynamic_shifting': False, 'solver_order': 2, 'sigma_schedule': None, 'use_beta_sigmas': False, 'algorithm_type': 'dpmsolver++sde', 'use_noise_sampler': True, 'beta_start': 0.00085, 'beta_end': 0.012 },
    'DPM2++ 2M SDE FlowMatch': { 'shift': 1, 'use_dynamic_shifting': False, 'solver_order': 2, 'sigma_schedule': None, 'use_beta_sigmas': False, 'algorithm_type': 'dpmsolver++2Msde', 'use_noise_sampler': True, 'beta_start': 0.00085, 'beta_end': 0.012 },
    'DPM2++ 3M SDE FlowMatch': { 'shift': 1, 'use_dynamic_shifting': False, 'solver_order': 3, 'sigma_schedule': None, 'use_beta_sigmas': False, 'algorithm_type': 'dpmsolver++3Msde', 'use_noise_sampler': True, 'beta_start': 0.00085, 'beta_end': 0.012 },

    'Heun': { 'use_beta_sigmas': False, 'use_karras_sigmas': False, 'use_exponential_sigmas': False, 'timestep_spacing': 'linspace' },
    'Heun FlowMatch': { 'timestep_spacing': "linspace", 'shift': 1 },
    'LCM FlowMatch': { 'beta_start': 0.00085, 'beta_end': 0.012, 'beta_schedule': "scaled_linear", 'set_alpha_to_one': True, 'rescale_betas_zero_snr': False, 'thresholding': False, 'timestep_spacing': 'linspace' },

    'DEIS': { 'solver_order': 2, 'thresholding': False, 'sample_max_value': 1.0, 'algorithm_type': "deis", 'solver_type': "logrho", 'lower_order_final': True, 'timestep_spacing': 'linspace', 'use_karras_sigmas': False, 'use_exponential_sigmas': False, 'use_flow_sigmas': False, 'use_beta_sigmas': False },
    'SA Solver': {'predictor_order': 2, 'corrector_order': 2, 'thresholding': False, 'lower_order_final': True, 'use_karras_sigmas': False, 'use_flow_sigmas': False, 'use_exponential_sigmas': False, 'use_beta_sigmas': False, 'timestep_spacing': 'linspace'},
    'DC Solver': { 'beta_start': 0.0001, 'beta_end': 0.02, 'solver_order': 2, 'prediction_type': "epsilon", 'thresholding': False, 'solver_type': 'bh2', 'lower_order_final': True, 'dc_order': 2, 'disable_corrector': [0] },
    'VDM Solver': { 'clip_sample_range': 2.0, },
    'TCD': { 'set_alpha_to_one': True, 'rescale_betas_zero_snr': False, 'beta_schedule': 'scaled_linear' },
    'TDD': { },
    'Flash FlowMatch': { 'shift': 1, 'use_dynamic_shifting': False, 'use_karras_sigmas': False, 'use_exponential_sigmas': False, 'use_beta_sigmas': False },
    'PeRFlow': { 'prediction_type': 'ddim_eps' },
    'UFOGen': { },
    'BDIA DDIM': { 'clip_sample': False, 'set_alpha_to_one': True, 'steps_offset': 0, 'clip_sample_range': 1.0, 'sample_max_value': 1.0, 'timestep_spacing': 'leading', 'rescale_betas_zero_snr': False, 'thresholding': False, 'gamma': 1.0 },

    'PNDM': { 'skip_prk_steps': False, 'set_alpha_to_one': False, 'steps_offset': 0, 'timestep_spacing': 'linspace' },
    'IPNDM': { },
    'DDPM': { 'variance_type': "fixed_small", 'clip_sample': False, 'thresholding': False, 'clip_sample_range': 1.0, 'sample_max_value': 1.0, 'timestep_spacing': 'linspace', 'rescale_betas_zero_snr': False },
    'LMSD': { 'use_karras_sigmas': False, 'use_exponential_sigmas': False, 'use_beta_sigmas': False, 'timestep_spacing': 'linspace', 'steps_offset': 0 },
    'KDPM2': { 'use_karras_sigmas': False, 'use_exponential_sigmas': False, 'use_beta_sigmas': False, 'steps_offset': 0, 'timestep_spacing': 'linspace' },
    'KDPM2 a': { 'use_karras_sigmas': False, 'use_exponential_sigmas': False, 'use_beta_sigmas': False, 'steps_offset': 0, 'timestep_spacing': 'linspace' },
    'CMSI': { },
}

samplers_data_diffusers = [
    SamplerData('Default', None, [], {}),

    SamplerData('UniPC', lambda model: DiffusionSampler('UniPC', UniPCMultistepScheduler, model), [], {}),
    SamplerData('DDIM', lambda model: DiffusionSampler('DDIM', DDIMScheduler, model), [], {}),
    SamplerData('Euler', lambda model: DiffusionSampler('Euler', EulerDiscreteScheduler, model), [], {}),
    SamplerData('Euler a', lambda model: DiffusionSampler('Euler a', EulerAncestralDiscreteScheduler, model), [], {}),
    SamplerData('Euler SGM', lambda model: DiffusionSampler('Euler SGM', EulerDiscreteScheduler, model), [], {}),
    SamplerData('Euler EDM', lambda model: DiffusionSampler('Euler EDM', EDMEulerScheduler, model), [], {}),
    SamplerData('Euler FlowMatch', lambda model: DiffusionSampler('Euler FlowMatch', FlowMatchEulerDiscreteScheduler, model), [], {}),

    SamplerData('DPM++', lambda model: DiffusionSampler('DPM++', DPMSolverMultistepScheduler, model), [], {}),
    SamplerData('DPM++ 2M', lambda model: DiffusionSampler('DPM++ 2M', DPMSolverMultistepScheduler, model), [], {}),
    SamplerData('DPM++ 3M', lambda model: DiffusionSampler('DPM++ 3M', DPMSolverMultistepScheduler, model), [], {}),
    SamplerData('DPM++ 1S', lambda model: DiffusionSampler('DPM++ 1S', DPMSolverSinglestepScheduler, model), [], {}),
    SamplerData('DPM++ SDE', lambda model: DiffusionSampler('DPM++ SDE', DPMSolverMultistepScheduler, model), [], {}),
    SamplerData('DPM++ 2M SDE', lambda model: DiffusionSampler('DPM++ 2M SDE', DPMSolverMultistepScheduler, model), [], {}),
    SamplerData('DPM++ 2M EDM', lambda model: DiffusionSampler('DPM++ 2M EDM', EDMDPMSolverMultistepScheduler, model), [], {}),
    SamplerData('DPM++ Cosine', lambda model: DiffusionSampler('DPM++ 2M EDM', CosineDPMSolverMultistepScheduler, model), [], {}),
    SamplerData('DPM SDE', lambda model: DiffusionSampler('DPM SDE', DPMSolverSDEScheduler, model), [], {}),

    SamplerData('DPM++ Inverse', lambda model: DiffusionSampler('DPM++ Inverse', DPMSolverMultistepInverseScheduler, model), [], {}),
    SamplerData('DPM++ 2M Inverse', lambda model: DiffusionSampler('DPM++ 2M Inverse', DPMSolverMultistepInverseScheduler, model), [], {}),
    SamplerData('DPM++ 3M Inverse', lambda model: DiffusionSampler('DPM++ 3M Inverse', DPMSolverMultistepInverseScheduler, model), [], {}),

    SamplerData('UniPC FlowMatch', lambda model: DiffusionSampler('UniPC FlowMatch', FlowUniPCMultistepScheduler, model), [], {}),
    SamplerData('DPM2 FlowMatch', lambda model: DiffusionSampler('DPM2 FlowMatch', FlowMatchDPMSolverMultistepScheduler, model), [], {}),
    SamplerData('DPM2a FlowMatch', lambda model: DiffusionSampler('DPM2a FlowMatch', FlowMatchDPMSolverMultistepScheduler, model), [], {}),
    SamplerData('DPM2++ 2M FlowMatch', lambda model: DiffusionSampler('DPM2++ 2M FlowMatch', FlowMatchDPMSolverMultistepScheduler, model), [], {}),
    SamplerData('DPM2++ 2S FlowMatch', lambda model: DiffusionSampler('DPM2++ 2S FlowMatch', FlowMatchDPMSolverMultistepScheduler, model), [], {}),
    SamplerData('DPM2++ SDE FlowMatch', lambda model: DiffusionSampler('DPM2++ SDE FlowMatch', FlowMatchDPMSolverMultistepScheduler, model), [], {}),
    SamplerData('DPM2++ 2M SDE FlowMatch', lambda model: DiffusionSampler('DPM2++ 2M SDE FlowMatch', FlowMatchDPMSolverMultistepScheduler, model), [], {}),
    SamplerData('DPM2++ 3M SDE FlowMatch', lambda model: DiffusionSampler('DPM2++ 3M SDE FlowMatch', FlowMatchDPMSolverMultistepScheduler, model), [], {}),

    SamplerData('Heun', lambda model: DiffusionSampler('Heun', HeunDiscreteScheduler, model), [], {}),
    SamplerData('Heun FlowMatch', lambda model: DiffusionSampler('Heun FlowMatch', FlowMatchHeunDiscreteScheduler, model), [], {}),
    SamplerData('Flash FlowMatch', lambda model: DiffusionSampler('Flash FlowMatch', FlashFlowMatchEulerDiscreteScheduler, model), [], {}),

    SamplerData('DEIS', lambda model: DiffusionSampler('DEIS', DEISMultistepScheduler, model), [], {}),
    SamplerData('SA Solver', lambda model: DiffusionSampler('SA Solver', SASolverScheduler, model), [], {}),
    SamplerData('DC Solver', lambda model: DiffusionSampler('DC Solver', DCSolverMultistepScheduler, model), [], {}),
    SamplerData('VDM Solver', lambda model: DiffusionSampler('VDM Solver', VDMScheduler, model), [], {}),
    SamplerData('BDIA DDIM', lambda model: DiffusionSampler('BDIA DDIM g=0', BDIA_DDIMScheduler, model), [], {}),

    SamplerData('PNDM', lambda model: DiffusionSampler('PNDM', PNDMScheduler, model), [], {}),
    SamplerData('IPNDM', lambda model: DiffusionSampler('IPNDM', IPNDMScheduler, model), [], {}),
    SamplerData('DDPM', lambda model: DiffusionSampler('DDPM', DDPMScheduler, model), [], {}),
    SamplerData('LMSD', lambda model: DiffusionSampler('LMSD', LMSDiscreteScheduler, model), [], {}),
    SamplerData('KDPM2', lambda model: DiffusionSampler('KDPM2', KDPM2DiscreteScheduler, model), [], {}),
    SamplerData('KDPM2 a', lambda model: DiffusionSampler('KDPM2 a', KDPM2AncestralDiscreteScheduler, model), [], {}),
    SamplerData('CMSI', lambda model: DiffusionSampler('CMSI', CMStochasticIterativeScheduler, model), [], {}),

    SamplerData('LCM', lambda model: DiffusionSampler('LCM', LCMScheduler, model), [], {}),
    SamplerData('LCM FlowMatch', lambda model: DiffusionSampler('LCM FlowMatch', FlowMatchLCMScheduler, model), [], {}),
    SamplerData('TCD', lambda model: DiffusionSampler('TCD', TCDScheduler, model), [], {}),
    SamplerData('TDD', lambda model: DiffusionSampler('TDD', TDDScheduler, model), [], {}),
    SamplerData('PeRFlow', lambda model: DiffusionSampler('PeRFlow', PeRFlowScheduler, model), [], {}),
    SamplerData('UFOGen', lambda model: DiffusionSampler('UFOGen', UFOGenScheduler, model), [], {}),

    SamplerData('Same as primary', None, [], {}),
]


class DiffusionSampler:
    def __init__(self, name, constructor, model, **kwargs):
        if name == 'Default':
            return
        self.name = name
        self.config = {}
        self.sampler = None

        if getattr(model, "default_scheduler", None) is None and (model is not None): # sanity check
            model.default_scheduler = copy.deepcopy(model.scheduler)
        for key, value in config.get('All', {}).items(): # apply global defaults
            self.config[key] = value
        debug_log(f'Sampler: all="{self.config}"')
        if model is None:
            orig_config = {}
        elif hasattr(model.default_scheduler, 'scheduler_config'): # find model defaults
            orig_config = model.default_scheduler.scheduler_config
        else:
            orig_config = model.default_scheduler.config
        debug_log(f'Sampler: diffusers="{self.config}"')
        debug_log(f'Sampler: original="{orig_config}"')
        for key, value in orig_config.items(): # apply model defaults
            if key in self.config:
                self.config[key] = value
        debug_log(f'Sampler: default="{self.config}"')
        for key, value in config.get(name, {}).items(): # apply diffusers per-scheduler defaults
            self.config[key] = value
        for key, value in kwargs.items(): # apply user args, if any
            if key in self.config:
                self.config[key] = value

        # finally apply user preferences
        if shared.opts.schedulers_prediction_type != 'default':
            self.config['prediction_type'] = shared.opts.schedulers_prediction_type
        if shared.opts.schedulers_beta_schedule != 'default':
            if shared.opts.schedulers_beta_schedule == 'linear':
                self.config['beta_schedule'] = 'linear'
            elif shared.opts.schedulers_beta_schedule == 'scaled':
                self.config['beta_schedule'] = 'scaled_linear'
            elif shared.opts.schedulers_beta_schedule == 'cosine':
                self.config['beta_schedule'] = 'squaredcos_cap_v2'
            elif shared.opts.schedulers_beta_schedule == 'sigmoid':
                self.config['beta_schedule'] = 'sigmoid'

        timesteps = re.split(',| ', shared.opts.schedulers_timesteps)
        timesteps = [int(x) for x in timesteps if x.isdigit()]
        if len(timesteps) == 0:
            if 'sigma_schedule' in self.config:
                self.config['sigma_schedule'] = shared.opts.schedulers_sigma if shared.opts.schedulers_sigma != 'default' else None
            if shared.opts.schedulers_sigma == 'default' and shared.sd_model_type in flow_models and 'use_flow_sigmas' in self.config:
                self.config['use_flow_sigmas'] = True
            elif shared.opts.schedulers_sigma == 'betas' and 'use_beta_sigmas' in self.config:
                self.config['use_beta_sigmas'] = True
            elif shared.opts.schedulers_sigma == 'karras' and 'use_karras_sigmas' in self.config:
                self.config['use_karras_sigmas'] = True
            elif shared.opts.schedulers_sigma == 'flowmatch' and 'use_flow_sigmas' in self.config:
                self.config['use_flow_sigmas'] = True
            elif shared.opts.schedulers_sigma == 'exponential' and 'use_exponential_sigmas' in self.config:
                self.config['use_exponential_sigmas'] = True
            elif shared.opts.schedulers_sigma == 'lambdas' and 'use_lu_lambdas' in self.config:
                self.config['use_lu_lambdas'] = True
        else:
            pass # timesteps are set using set_timesteps in set_pipeline_args

        if 'thresholding' in self.config:
            self.config['thresholding'] = shared.opts.schedulers_use_thresholding
        if 'lower_order_final' in self.config:
            self.config['lower_order_final'] = shared.opts.schedulers_use_loworder
        if 'solver_order' in self.config and int(shared.opts.schedulers_solver_order) > 0:
            self.config['solver_order'] = int(shared.opts.schedulers_solver_order)
        if 'predict_x0' in self.config:
            self.config['solver_type'] = shared.opts.uni_pc_variant
        if 'beta_start' in self.config and shared.opts.schedulers_beta_start > 0:
            self.config['beta_start'] = shared.opts.schedulers_beta_start
        if 'beta_end' in self.config and shared.opts.schedulers_beta_end > 0:
            self.config['beta_end'] = shared.opts.schedulers_beta_end
        if 'shift' in self.config:
            self.config['shift'] = shared.opts.schedulers_shift if shared.opts.schedulers_shift > 0 else 3
        if 'flow_shift' in self.config:
            self.config['flow_shift'] = shared.opts.schedulers_shift if shared.opts.schedulers_shift > 0 else 3
        if 'use_dynamic_shifting' in self.config:
            self.config['use_dynamic_shifting'] = True if shared.opts.schedulers_shift == 0 else shared.opts.schedulers_dynamic_shift
        if 'use_beta_sigmas' in self.config and 'sigma_schedule' in self.config:
            self.config['use_beta_sigmas'] = 'StableDiffusion3' in model.__class__.__name__
        if 'rescale_betas_zero_snr' in self.config:
            self.config['rescale_betas_zero_snr'] = shared.opts.schedulers_rescale_betas
        if 'timestep_spacing' in self.config and shared.opts.schedulers_timestep_spacing != 'default' and shared.opts.schedulers_timestep_spacing is not None:
            self.config['timestep_spacing'] = shared.opts.schedulers_timestep_spacing
        if 'num_train_timesteps' in self.config:
            self.config['num_train_timesteps'] = shared.opts.schedulers_timesteps_range
        if 'EDM' in name:
            del self.config['beta_start']
            del self.config['beta_end']
            del self.config['beta_schedule']
        if name in {'IPNDM', 'CMSI', 'VDM Solver'}:
            del self.config['beta_start']
            del self.config['beta_end']
            del self.config['beta_schedule']
            del self.config['prediction_type']
        if 'prediction_type' in self.config and 'Flow' in name:
            self.config['prediction_type'] = 'flow_prediction'
        if 'SGM' in name:
            self.config['timestep_spacing'] = 'trailing'

        # validate all config params
        signature = inspect.signature(constructor, follow_wrapped=True)
        possible = signature.parameters.keys()
        for key in self.config.copy().keys():
            if key not in possible:
                del self.config[key]
        debug_log(f'Sampler: name="{name}"')
        debug_log(f'Sampler: config={self.config}')
        debug_log(f'Sampler: signature={possible}')

        # finally create the new sampler
        try:
            sampler = constructor(**self.config)
        except Exception as e:
            shared.log.error(f'Sampler: "{name}" {e}')
            if debug:
                errors.display(e, 'Samplers')
            self.sampler = None
            return

        if hasattr(sampler, 'set_timesteps'):
            accept_sigmas = "sigmas" in set(inspect.signature(sampler.set_timesteps).parameters.keys())
            accepts_timesteps = "timesteps" in set(inspect.signature(sampler.set_timesteps).parameters.keys())
            accept_scale_noise = hasattr(sampler, "scale_noise")
            debug_log(f'Sampler: "{name}" sigmas={accept_sigmas} timesteps={accepts_timesteps}')
            if ('Flux' in model.__class__.__name__) and (not accept_sigmas):
                shared.log.warning(f'Sampler: "{name}" does not accept sigmas')
                self.sampler = None
                return
            if ('StableDiffusion3' in model.__class__.__name__) and (not accept_scale_noise):
                shared.log.warning(f'Sampler: "{name}" does not implement scale noise')
                self.sampler = None
                return

        # monkey-patch to allow sdxl pipeline to execute flowmatch samplers
        if not hasattr(sampler, 'scale_model_input'):
            sampler.scale_model_input = lambda x, _y: x
        if not hasattr(sampler, 'init_noise_sigma'):
            sampler.init_noise_sigma = 1.0

        self.sampler = sampler

        # shared.log.debug_log(f'Sampler: class="{self.sampler.__class__.__name__}" config={self.sampler.config}')
        self.sampler.name = name
