_orig_unipc_set_timesteps = None


def init_samplers_hijack():
    global _orig_unipc_set_timesteps # pylint: disable=global-statement

    from diffusers import UniPCMultistepScheduler
    _orig_unipc_set_timesteps = UniPCMultistepScheduler.set_timesteps

    def _unipc_set_timesteps_device_fix(self, num_inference_steps=None, device=None, **kwargs):
        _orig_unipc_set_timesteps(self, num_inference_steps=num_inference_steps, device=device, **kwargs)
        if device is not None:
            self.sigmas = self.sigmas.to(device)

    UniPCMultistepScheduler.set_timesteps = _unipc_set_timesteps_device_fix
