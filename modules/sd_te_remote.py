import os
import time
import json
import torch
import requests
from modules import devices, errors
from modules.logger import log


def get_t5_prompt_embeds(
    prompt: str | list[str] | None = None,
    num_images_per_prompt: int = 1, # pylint: disable=unused-argument
    max_sequence_length: int = 512, # pylint: disable=unused-argument
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
):
    device = device or devices.device
    dtype = dtype or devices.dtype
    url = os.environ.get('SD_REMOTE_T5', None)
    if url is None:
        log.error('Remote-TE: url is not set')
        return None
    try:
        t0 = time.time()
        response = requests.post(
            url=url,
            headers={ "Content-Type": "application/json" },
            json=prompt,
            timeout=300,
        )
        t1 = time.time()
        shape = json.loads(response.headers["shape"])
        buffer = bytearray(response.content)
        tensor = torch.frombuffer(buffer, dtype=dtype).reshape(shape)
        log.debug(f'Remote-TE: url="{url}" prompt="{prompt}" shape={shape} time={t1-t0:.3f}')
        return tensor.to(device=device, dtype=dtype)
    except Exception as e:
        log.error(f'Remote-TE: {e}')
        errors.display(e, 'remote-te')
        return None
