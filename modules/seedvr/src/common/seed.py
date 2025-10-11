# // Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# //
# // Licensed under the Apache License, Version 2.0 (the "License");
# // you may not use this file except in compliance with the License.
# // You may obtain a copy of the License at
# //
# //     http://www.apache.org/licenses/LICENSE-2.0
# //
# // Unless required by applicable law or agreed to in writing, software
# // distributed under the License is distributed on an "AS IS" BASIS,
# // WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# // See the License for the specific language governing permissions and
# // limitations under the License.

import random
from typing import Optional
import numpy as np
import torch

from .distributed import get_global_rank


def set_seed(seed: Optional[int], same_across_ranks: bool = False):
    """Function that sets the seed for pseudo-random number generators."""
    if seed is not None:
        seed += get_global_rank() if not same_across_ranks else 0
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

