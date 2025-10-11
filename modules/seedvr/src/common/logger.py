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

"""
Logging utility functions.
"""

import logging
import sys
from typing import Optional

from .distributed import get_global_rank, get_local_rank, get_world_size

_default_handler = logging.StreamHandler(sys.stdout)
_default_handler.setFormatter(
    logging.Formatter(
        "%(asctime)s "
        + (f"[Rank:{get_global_rank()}]" if get_world_size() > 1 else "")
        + (f"[LocalRank:{get_local_rank()}]" if get_world_size() > 1 else "")
        + "[%(threadName).12s][%(name)s][%(levelname).5s] "
        + "%(message)s"
    )
)


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get a logger.
    """
    logger = logging.getLogger(name)
    logger.addHandler(_default_handler)
    logger.setLevel(logging.INFO)
    return logger
