import re
import limits
from fastapi.exceptions import HTTPException
from modules.logger import log


requests_summary = {}
request_cost = {  # value is cost, 0=not rate limited, 1=default, >1 more expensive
    "/file": 0,
    "/run/predict": 0,
    "/sdapi/v1/browser/thumb": 0,
    "/sdapi/v1/network/thumb": 0,
    "/sdapi/v2/browser/thumb": 0,
    "/sdapi/v2/network/thumb": 0,
    "/sdapi/v1/txt2img": 5,
    "/sdapi/v1/img2img": 5,
    "/sdapi/v1/control": 5,
    "/sdapi/v2/txt2img": 5,
    "/sdapi/v2/img2img": 5,
    "/sdapi/v2/control": 5,
}
backend = limits.storage.MemoryStorage()
strategy = limits.strategies.SlidingWindowCounterRateLimiter(backend)
limiter = limits.parse("300/minute")


def get_stats():
    for k, v in requests_summary.items():
        if v > 1:
            log.trace(f'API stats: {k}={v}')


def rate_limit(key):
    cost = request_cost.get(key, 1)
    if not strategy.hit(limiter, key, cost=cost):
        log.warning(f'API: key={key} rate limit exceeded')
        raise HTTPException(status_code=429, detail=f'{key}: rate limit exceeded')


def validate_request(client, endpoint):
    api = re.match(r"^[^?#&=]+", endpoint).group(0)
    key = f"{client}:{api}"
    if key not in requests_summary:
        requests_summary[key] = 0
    requests_summary[key] += 1
    rate_limit(key)
