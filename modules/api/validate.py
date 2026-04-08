import re
from modules.logger import log


# value is cost: -1=disabled, 0=unlimited, 1=default, >1 expensive
request_cost = {
    "/file": 0,
    "/run/predict": 0,
    "/sdapi/v1/browser/thumb": 0,
    "/sdapi/v1/network/thumb": 0,
    "/sdapi/v1/txt2img": 5,
    "/sdapi/v1/img2img": 5,
    "/sdapi/v1/control": 5,
}
log_cost = {
    "/file": -1,
    "/token": -1,
    "/theme.css": -1,
    "/sdapi/v1/browser/thumb": -1,
    "/sdapi/v1/network/thumb": -1,
    "/run/predict": -1,
    "/internal/progress": -1,
    "/sdapi/v1/version": -1,
    "/sdapi/v1/log": -1,
    "/sdapi/v1/torch": -1,
    "/sdapi/v1/gpu": -1,
    "/sdapi/v1/memory": -1,
    "/sdapi/v1/platform": -1,
    "/sdapi/v1/checkpoint": -1,
    "/sdapi/v1/status": 60,
    "/sdapi/v1/progress": 60,
}
log_exclude_suffix = ['.css', '.js', '.ico', '.svg']
log_exclude_prefix = ['/assets']

class Limiter():
    def __init__(self, limit):
        import limits
        self.request_backend = limits.storage.MemoryStorage()
        self.request_limit = limit # default is 300 requests per minute
        self.request_strategy = limits.strategies.SlidingWindowCounterRateLimiter(self.request_backend)
        self.request_limiter = limits.parse(f"{self.request_limit}/minute")
        self.log_backend = limits.storage.MemoryStorage()
        self.log_limit = limit // 5 # default is 300/5=60 logs per minute
        self.log_strategy = limits.strategies.FixedWindowRateLimiter(self.log_backend)
        self.log_limiter = limits.parse(f"{self.log_limit}/minute")
        self.summary = {}
        log.info(f'API: limit={self.request_limit} strategy={self.request_strategy.__class__.__name__} backend={self.request_backend.__class__.__name__}')

    def stats(self):
        for k, v in self.summary.items():
            if v > 1:
                log.trace(f'API stats: {k}={v}')

    def check_request(self, client: str, api: str, quiet: bool = False):
        if self.request_limit <= 0:
            return True
        cost = request_cost.get(api, 1)
        if cost < 0:
            return False
        status = self.request_strategy.hit(self.request_limiter, client, api, cost=cost)
        if not status and not quiet:
            log.warning(f'API: client={client} api={api} rate limit exceeded')
            from fastapi.exceptions import HTTPException
            raise HTTPException(status_code=429, detail=f"{client}:{api}: rate limit exceeded")
        return status

    def check_log(self, client: str, api: str):
        if self.log_limit < 0:
            return True
        if any(api.endswith(s) for s in log_exclude_suffix):
            return False
        if any(api.startswith(s) for s in log_exclude_prefix):
            return False
        cost = log_cost.get(api, 1)
        if cost < 0:
            return False
        status = self.log_strategy.hit(self.log_limiter, client, api, cost=cost)
        return status


limiter = Limiter(300)


def get_api_stats():
    limiter.stats()


def validate_request(client, endpoint):
    global limiter # pylint: disable=global-statement
    from modules.shared import opts
    if opts.server_rate_limit != limiter.request_limit:
        limiter = Limiter(opts.server_rate_limit)
    api = re.match(r"^[^?#&=]+", endpoint).group(0)
    key = f"{client}:{api}"
    if key not in limiter.summary:
        limiter.summary[key] = 0
    limiter.summary[key] += 1
    return limiter.check_request(client, api)


def validate_log(client, endpoint):
    api = re.match(r"^[^?#&=]+", endpoint).group(0)
    return limiter.check_log(client, api)
