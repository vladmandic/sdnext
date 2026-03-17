import re
from modules.logger import log


request_cost = {  # value is cost, 0=not rate limited, 1=default, >1 more expensive
    "/file": 0,
    "/run/predict": 0,
    "/sdapi/v1/browser/thumb": 0,
    "/sdapi/v1/network/thumb": 0,
    "/sdapi/v1/txt2img": 5,
    "/sdapi/v1/img2img": 5,
    "/sdapi/v1/control": 5,
}


class Limiter():
    def __init__(self, limit):
        import limits
        self.limit = limit
        self.backend = limits.storage.MemoryStorage()
        self.strategy = limits.strategies.SlidingWindowCounterRateLimiter(self.backend)
        self.limiter = limits.parse(f'{self.limit}/minute')
        self.summary = {}
        log.info(f'API: limit={self.limit} strategy={self.strategy.__class__.__name__} backend={self.backend.__class__.__name__}')

    def stats(self):
        for k, v in self.summary.items():
            if v > 1:
                log.trace(f'API stats: {k}={v}')

    def check(self, client: str, api: str, quiet: bool = False):
        if self.limit <= 0:
            return True
        cost = request_cost.get(api, 1)
        status = self.strategy.hit(self.limiter, client, api, cost=cost)
        if not status and not quiet:
            log.warning(f'API: client={client} api={api} rate limit exceeded')
            from fastapi.exceptions import HTTPException
            raise HTTPException(status_code=429, detail=f"{client}:{api}: rate limit exceeded")
        return status


limiter = Limiter(300)


def get_api_stats():
    limiter.stats()


def validate_request(client, endpoint):
    global limiter # pylint: disable=global-statement
    from modules.shared import opts
    if opts.server_rate_limit != limiter.limit:
        limiter = Limiter(opts.server_rate_limit)
    api = re.match(r"^[^?#&=]+", endpoint).group(0)
    key = f"{client}:{api}"
    if key not in limiter.summary:
        limiter.summary[key] = 0
    limiter.summary[key] += 1
    return limiter.check(client, api)
