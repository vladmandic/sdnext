import ssl
import time
import logging
from asyncio.exceptions import CancelledError
import anyio
import starlette
import uvicorn
import fastapi
from starlette.responses import JSONResponse
from fastapi import FastAPI, Request, Response
from fastapi.exceptions import HTTPException
from fastapi.encoders import jsonable_encoder
from modules.logger import log
import modules.errors as errors


ssl._create_default_https_context = ssl._create_unverified_context # pylint: disable=protected-access
errors.install()
ignore_endpoints = [
    '/sdapi/v1/version',
    '/sdapi/v1/log',
    '/sdapi/v1/gpu',
    '/sdapi/v1/network/thumb',
    '/sdapi/v1/progress',
    '/sdapi/v1/memory',
    '/sdapi/v1/status',
    '/sdapi/v1/sd-models',
    '/sdapi/v1/options',
    '/sdapi/v1/checkpoint',
    '/sdapi/v1/prompt-styles',
    '/sdapi/v2/browser',
    '/sdapi/v2/loaded-models',
    '/sdapi/v2/server-info',
    '/sdapi/v2/memory',
    '/sdapi/v2/gpu',
    '/sdapi/v2/system-info',
    '/sdapi/v2/sd-vae',
    '/sdapi/v2/upscalers',
    '/sdapi/v2/prompt-styles',
    '/sdapi/v2/options',
    '/sdapi/v2/log',
]


def setup_middleware(app: FastAPI, cmd_opts):
    uvicorn_logger=logging.getLogger("uvicorn.error")
    uvicorn_logger.disabled = True
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.middleware.gzip import GZipMiddleware
    app.user_middleware = [x for x in app.user_middleware if x.cls.__name__ != 'CORSMiddleware']
    app.middleware_stack = None # reset current middleware to allow modifying user provided list
    app.add_middleware(GZipMiddleware, minimum_size=2048)
    cors_origins = ["http://localhost:5173", "http://127.0.0.1:5173"]
    cors_regex = None
    if cmd_opts.cors_origins:
        cors_origins += [o.strip() for o in cmd_opts.cors_origins.split(",")]
    if cmd_opts.cors_regex:
        cors_regex = cmd_opts.cors_regex
    if cors_regex:
        app.add_middleware(CORSMiddleware, allow_origins=cors_origins, allow_origin_regex=cors_regex, allow_methods=['*'], allow_credentials=True, allow_headers=['*'])
    else:
        app.add_middleware(CORSMiddleware, allow_origins=cors_origins, allow_methods=['*'], allow_credentials=True, allow_headers=['*'])

    rate_limited_paths = {'/sdapi/v1/txt2img', '/sdapi/v1/img2img', '/sdapi/v1/control', '/sdapi/v2/jobs'}

    @app.middleware("http")
    async def log_and_time(req: Request, call_next):
        try:
            ts = time.time()
            endpoint = req.scope.get('path', 'err')
            if cmd_opts.api_rate_limit and endpoint in rate_limited_paths and req.method == 'POST':
                from modules.api.security import generation_limiter
                client_ip = req.client.host if req.client else "unknown"
                if not generation_limiter.is_allowed(client_ip):
                    return JSONResponse(status_code=429, content={"error": "Rate limit exceeded"})
            res: Response = await call_next(req)
            duration = str(round(time.time() - ts, 4))
            res.headers["X-Process-Time"] = duration
            token = req.cookies.get("access-token") or req.cookies.get("access-token-unsecure")
            if (cmd_opts.api_log) and endpoint.startswith('/sdapi'):
                if any([endpoint.startswith(x) for x in ignore_endpoints]): # noqa C419 # pylint: disable=use-a-generator
                    return res
                log.info('API user={user} code={code} {prot}/{ver} {method} {endpoint} {cli} {duration}'.format( # pylint: disable=consider-using-f-string, logging-format-interpolation
                    user = app.tokens.get(token) if hasattr(app, 'tokens') else None,
                    code = res.status_code,
                    ver = req.scope.get('http_version', '0.0'),
                    cli = req.scope.get('client', ('0:0.0.0', 0))[0],
                    prot = req.scope.get('scheme', 'err'),
                    method = req.scope.get('method', 'err'),
                    endpoint = endpoint,
                    duration = duration,
                ))
            return res
        except CancelledError:
            log.warning('WebSocket closed (ignore asyncio.exceptions.CancelledError)')
        except BaseException as e:
            return handle_exception(req, e)

    def handle_exception(req: Request, e: Exception):
        err = {
            "error": type(e).__name__,
            "code": vars(e).get('status_code', 500),
            "detail": vars(e).get('detail', ''),
            "body": vars(e).get('body', ''),
            "errors": str(e),
        }
        if err['code'] == 401 and 'file=' in req.url.path: # dont spam with unauth
            return JSONResponse(status_code=err['code'], content=jsonable_encoder(err))
        if err['code'] == 404 and 'file=html/' in req.url.path: # dont spam with locales
            return JSONResponse(status_code=err['code'], content=jsonable_encoder(err))

        if not any([req.url.path.endswith(x) for x in ignore_endpoints]): # noqa C419 # pylint: disable=use-a-generator
            log.error(f"API error: {req.method}: {req.url} {err}")

        if not isinstance(e, HTTPException) and err['error'] != 'TypeError': # do not print backtrace on known httpexceptions
            errors.display(e, 'HTTP API', [anyio, fastapi, uvicorn, starlette])
        elif err['code'] in [404, 401, 400]:
            pass
        else:
            log.debug(e, exc_info=True) # print stack trace
        return JSONResponse(status_code=err['code'], content=jsonable_encoder(err))

    @app.exception_handler(HTTPException)
    async def http_exception_handler(req: Request, e: HTTPException):
        return handle_exception(req, e)

    @app.exception_handler(Exception)
    async def general_exception_handler(req: Request, e: Exception):
        if isinstance(e, TypeError):
            return JSONResponse(status_code=500, content=jsonable_encoder(str(e)))
        else:
            return handle_exception(req, e)

    app.build_middleware_stack() # rebuild middleware stack on-the-fly
    log.debug(f'API middleware: {[m.cls for m in app.user_middleware]}')
