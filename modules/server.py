import threading
import logging
import time
import asyncio
import uvicorn
import fastapi
from modules.logger import log


class UvicornServer(uvicorn.Server):
    def __init__(self, app: fastapi.FastAPI, host = None, listen = None, port = None, keyfile = None, certfile = None, loop = "auto", http = "auto"):
        self.app: fastapi.FastAPI = app
        self.thread: threading.Thread = None
        self.loop = None
        self.wants_restart = False
        self.should_exit = False
        kwargs = {
            'loop': loop, # auto, asyncio, uvloop
            'http': http, # auto, h11, httptools
            'interface': "auto", # auto, asgi3, asgi2, wsgi
            'ws': "auto", # auto, websockets, wsproto, websockets-sansio
            'timeout_keep_alive': 60, # default=5
            'ws_max_size': 1024 * 1024 * 1024,  # default 16MB
            'ws_max_queue': 64, # default=32
            'ws_ping_interval': 30, # default=20
            'ws_ping_timeout': 60, # default=20
            'timeout_graceful_shutdown': 5, # default=None
            'access_log': False, # default=True
            'server_header': False, # default=True
            'date_header': False, # default=True
            'backlog': 4096, # default=2048
            'reload': False, # default=False
        }
        self.config = uvicorn.Config(
            app=self.app,
            host = host or ("0.0.0.0" if listen else "127.0.0.1"),
            port = port or 7860,
            log_level = logging.WARNING,
            ssl_keyfile = keyfile,
            ssl_certfile = certfile,
            **kwargs
        )
        super().__init__(config=self.config)
        log.info(f'Server: uvicorn={kwargs}')

    def start(self):
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.wants_restart = False
        self.thread.start()
        start = time.time()
        while not self.started:
            time.sleep(1e-3)
            if time.time() - start > 5:
                raise RuntimeError("Server failed to start. Please check that the port is available.")
        policy = asyncio.get_event_loop_policy()
        self.loop = f"{type(policy).__module__}.{type(policy).__name__}"

    def stop(self):
        self.should_exit = True
        self.thread.join()

    def restart(self):
        self.wants_restart = True
        self.stop()
        self.start()


class HypercornServer:
    def __init__(self, app: fastapi.FastAPI, listen = None, port = None, keyfile = None, certfile = None, loop = "auto", http = None):
        import hypercorn
        self.app: fastapi.FastAPI = app
        self.server: HypercornServer = None
        self.thread = None
        self.task = None
        self.wants_restart = False
        self.loop = 'trio' if loop == 'auto' else loop # asyncio, uvloop, trio
        self.config = hypercorn.config.Config()
        self.config.bind = [f'{"0.0.0.0" if listen else "127.0.0.1"}:{port or 7861}']
        self.config.keyfile = keyfile
        self.config.certfile = certfile
        self.config.keep_alive_timeout = 60 # default=5
        self.config.backlog = 4096 # default=100
        self.config.loglevel = "WARNING"
        self.config.max_app_queue_size = 64 # default=10
        self.http = http # unused

    def run(self):
        import trio
        from hypercorn.trio import serve
        self.server = trio.run(serve, self.app, self.config)

    def start(self):
        if self.loop == 'trio':
            self.thread = threading.Thread(target=self.run, daemon=True)
            self.thread.start()
        elif self.loop == 'asyncio': # does not run in thread
            from hypercorn.asyncio import serve
            self.server = serve(self.app, self.config)
            asyncio.run(self.server)
        elif self.loop == 'uvloop': # does not run in thread
            import uvloop
            from hypercorn.asyncio import serve
            uvloop.install()
            from hypercorn.asyncio import serve
            self.server = serve(self.app, self.config)
            asyncio.run(self.server)
