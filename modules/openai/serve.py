import threading
from typing import Optional
import uvicorn
from fastapi import FastAPI
from fastapi.security import HTTPBearer
from modules.logger import log
from .classes import Model, Config
from .helpers import get_attention_config
from .routes import setup_routes


class OpenAIServer:
    def __init__(
        self,
        model,
        tokenizer,
        processor=None,
        host: str = "127.0.0.1",
        port: int = 8888,
        server: Optional[uvicorn.Server] = None,
        max_context_tokens: Optional[int] = None,
        max_new_tokens: Optional[int] = None,
        stream: Optional[bool] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
        api_key: Optional[str] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.processor = processor
        self.host = host
        self.port = port
        self.model_info = Model(
            name=getattr(model.config, "_name_or_path", "local-transformer"),
            cls=model.__class__.__name__,
            tokenizer=tokenizer.__class__.__name__,
            processor=processor.__class__.__name__ if processor else None,
            type=getattr(model.config, "model_type", None)
        )
        self.config = Config(
            max_context_tokens=max_context_tokens if max_context_tokens is not None else 4096,
            max_new_tokens=max_new_tokens if max_new_tokens is not None else 512,
            stream=stream if stream is not None else False,
            temperature=temperature if temperature is not None else 0.2,
            top_p=top_p if top_p is not None else 0.9,
            top_k=top_k if top_k is not None else 50,
            repetition_penalty=repetition_penalty if repetition_penalty is not None else 1.
        )
        log.info(f"OpenAI: {self.model_info}")
        attention = get_attention_config(self)
        log.debug(f'OpenAI: {attention}')

        if server:
            self.use_server = True
            self.app = server
            self.server = server
            self._is_running = True
        else:
            self.use_server = False
            self.app = FastAPI(title="SD.Next OpenAI-compatible LLM Server", version="1.0")
            self.server: Optional[uvicorn.Server] = None
            self._is_running = False
        self._startup_event = threading.Event()
        self.api_key = api_key
        self._lock = threading.Lock()
        self._security = HTTPBearer(auto_error=False)
        self.thread: Optional[threading.Thread] = None
        setup_routes(self)

    def start(self, timeout_seconds: float = 10.0):
        """Spawns the serving interface safely using an active background thread worker."""
        if self._is_running:
            # log.warning("OpenAI: Server('already running')")
            return
        with self._lock:
            if self.server is None:
                self._startup_event.clear()
                config = uvicorn.Config(app=self.app, host=self.host, port=self.port, log_level="info", loop="asyncio", workers=1)
                self.server = uvicorn.Server(config)
                self.server.install_signal_handlers = lambda *args, **kwargs: None
                original_startup = self.server.startup

                async def patched_startup(*args, **kwargs):
                    await original_startup(*args, **kwargs)
                    self._startup_event.set()

                self.server.startup = patched_startup
                self.thread = threading.Thread(target=self.server.run, name="TransformersServeWorkerThread", daemon=True)
                self.thread.start()
            if not self._startup_event.wait(timeout=timeout_seconds):
                self.stop()
                raise TimeoutError("OpenAI: init timeout")
        self._is_running = True
        url = f"http://{self.host}:{self.port}/v1"
        log.info(f"OpenAI: Server(url={url})")

    def stop(self, timeout_seconds: float = 5.0):
        """Safely winds down network sockets and detached worker threads."""
        if self.use_server:
            return
        with self._lock:
            if not self._is_running or not self.server:
                return
            self.server.should_exit = True
            if self.thread and self.thread.is_alive():
                self.thread.join(timeout=timeout_seconds)
            self.server = None
            self.thread = None
            self.server = None
            self.thread = None
            self._is_running = False
            log.info("OpenAI: Server(None)")
