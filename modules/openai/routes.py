import time
from typing import Optional
from fastapi import Request, HTTPException, Depends
from fastapi.security import HTTPAuthorizationCredentials
from modules.logger import log
from .generate import execute_generation, enforce_rolling_context
from .helpers import get_prompt_template
from .classes import Stats, Req


def setup_routes(self):
    """Exposes standard endpoints and safely routes traffic requests."""

    async def verify_api_key(credentials: Optional[HTTPAuthorizationCredentials] = Depends(self._security)):
        if not self.api_key:
            return
        if credentials is None or credentials.credentials != self.api_key:
            raise HTTPException(
                status_code=401,
                detail="Invalid or missing API key in Authorization header."
            )

    @self.app.get("/health")
    async def health_check():
        return {"status": "healthy", "timestamp": time.time()}

    @self.app.get("/v1/models", dependencies=[Depends(verify_api_key)])
    async def list_models():
        model_name = getattr(self.model.config, "_name_or_path", "local-transformer")
        return {
            "object": "list",
            "data": [{
                "id": model_name,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "transformers"
            }]
        }

    @self.app.get("/v1/models/{model_id}", dependencies=[Depends(verify_api_key)])
    async def retrieve_model(model_id: str):
        model_name = getattr(self.model.config, "_name_or_path", "local-transformer")
        if model_id != model_name:
            raise HTTPException(
                status_code=404,
                detail=f"Model '{model_id}' not found. Active model is '{model_name}'."
            )
        return {
            "id": model_name,
            "object": "model",
            "created": int(time.time()),
            "owned_by": "transformers"
        }

    @self.app.post("/v1/completions", dependencies=[Depends(verify_api_key)], tags=["production_hardened"])
    async def text_completions(request: Request):
        json_body = await request.json()
        prompt_str = json_body.get("prompt", "")
        if isinstance(prompt_str, list):
            prompt_str = prompt_str[0] if prompt_str else ""
        temperature = float(json_body.get("temperature", self.config.temperature))
        config = {
            "max_new_tokens": json_body.get("max_tokens", self.config.max_new_tokens),
            "temperature": temperature,
            "top_p": float(json_body.get("top_p", self.config.top_p)),
            "top_k": int(json_body.get("top_k", self.config.top_k)),
            "repetition_penalty": float(json_body.get("frequency_penalty", self.config.repetition_penalty)),
            "do_sample": True if temperature > 0.0 else False
        }
        stats = Stats()
        stats.id = id(request)
        stats.prompt = len(prompt_str)

        req = Req(id=id(request), config=config, client=request.scope.get('client', ('0:0.0.0', 0))[0], url=request.scope.get('path', 'err'))
        log.debug(f"OpenAI: {req}")

        return await execute_generation(
            self,
            stats=stats,
            request=request,
            prompt=prompt_str,
            config=config,
            stream=json_body.get("stream", self.config.stream),
            stream_options=json_body.get("stream_options", None),
            images=None
        )

    @self.app.post("/v1/chat/completions", dependencies=[Depends(verify_api_key)], tags=["production_hardened"])
    async def chat_completions(request: Request):
        json_body = await request.json()
        messages = json_body.get("messages", [])
        tools = json_body.get("tools", None)
        stream_options = json_body.get("stream_options", None)
        images = json_body.get("images", None)

        temperature = float(json_body.get("temperature", self.config.temperature))
        config = {
            "max_new_tokens": json_body.get("max_tokens", self.config.max_new_tokens),
            "temperature": temperature,
            "top_p": float(json_body.get("top_p", self.config.top_p)),
            "top_k": int(json_body.get("top_k", self.config.top_k)),
            "repetition_penalty": float(json_body.get("frequency_penalty", self.config.repetition_penalty)),
            "do_sample": True if temperature > 0.0 else False
        }

        sanitized_messages = enforce_rolling_context(self, messages)
        try:
            prompt_str = get_prompt_template(self, sanitized_messages, tools)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"LLM: template execution failure: {str(e)}") from e

        stats = Stats()
        stats.id = id(request)
        stats.messages = len(messages)
        stats.prompt = len(prompt_str)

        req = Req(id=id(request), config=config, client=request.scope.get('client', ('0:0.0.0', 0))[0], url=request.scope.get('path', 'err'))
        log.debug(f"OpenAI: {req}")

        return await execute_generation(
            self,
            stats=stats,
            request=request,
            prompt=prompt_str,
            config=config,
            stream=json_body.get("stream", self.config.stream),
            stream_options=stream_options,
            images=images
        )
