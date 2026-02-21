from threading import Lock
from secrets import compare_digest
from fastapi import FastAPI, APIRouter, Depends, Request
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.exceptions import HTTPException
from modules import errors, shared
from modules.logger import log
from modules.api import models, endpoints, script, helpers, server, generate, process, control, docs, gpu


errors.install()


class Api:
    def __init__(self, app: FastAPI, queue_lock: Lock):
        self.credentials = {}
        if shared.cmd_opts.auth:
            for auth in shared.cmd_opts.auth.split(","):
                user, password = auth.split(":")
                self.credentials[user.replace('"', '').strip()] = password.replace('"', '').strip()
        if shared.cmd_opts.auth_file:
            with open(shared.cmd_opts.auth_file, encoding="utf8") as file:
                for line in file.readlines():
                    user, password = line.split(":")
                    self.credentials[user.replace('"', '').strip()] = password.replace('"', '').strip()
        self.router = APIRouter()
        if shared.cmd_opts.docs:
            docs.create_docs(app)
            docs.create_redocs(app)
        self.app = app
        self.queue_lock = queue_lock
        self.generate = generate.APIGenerate(queue_lock)
        self.process = process.APIProcess(queue_lock)
        self.control = control.APIControl(queue_lock)
        # compatibility api
        self.text2imgapi = self.generate.post_text2img
        self.img2imgapi = self.generate.post_img2img

    def register(self):
        # fetch js/css
        self.add_api_route("/js", server.get_js, methods=["GET"], auth=False)
        # server api
        self.add_api_route("/sdapi/v1/motd", server.get_motd, methods=["GET"], response_model=str, tags=["Server"])
        self.add_api_route("/sdapi/v1/log", server.get_log, methods=["GET"], response_model=list[str], tags=["Server"])
        self.add_api_route("/sdapi/v1/log", server.post_log, methods=["POST"], tags=["Server"])
        self.add_api_route("/sdapi/v1/start", self.get_session_start, methods=["GET"], tags=["Server"])
        self.add_api_route("/sdapi/v1/version", server.get_version, methods=["GET"], tags=["Server"])
        self.add_api_route("/sdapi/v1/status", server.get_status, methods=["GET"], response_model=models.ResStatus, tags=["Server"])
        self.add_api_route("/sdapi/v1/platform", server.get_platform, methods=["GET"], tags=["Server"])
        self.add_api_route("/sdapi/v1/progress", server.get_progress, methods=["GET"], response_model=models.ResProgress, tags=["Server"])
        self.add_api_route("/sdapi/v1/history", server.get_history, methods=["GET"], response_model=list[models.ResHistory], tags=["Server"])
        self.add_api_route("/sdapi/v1/interrupt", server.post_interrupt, methods=["POST"], tags=["Server"])
        self.add_api_route("/sdapi/v1/skip", server.post_skip, methods=["POST"], tags=["Server"])
        self.add_api_route("/sdapi/v1/shutdown", server.post_shutdown, methods=["POST"], tags=["Server"])
        self.add_api_route("/sdapi/v1/memory", server.get_memory, methods=["GET"], response_model=models.ResMemory, tags=["Server"])
        self.add_api_route("/sdapi/v1/options", server.get_config, methods=["GET"], response_model=models.OptionsModel, tags=["Server"])
        self.add_api_route("/sdapi/v1/options", server.set_config, methods=["POST"], tags=["Server"])
        self.add_api_route("/sdapi/v1/cmd-flags", server.get_cmd_flags, methods=["GET"], response_model=models.FlagsModel, tags=["Server"])
        self.add_api_route("/sdapi/v1/gpu", gpu.get_gpu_status, methods=["GET"], response_model=list[models.ResGPU], tags=["Server"])
        self.add_api_route("/sdapi/v2/server-info", server.get_server_info, methods=["GET"], tags=["Server"])
        self.add_api_route("/sdapi/v2/options-info", server.get_options_info, methods=["GET"], tags=["Server"])
        self.add_api_route("/sdapi/v2/secrets-status", server.get_secrets_status, methods=["GET"], tags=["Server"])

        # core api using locking
        self.add_api_route("/sdapi/v1/txt2img", self.generate.post_text2img, methods=["POST"], response_model=models.ResTxt2Img, tags=["Generation"])
        self.add_api_route("/sdapi/v1/img2img", self.generate.post_img2img, methods=["POST"], response_model=models.ResImg2Img, tags=["Generation"])
        self.add_api_route("/sdapi/v1/control", self.control.post_control, methods=["POST"], response_model=control.ResControl, tags=["Generation"])
        self.add_api_route("/sdapi/v1/extra-single-image", self.process.extras_single_image_api, methods=["POST"], response_model=models.ResProcessImage, tags=["Processing"])
        self.add_api_route("/sdapi/v1/extra-batch-images", self.process.extras_batch_images_api, methods=["POST"], response_model=models.ResProcessBatch, tags=["Processing"])
        self.add_api_route("/sdapi/v1/preprocess", self.process.post_preprocess, methods=["POST"], tags=["Processing"])
        self.add_api_route("/sdapi/v1/mask", self.process.post_mask, methods=["POST"], tags=["Processing"])
        self.add_api_route("/sdapi/v1/detect", self.process.post_detect, methods=["POST"], tags=["Processing"])
        self.add_api_route("/sdapi/v1/prompt-enhance", self.process.post_prompt_enhance, methods=["POST"], response_model=models.ResPromptEnhance, tags=["Generation"])
        self.add_api_route("/sdapi/v2/prompt-enhance/models", self.process.get_prompt_enhance_models, methods=["GET"])

        # api dealing with optional scripts
        self.add_api_route("/sdapi/v1/scripts", script.get_scripts_list, methods=["GET"], response_model=models.ResScripts, tags=["Scripts"])
        self.add_api_route("/sdapi/v1/script-info", script.get_script_info, methods=["GET"], response_model=list[models.ItemScript], tags=["Scripts"])

        # enumerator api
        self.add_api_route("/sdapi/v1/preprocessors", self.process.get_preprocess, methods=["GET"], response_model=list[process.ItemPreprocess], tags=["Enumerators"])
        self.add_api_route("/sdapi/v1/masking", self.process.get_mask, methods=["GET"], response_model=process.ItemMask, tags=["Enumerators"])
        self.add_api_route("/sdapi/v1/samplers", endpoints.get_samplers, methods=["GET"], response_model=list[models.ItemSampler], tags=["Enumerators"])
        self.add_api_route("/sdapi/v1/upscalers", endpoints.get_upscalers, methods=["GET"], response_model=list[models.ItemUpscaler], tags=["Enumerators"])
        self.add_api_route("/sdapi/v1/sd-models", endpoints.get_sd_models, methods=["GET"], response_model=list[models.ItemModel], tags=["Enumerators"])
        self.add_api_route("/sdapi/v1/controlnets", endpoints.get_controlnets, methods=["GET"], response_model=list[str], tags=["Enumerators"])
        self.add_api_route("/sdapi/v2/control-models", endpoints.get_control_models, methods=["GET"], response_model=list[str], tags=["Enumerators"])
        self.add_api_route("/sdapi/v2/control-modes", endpoints.get_control_modes, methods=["GET"], tags=["Enumerators"])
        self.add_api_route("/sdapi/v2/ip-adapters", endpoints.get_ip_adapters, methods=["GET"], response_model=list[str], tags=["Enumerators"])
        self.add_api_route("/sdapi/v1/face-restorers", endpoints.get_restorers, methods=["GET"], response_model=list[models.ItemDetailer], tags=["Enumerators"])
        self.add_api_route("/sdapi/v1/detailers", endpoints.get_detailers, methods=["GET"], response_model=list[models.ItemDetailer], tags=["Enumerators"])
        self.add_api_route("/sdapi/v1/prompt-styles", endpoints.get_prompt_styles, methods=["GET"], response_model=list[models.ItemStyle], tags=["Enumerators"])
        self.add_api_route("/sdapi/v1/embeddings", endpoints.get_embeddings, methods=["GET"], response_model=models.ResEmbeddings, tags=["Enumerators"])
        self.add_api_route("/sdapi/v1/sd-vae", endpoints.get_sd_vaes, methods=["GET"], response_model=list[models.ItemVae], tags=["Enumerators"])
        self.add_api_route("/sdapi/v1/extensions", endpoints.get_extensions_list, methods=["GET"], response_model=list[models.ItemExtension], tags=["Enumerators"])
        self.add_api_route("/sdapi/v1/extra-networks", endpoints.get_extra_networks, methods=["GET"], response_model=list[models.ItemExtraNetwork], tags=["Enumerators"])
        self.add_api_route("/sdapi/v2/extra-networks/detail", endpoints.get_extra_network_detail, methods=["GET"], response_model=models.ItemExtraNetworkDetail, tags=["Enumerators"])
        self.add_api_route("/sdapi/v2/extra-networks/details", endpoints.get_extra_network_details, methods=["GET"], response_model=models.ResExtraNetworkDetails, tags=["Enumerators"])

        # functional api
        self.add_api_route("/sdapi/v1/png-info", endpoints.post_pnginfo, methods=["POST"], response_model=models.ResImageInfo, tags=["Functional"])
        self.add_api_route("/sdapi/v1/checkpoint", endpoints.get_checkpoint, methods=["GET"], tags=["Functional"])
        self.add_api_route("/sdapi/v1/checkpoint", endpoints.set_checkpoint, methods=["POST"], tags=["Functional"])
        self.add_api_route("/sdapi/v1/refresh-checkpoints", endpoints.post_refresh_checkpoints, methods=["POST"], tags=["Functional"])
        self.add_api_route("/sdapi/v1/unload-checkpoint", endpoints.post_unload_checkpoint, methods=["POST"], tags=["Functional"])
        self.add_api_route("/sdapi/v1/reload-checkpoint", endpoints.post_reload_checkpoint, methods=["POST"], tags=["Functional"])
        self.add_api_route("/sdapi/v1/lock-checkpoint", endpoints.post_lock_checkpoint, methods=["POST"], tags=["Functional"])
        self.add_api_route("/sdapi/v1/refresh-vae", endpoints.post_refresh_vae, methods=["POST"], tags=["Functional"])
        self.add_api_route("/sdapi/v1/latents", endpoints.get_latent_history, methods=["GET"], response_model=list[str], tags=["Functional"])
        self.add_api_route("/sdapi/v1/latents", endpoints.post_latent_history, methods=["POST"], response_model=int, tags=["Functional"])
        self.add_api_route("/sdapi/v1/modules", endpoints.get_modules, methods=["GET"], tags=["Functional"])
        self.add_api_route("/sdapi/v1/sampler", endpoints.get_sampler, methods=["GET"], response_model=dict, tags=["Functional"])

        # caption api
        from modules.api import caption
        caption.register_api()

        # lora api
        from modules.api import loras
        loras.register_api()

        # websocket api
        from modules.api import ws
        ws.register_ws(self.app)

        # gallery api
        from modules.api import gallery
        gallery.register_api(self.app)

        # nudenet api
        from modules.api import nudenet
        nudenet.register_api()

        # xyz-grid api
        from modules.api import xyz_grid
        xyz_grid.register_api()

        # civitai api
        from modules.civitai import api_civitai
        api_civitai.register_api()

        # loaded-models inventory api
        from modules.api import loaded_models
        loaded_models.register_api()

        # model operations api
        from modules.api import models_ops
        models_ops.register_api()

        # system operations api
        from modules.api import system_ops
        system_ops.register_api()

        # v2 async job queue api
        from modules.api.v2 import register_v2
        register_v2(self.app)

        # hide trailing-slash duplicates from OpenAPI schema
        from fastapi.routing import APIRoute
        paths = {r.path for r in self.app.routes if hasattr(r, 'path')}
        for route in self.app.routes:
            if isinstance(route, APIRoute) and len(route.path) > 1 and route.path.endswith('/') and route.path[:-1] in paths:
                route.include_in_schema = False

    def add_api_route(self, path: str, fn, auth: bool = True, **kwargs):
        if auth and self.credentials:
            deps = list(kwargs.get('dependencies', []))
            deps.append(Depends(self.auth))
            kwargs['dependencies'] = deps
        if shared.opts.subpath is not None and len(shared.opts.subpath) > 0:
            self.app.add_api_route(f'{shared.opts.subpath}{path}', endpoint=fn, **kwargs)
        self.app.add_api_route(path, endpoint=fn, **kwargs)

    def auth(self, credentials: HTTPBasicCredentials = Depends(HTTPBasic())):
        if not self.credentials:
            return True
        if credentials.username in self.credentials:
            if compare_digest(credentials.password, self.credentials[credentials.username]):
                return True
            if hasattr(self.app, 'tokens') and (self.app.tokens is not None):
                if credentials.password in self.app.tokens.keys():
                    return True
        log.error(f'API authentication: user="{credentials.username}"')
        raise HTTPException(status_code=401, detail="Unauthorized", headers={"WWW-Authenticate": "Basic"})

    def get_session_start(self, req: Request, agent: str | None = None):
        """Log a new browser session with client IP, authenticated user, and user-agent string."""
        token = req.cookies.get("access-token") or req.cookies.get("access-token-unsecure")
        user = self.app.tokens.get(token) if hasattr(self.app, 'tokens') else None
        log.info(f'Browser session: user={user} client={req.client.host} agent={agent}')
        return {}

    def launch(self):
        config = {
            "listen": shared.cmd_opts.listen,
            "port": shared.cmd_opts.port,
            "keyfile": shared.cmd_opts.tls_keyfile,
            "certfile": shared.cmd_opts.tls_certfile,
            "loop": "auto", # auto, asyncio, uvloop
            "http": "auto", # auto, h11, httptools
        }
        from modules.server import UvicornServer
        http_server = UvicornServer(self.app, **config)
        # from modules.server import HypercornServer
        # server = HypercornServer(self.app, **config)
        http_server.start()
        log.info(f'API server: Uvicorn options={config}')
        return http_server


# compatibility items
decode_base64_to_image = helpers.decode_base64_to_image
encode_pil_to_base64 = helpers.encode_pil_to_base64
validate_sampler_name = helpers.validate_sampler_name
