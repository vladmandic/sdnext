import os
import tempfile


def register_v2(app, dependencies=None):
    from modules import shared
    from modules.api.v2.job_queue import job_queue
    from modules.api.v2.routes import router
    from modules.api.v2.upload import upload_router, init_upload_store
    from modules.api.v2.ws import ws_job_endpoint

    deps = dependencies or []

    data_path = os.path.dirname(shared.cmd_opts.config)
    if not data_path:
        data_path = '.'
    job_queue.init(data_path)

    staging_dir = os.path.join(shared.opts.temp_dir or tempfile.gettempdir(), 'uploads')
    init_upload_store(staging_dir, ttl=1800)

    app.include_router(router, dependencies=deps)
    app.include_router(upload_router, dependencies=deps)
    app.add_api_websocket_route("/sdapi/v2/jobs/{job_id}/ws", ws_job_endpoint)

    from modules.api.v2.endpoints import router as endpoints_router
    app.include_router(endpoints_router, dependencies=deps)

    from modules.api.v2.server import router as server_router
    app.include_router(server_router, dependencies=deps)

    from modules.api.v2.caption import router as caption_router
    app.include_router(caption_router, dependencies=deps)

    from modules.api.v2.prompt_enhance import router as prompt_enhance_router
    app.include_router(prompt_enhance_router, dependencies=deps)

    from modules.logger import log
    log.info('API v2: registered')
