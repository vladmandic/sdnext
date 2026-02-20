import os
import tempfile


def register_v2(app):
    from modules import shared
    from modules.api.v2.job_queue import job_queue
    from modules.api.v2.routes import router
    from modules.api.v2.upload import upload_router, init_upload_store
    from modules.api.v2.ws import ws_job_endpoint

    data_path = os.path.dirname(shared.cmd_opts.config)
    if not data_path:
        data_path = '.'
    job_queue.init(data_path)

    staging_dir = os.path.join(shared.opts.temp_dir or tempfile.gettempdir(), 'uploads')
    init_upload_store(staging_dir, ttl=1800)

    app.include_router(router)
    app.include_router(upload_router)
    app.add_api_websocket_route("/sdapi/v2/jobs/{job_id}/ws", ws_job_endpoint)

    shared.log.info('API v2: registered')
