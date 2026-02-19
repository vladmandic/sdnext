import os


def register_v2(app):
    from modules import shared
    from modules.api.v2.job_queue import job_queue
    from modules.api.v2.routes import router
    from modules.api.v2.ws import ws_job_endpoint

    data_path = os.path.dirname(shared.cmd_opts.config)
    if not data_path:
        data_path = '.'
    job_queue.init(data_path)

    app.include_router(router)
    app.add_api_websocket_route("/sdapi/v2/jobs/{job_id}/ws", ws_job_endpoint)

    shared.log.info('API v2: registered')
