import json
from starlette.responses import HTMLResponse
from fastapi import FastAPI
from fastapi.openapi.docs import get_redoc_html, swagger_ui_default_parameters
from fastapi.encoders import jsonable_encoder


def get_swagger_ui_html(*,
                        openapi_url: str,
                        title: str,
                        swagger_js_url: str = "https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui-bundle.js",
                        swagger_css_url: str = "https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui.css",
                        swagger_extra_css_url: str = None,
                        swagger_favicon_url: str = "https://fastapi.tiangolo.com/img/favicon.png",
                        oauth2_redirect_url: str = None,
                        init_oauth: dict = None,
                        swagger_ui_parameters: dict = None,
                       ) -> HTMLResponse:
    current_swagger_ui_parameters = swagger_ui_default_parameters.copy()
    if swagger_ui_parameters:
        current_swagger_ui_parameters.update(swagger_ui_parameters)
    html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <link type="text/css" rel="stylesheet" href="{swagger_css_url}">
            <link rel="shortcut icon" href="{swagger_favicon_url}">
            <title>{title}</title>
        </head>
        <body>
            <div id="swagger-ui"></div>
            <script src="{swagger_js_url}"></script>
            <script>
                const ui = SwaggerUIBundle({{
                    url: '{openapi_url}',
    """
    if swagger_extra_css_url is not None:
        html = html.replace('</head>', f'<link type="text/css" rel="stylesheet" href="{swagger_extra_css_url}"></head>')
    for key, value in current_swagger_ui_parameters.items():
        html += f"{json.dumps(key)}: {json.dumps(jsonable_encoder(value))},\n"
    if oauth2_redirect_url:
        html += f"oauth2RedirectUrl: window.location.origin + '{oauth2_redirect_url}',"
    html += """
        presets: [
            SwaggerUIBundle.presets.apis,
            SwaggerUIBundle.SwaggerUIStandalonePreset
            ],
        })"""
    if init_oauth:
        html += f"ui.initOAuth({json.dumps(jsonable_encoder(init_oauth))})"
    html += """
        </script>
        </body>
        </html>
        """
    return HTMLResponse(html)


def create_docs(app: FastAPI):
    swagger_ui_parameters = {
        "displayOperationId": True,
        "layout": "BaseLayout",
        "showExtensions": True,
        "showCommonExtensions": True,
        "deepLinking": False,
        "dom_id": "#swagger-ui",
    }

    @app.get("/docs", include_in_schema=True)
    async def custom_swagger_html():
        res = get_swagger_ui_html(
            title=f'{app.title}: Swagger UI',
            openapi_url=app.openapi_url,
            swagger_favicon_url='/file=html/favicon.svg',
            swagger_css_url='/file=html/swagger.css',
            swagger_ui_parameters=swagger_ui_parameters,
            # swagger_extra_css_url='file=html/swagger.css',
        )
        # res = inject_css(html.content, 'html/swagger.css')
        return res


def create_redocs(app: FastAPI):
    @app.get("/redocs", include_in_schema=True)
    async def custom_redoc_html():
        res = get_redoc_html(
            title=f'{app.title}: ReDoc',
            openapi_url=app.openapi_url,
            redoc_favicon_url='/file=html/favicon.svg',
        )
        return res
