from fastapi.exceptions import HTTPException


def get_lora(lora: str) -> dict:
    """Return full details for a single LoRA network by name. Returns 404 if not found."""
    from modules.lora import lora_load
    if lora not in lora_load.available_networks:
        raise HTTPException(status_code=404, detail=f"Lora '{lora}' not found")
    obj = lora_load.available_networks[lora]
    return obj.__dict__


def get_loras():
    """List all available LoRA networks with name, alias, path, and metadata."""
    from modules.lora import network, lora_load
    def create_lora_json(obj: network.NetworkOnDisk):
        return { "name": obj.name, "alias": obj.alias, "path": obj.filename, "metadata": obj.metadata }
    return [create_lora_json(obj) for obj in lora_load.available_networks.values()]


def post_refresh_loras():
    """Rescan LoRA directories and update the available networks list."""
    from modules.lora import lora_load
    result = lora_load.list_available_networks()
    _invalidate_extra_networks()
    return result


def _invalidate_extra_networks():
    """Reset extra-networks page caches so the v2 API picks up changes."""
    from modules import shared
    for page in shared.extra_networks:
        page.refresh_time = 0
        page.create_items('txt2img')


def register_api(app):
    app.add_api_route("/sdapi/v1/lora", get_lora, methods=["GET"], response_model=dict, tags=["Enumerators"])
    app.add_api_route("/sdapi/v1/loras", get_loras, methods=["GET"], response_model=list[dict], tags=["Enumerators"])
    app.add_api_route("/sdapi/v1/refresh-loras", post_refresh_loras, methods=["POST"], tags=["Functional"])
