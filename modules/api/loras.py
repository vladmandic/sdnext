from fastapi.exceptions import HTTPException


def get_lora(lora: str) -> dict:
    from modules.lora import lora_load
    if lora not in lora_load.available_networks:
        raise HTTPException(status_code=404, detail=f"Lora '{lora}' not found")
    obj = lora_load.available_networks[lora]
    obj.info = obj.get_info()
    obj.desc = obj.get_desc()
    print('HERE', obj)
    return obj.__dict__

def get_loras():
    from modules.lora import network, lora_load
    def create_lora_json(obj: network.NetworkOnDisk):
        return { "name": obj.name, "alias": obj.alias, "path": obj.filename, "metadata": obj.metadata }
    return [create_lora_json(obj) for obj in lora_load.available_networks.values()]

def post_refresh_loras():
    from modules.lora import lora_load
    return lora_load.list_available_networks()
