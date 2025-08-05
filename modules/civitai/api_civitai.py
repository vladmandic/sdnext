from starlette.responses import JSONResponse


def models_to_json(all_models:list, model_id:int=None):
    dct = []
    for model in all_models:
        if model_id is not None and model.id != model_id:
            continue
        model_dct = model.__dict__.copy()
        versions_dct = []
        for version in model.versions:
            version_dct = version.__dict__.copy()
            version_dct['files'] = [f.__dict__.copy() for f in version.files]
            version_dct['images'] = [i.__dict__.copy() for i in version.images]
            versions_dct.append(version_dct)
        model_dct['versions'] = versions_dct
        dct.append(model_dct)
    # obj = json.dumps(dct, indent=2, ensure_ascii=False)
    return dct


def get_civitai(
        model_id:int=None, # if model_id is provided assume fetch-from-cache
        query:str = '', # search query or tag is required
        tag:str = '', # search query or tag is required
        types:str = '', # Checkpoint, TextualInversion, Hypernetwork, AestheticGradient, LORA, Controlnet, Poses
        sort:str = '', # Highest Rated, Most Downloaded, Newest
        period:str = '', # AllTime, Year, Month, Week, Day
        nsfw:bool = None, # optional:bool
        limit:int = 0,
        base:str = '',
        token:str = None,
        exact:bool = True,
):
    from modules.civitai import search_civitai
    if model_id is not None:
        dct = models_to_json(search_civitai.models, model_id=model_id)
        return JSONResponse(content=dct, status_code=200)
    if len(query) > 0 or len(tag) > 0:
        models = search_civitai.search_civitai(
            query=query,
            tag=tag,
            types=types,
            sort=sort,
            period=period,
            nsfw=nsfw,
            limit=limit,
            base=base,
            token=token,
            exact=exact
        )
        dct = models_to_json(models)
        return JSONResponse(content=dct, status_code=200)
    return JSONResponse(content=[], status_code=200)


def register_api():
    from modules.shared import api
    api.add_api_route("/sdapi/v1/civitai", get_civitai, methods=["GET"], response_model=list)
