from fastapi import Body


async def post_rembg(
    input_image: str = Body("", title='rembg input image'),
    model: str = Body("u2net", title='rembg model'),
    return_mask: bool = Body(False, title='return mask'),
    alpha_matting: bool = Body(False, title='alpha matting'),
    alpha_matting_foreground_threshold: int = Body(240, title='alpha matting foreground threshold'),
    alpha_matting_background_threshold: int = Body(10, title='alpha matting background threshold'),
    alpha_matting_erode_size: int = Body(10, title='alpha matting erode size'),
    refine: bool = Body(False, title="refine foreground (ben2 only)")
):
    if not model or model == "None":
        return {}

    from modules.api import api
    input_image = api.decode_base64_to_image(input_image)
    if input_image is None:
        return {}

    if model == "ben2":
        from modules.rembg import ben2
        image = ben2.remove(input_image, refine=refine)
    else:
        from installer import install
        for pkg in ["dctorch==0.1.2", "pymatting", "pooch", "rembg"]:
            install(pkg, no_deps=True, ignore=False)
        import rembg
        image = rembg.remove(
            input_image,
            session=rembg.new_session(model),
            only_mask=return_mask,
            alpha_matting=alpha_matting,
            alpha_matting_foreground_threshold=alpha_matting_foreground_threshold,
            alpha_matting_background_threshold=alpha_matting_background_threshold,
            alpha_matting_erode_size=alpha_matting_erode_size,
        )
    return {"image": api.encode_pil_to_base64(image).decode("utf-8")}


def register_api(app):
    print('HERE')
    app.add_api_route("/sdapi/v1/rembg", post_rembg, methods=["POST"], tags=["REMBG"])
