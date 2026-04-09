from fastapi import Body
from modules.api import api


def nudenet_censor(
    image: str = Body("", title='NudeNet input image'),
    score: float = Body(0.2, title='nudenet threshold score'),
    blocks: int = Body(3, title='nudenet pixelation blocks'),
    censor: list = Body([], title='nudenet censorship items'),
    method: str = Body('pixelate', title='nudenet censorship method'),
    overlay: str = Body('', title='nudenet overlay image path'),
):
    """Detect and censor NSFW regions in an image using NudeNet. Returns detections and optionally censored image."""
    from scripts.nudenet import nudenet # pylint: disable=no-name-in-module
    base64image = image
    image = api.decode_base64_to_image(image)
    if nudenet.detector is None:
        nudenet.detector = nudenet.NudeDetector() # loads and initializes model once
    nudes = nudenet.detector.censor(image=image, method=method, min_score=score, censor=censor, blocks=blocks, overlay=overlay)
    if len(censor) > 0: # replace image if anything is censored
        base64image = api.encode_pil_to_base64(nudes.output).decode("utf-8")
    detections_dict = { d["label"]: d["score"] for d in nudes.detections }
    return { "image": base64image, "detections": detections_dict }


def prompt_check(
    prompt: str = Body("", title='prompt text'),
    lang: str = Body("eng", title='allowed languages'),
    alphabet: str = Body("latn", title='allowed alphabets'),
):
    """Check prompt language and alphabet against allowed values. Returns detected language and pass/fail flags."""
    from scripts.nudenet import langdetect # pylint: disable=no-name-in-module
    res = langdetect.lang_detect(prompt)
    res = ','.join(res) if isinstance(res, list) else res
    lang = [a.strip() for a in lang.split(',')] if lang else []
    alphabet = [a.strip() for a in alphabet.split(',')] if alphabet else []
    lang_ok = any(a in res for a in lang) if len(lang) > 0 else True
    alph_ok = any(a in res for a in alphabet) if len(alphabet) > 0 else True
    return { "lang": res, "lang_ok": lang_ok, "alph_ok": alph_ok }


def image_guard(
    image: str = Body("", title='input image'),
    policy: str = Body("", title='optional policy definition'),
    model: str = Body("", title='optional policy model name'),
):
    """Evaluate an image against a content policy using the ImageGuard classifier."""
    from scripts.nudenet import imageguard # pylint: disable=no-name-in-module
    image = api.decode_base64_to_image(image)
    res = imageguard.image_guard(image=image, policy=policy, model_name=model)
    return res


def banned_words(
    words: str = Body("", title='comma separated list of banned words'),
    prompt: str = Body("", title='prompt text'),
):
    """Check a prompt against a comma-separated list of banned words. Returns any matches found."""
    from scripts.nudenet import bannedwords # pylint: disable=no-name-in-module
    found = bannedwords.check_banned(words=words, prompt=prompt)
    return found


def register_api(app):
    app.add_api_route("/sdapi/v1/nudenet", nudenet_censor, methods=["POST"], response_model=dict, tags=["Processing"])
    app.add_api_route("/sdapi/v1/prompt-lang", prompt_check, methods=["POST"], response_model=dict, tags=["Processing"])
    app.add_api_route("/sdapi/v1/image-guard", image_guard, methods=["POST"], response_model=dict, tags=["Processing"])
    app.add_api_route("/sdapi/v1/prompt-banned", banned_words, methods=["POST"], response_model=list, tags=["Processing"])
