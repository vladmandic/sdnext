from fastapi import Body
from modules.api import api


def nudenet_censor(
    image: str = Body("", title='nudenet input image'),
    score: float = Body(0.2, title='nudenet threshold score'),
    blocks: int = Body(3, title='nudenet pixelation blocks'),
    censor: list = Body([], title='nudenet censorship items'),
    method: str = Body('pixelate', title='nudenet censorship method'),
    overlay: str = Body('', title='nudenet overlay image path'),
):
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
):
    from scripts.nudenet import imageguard # pylint: disable=no-name-in-module
    image = api.decode_base64_to_image(image)
    res = imageguard.image_guard(image=image, policy=policy)
    return res


def banned_words(
    words: str = Body("", title='comma separated list of banned words'),
    prompt: str = Body("", title='prompt text'),
):
    from scripts.nudenet import bannedwords # pylint: disable=no-name-in-module
    found = bannedwords.check_banned(words=words, prompt=prompt)
    return found


def register_api():
    from modules.shared import api as api_instance
    api_instance.add_api_route("/sdapi/v1/nudenet", nudenet_censor, methods=["POST"], response_model=dict)
    api_instance.add_api_route("/sdapi/v1/prompt-lang", prompt_check, methods=["POST"], response_model=dict)
    api_instance.add_api_route("/sdapi/v1/image-guard", image_guard, methods=["POST"], response_model=dict)
    api_instance.add_api_route("/sdapi/v1/prompt-banned", banned_words, methods=["POST"], response_model=list)
