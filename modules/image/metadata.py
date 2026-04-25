import io
import os
import re
import json
import piexif
from PIL import Image, ExifTags
from modules import sd_samplers
from modules.logger import log
from modules.image.watermark import get_watermark


debug = log.trace if os.environ.get("SD_METADATA_DEBUG", None) is not None else lambda *args, **kwargs: None


def safe_decode_string(s: bytes):
    remove_prefix = lambda text, prefix: text[len(prefix):] if text.startswith(prefix) else text  # pylint: disable=unnecessary-lambda-assignment
    s = remove_prefix(s, b'UNICODE')
    s = remove_prefix(s, b'ASCII')
    s = remove_prefix(s, b'\x00')
    for encoding in ["utf-16-le", "utf-16-be", "utf-8", "utf-16", "ascii", "latin_1", "cp1252", "cp437"]: # try different encodings
        try:
            if encoding == "utf-16-le":
                if not (len(s) >= 2 and len(s) % 2 == 0 and all(b == 0 for b in s[1 : min(len(s), 20) : 2])): # not utf-16-le
                    continue
            val = s.decode(encoding, errors="strict")
            val = re.sub(r'[\x00-\x09]', '', val).strip() # remove remaining special characters
            if len(val) == 0: # remove empty strings
                val = None
            debug(f'Metadata: decode="{val}" encoding="{encoding}"')
            return val
        except Exception:
            pass
    return None


def parse_comfy_metadata(data: dict):
    def parse_workflow():
        res = ''
        try:
            txt = data.get('workflow', {})
            dct = json.loads(txt)
            nodes = len(dct.get('nodes', []))
            version = dct.get('extra', {}).get('frontendVersion', 'unknown')
            if version is not None:
                res = f" | Version: {version} | Nodes: {nodes}"
        except Exception:
            pass
        return res

    def parse_prompt():
        res = ''
        try:
            txt = data.get('prompt', {})
            dct = json.loads(txt)
            for val in dct.values():
                inp = val.get('inputs', {})
                if 'model' in inp:
                    model = inp.get('model', None)
                    if isinstance(model, str) and len(model) > 0:
                        res += f" | Model: {model} | Class: {val.get('class_type', '')}"
        except Exception:
            pass
        return res

    workflow = parse_workflow()
    prompt = parse_prompt()
    if len(workflow) > 0 or len(prompt) > 0:
        parsed = f'App: ComfyUI{workflow}{prompt}'
        log.info(f'Image metadata: {parsed}')
        debug(f'Metadata: comfy="{parsed}"')
        return parsed
    return ''


def parse_invoke_metadata(data: dict):
    def parse_metadtaa():
        res = ''
        try:
            txt = data.get('invokeai_metadata', {})
            dct = json.loads(txt)
            if 'app_version' in dct:
                version = dct['app_version']
                if isinstance(version, str) and len(version) > 0:
                    res += f" | Version: {version}"
        except Exception:
            pass
        return res

    metadata = parse_metadtaa()
    if len(metadata) > 0:
        parsed = f'App: InvokeAI{metadata}'
        log.info(f'Image metadata: {parsed}')
        debug(f'Metadata: invoke="{parsed}"')
        return parsed
    return ''


def parse_novelai_metadata(data: dict):
    geninfo = ''
    if data.get("Software", None) == "NovelAI":
        try:
            dct = json.loads(data["Comment"])
            sampler = sd_samplers.samplers_map.get(dct["sampler"], "Euler a")
            geninfo = f'{data["Description"]} Negative prompt: {dct["uc"]} Steps: {dct["steps"]}, Sampler: {sampler}, CFG scale: {dct["scale"]}, Seed: {dct["seed"]}, Clip skip: 2, ENSD: 31337'
            debug(f'Metadata: novelai="{geninfo}"')
            return geninfo
        except Exception:
            pass
    return ''


def parse_xmp_metadata(data: dict):
    # Extract XMP dc:subject tags into a readable field
    geninfo = ''
    xmp_raw = data.get("xmp")
    if xmp_raw and isinstance(xmp_raw, (str, bytes)):
        xmp_str = xmp_raw if isinstance(xmp_raw, str) else xmp_raw.decode("utf-8", errors="replace")
        xmp_tags = re.findall(r"<rdf:li>([^<]+)</rdf:li>", xmp_str)
        if xmp_tags:
            geninfo = f"XMP Tags: {', '.join(xmp_tags)}"
            debug(f'Metadata: xmp="{geninfo}"')
            return geninfo
    return ''


def read_info_from_image(image: Image.Image, watermark: bool = False) -> tuple[str, dict]:
    if image is None:
        return '', {}
    if isinstance(image, str):
        try:
            image = Image.open(image)
            image.load()
        except Exception:
            return '', {}
    items = dict(image.info) if image.info else {}  # copy so popping doesn't mutate the source image's info dict
    geninfo = items.pop('parameters', None) or items.pop('UserComment', None) or ''
    if isinstance(geninfo, dict):
        if 'UserComment' in geninfo:
            geninfo = geninfo['UserComment'] # Info was nested
        else:
            geninfo = '' # Unknown format. Ignore contents
        items['UserComment'] = geninfo

    if "exif" in items:
        try:
            exif = piexif.load(items["exif"])
        except Exception as e:
            log.error(f'Error loading EXIF data: {e}')
            exif = {}
        for _key, subkey in exif.items():
            debug(f'Metadata EXIF: key="{_key}" subkey="{subkey}" type="{type(subkey)}"')
            if isinstance(subkey, dict):
                for key, val in subkey.items():
                    if isinstance(val, bytes): # decode bytestring
                        val = safe_decode_string(val)
                    if isinstance(val, tuple) and isinstance(val[0], int) and isinstance(val[1], int) and val[1] > 0: # convert camera ratios
                        val = round(val[0] / val[1], 2)
                    if val is not None and key in ExifTags.TAGS: # add known tags
                        if ExifTags.TAGS[key] == 'UserComment': # add geninfo from UserComment
                            geninfo = str(val)
                            items['parameters'] = val
                        else:
                            items[ExifTags.TAGS[key]] = val
                    elif val is not None and key in ExifTags.GPSTAGS:
                        items[ExifTags.GPSTAGS[key]] = val

    if watermark:
        wm = get_watermark(image)
        if wm != '':
            debug(f'Metadata: watermark="{wm}"')
            # geninfo += f' Watermark: {wm}'
            items['watermark'] = wm

    for key, val in items.items():
        if isinstance(val, bytes): # decode bytestring
            items[key] = safe_decode_string(val)
            debug(f'Metadata: key="{key}" value="{items[key]}"')

    geninfo += parse_comfy_metadata(items)
    geninfo += parse_invoke_metadata(items)
    geninfo += parse_novelai_metadata(items)
    geninfo += parse_xmp_metadata(items)

    for key in ['exif', 'ExifOffset', 'JpegIFOffset', 'JpegIFByteCount', 'ExifVersion', 'icc_profile', 'jfif', 'jfif_version', 'jfif_unit', 'jfif_density', 'adobe', 'photoshop', 'loop', 'duration', 'dpi', 'xmp']: # remove unwanted tags
        items.pop(key, None)

    if geninfo and 'parameters' not in items: # restore so callers re-stamping items preserve original generation params
        items['parameters'] = geninfo

    debug(f'Metadata geninfoi: "{geninfo}"')
    debug(f'Metadata items: "{items}"')
    return geninfo, items


def image_data(data):
    import gradio as gr
    if data is None:
        return gr.update(), None
    err1 = None
    err2 = None
    try:
        image = Image.open(io.BytesIO(data))
        image.load()
        info, _ = read_info_from_image(image)
        log.debug(f'Decoded object: image={image} metadata={info}')
        return info, None
    except Exception as e:
        err1 = e
    try:
        if len(data) > 1024 * 10:
            log.warning(f'Error decoding object: data too long: {len(data)}')
            return gr.update(), None
        info = data.decode('utf8')
        log.debug(f'Decoded object: data={len(data)} metadata={info}')
        return info, None
    except Exception as e:
        err2 = e
    log.error(f'Error decoding object: {err1 or err2}')
    return gr.update(), None
