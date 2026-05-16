from __future__ import annotations

"""Sharpfin color management (ICC profile handling).

Vendored from https://github.com/drhead/sharpfin (Apache 2.0)
"""

from io import BytesIO
from typing import TYPE_CHECKING, Any, cast
from warnings import warn

import numpy as np

from PIL import Image
from PIL.ImageCms import (
    Direction, Intent, ImageCmsProfile, PyCMSError,
    createProfile, getDefaultIntent, isIntentSupported, profileToProfile
)
from PIL.ImageCms import Flags as cms_flags
from PIL.ImageOps import exif_transpose

if TYPE_CHECKING:
    from PIL import ImageFile
    from torch import Tensor

Image.MAX_IMAGE_PIXELS = None

_SRGB = createProfile(colorSpace='sRGB')

_INTENT_FLAGS = {
    Intent.PERCEPTUAL: cms_flags.HIGHRESPRECALC,
    Intent.RELATIVE_COLORIMETRIC: (
        cms_flags.HIGHRESPRECALC |
        cms_flags.BLACKPOINTCOMPENSATION
    ),
    Intent.ABSOLUTE_COLORIMETRIC: cms_flags.HIGHRESPRECALC
}

class CMSWarning(UserWarning):
    def __init__(
        self,
        message: str,
        *,
        path: str | None = None,
        cms_info: dict[str, Any] | None = None,
        cause: Exception | None = None,
    ):
        super().__init__(message)
        self.__cause__ = cause

        self.path = path
        self.cms_info = cms_info

def _coalesce_intent(intent: Intent | int) -> Intent:
    if isinstance(intent, Intent):
        return intent

    match intent:
        case 0:
            return Intent.PERCEPTUAL
        case 1:
            return Intent.RELATIVE_COLORIMETRIC
        case 2:
            return Intent.SATURATION
        case 3:
            return Intent.ABSOLUTE_COLORIMETRIC
        case _:
            raise ValueError("invalid intent")

def _add_info(info: dict[str, Any], source: object, key: str) -> None:
    try:
        if (value := getattr(source, key, None)) is not None:
            info[key] = value
    except Exception:
        pass

def apply_srgb(img: Image.Image | ImageFile.ImageFile) -> Image.Image:
    path = getattr(img, "filename", "")

    try:
        img.load()

        try:
            exif_transpose(img, in_place=True)
        except Exception:
            pass # corrupt EXIF metadata is fine

        if (icc_raw := img.info.get("icc_profile")) is not None:
            cms_info: dict[str, Any] = {
                "native_mode": img.mode,
                "transparency": img.has_transparency_data,
            }

            try:
                profile = ImageCmsProfile(BytesIO(icc_raw))
                _add_info(cms_info, profile.profile, "profile_description")
                _add_info(cms_info, profile.profile, "target")
                _add_info(cms_info, profile.profile, "xcolor_space")
                _add_info(cms_info, profile.profile, "connection_space")
                _add_info(cms_info, profile.profile, "colorimetric_intent")
                _add_info(cms_info, profile.profile, "rendering_intent")

                working_mode = img.mode
                if img.mode.startswith(("RGB", "BGR", "P")):
                    working_mode = "RGBA" if img.has_transparency_data else "RGB"
                elif img.mode.startswith(("L", "I", "F")) or img.mode == "1":
                    working_mode = "LA" if img.has_transparency_data else "L"

                if img.mode != working_mode:
                    cms_info["working_mode"] = working_mode
                    img = img.convert(working_mode)

                mode = "RGBA" if img.has_transparency_data else "RGB"

                intent = Intent.RELATIVE_COLORIMETRIC
                if isIntentSupported(profile, intent, Direction.INPUT) != 1:
                    intent = _coalesce_intent(getDefaultIntent(profile))

                cms_info["conversion_intent"] = intent

                if (flags := _INTENT_FLAGS.get(intent)) is not None:
                    if img.mode == mode:
                        profileToProfile(
                            img,
                            profile,
                            _SRGB,
                            renderingIntent=intent,
                            inPlace=True,
                            flags=flags
                        )
                    else:
                        img = cast('Image.Image', profileToProfile(
                            img,
                            profile,
                            _SRGB,
                            renderingIntent=intent,
                            outputMode=mode,
                            flags=flags
                        ))
                else:
                    warn(CMSWarning(
                        f"unsupported intent on {path} assuming sRGB: {cms_info}",
                        path=path,
                        cms_info=cms_info
                    ), stacklevel=2)
            except PyCMSError as ex:
                warn(CMSWarning(
                    f"{ex} on {path}, assuming sRGB: {cms_info}",
                    path=path,
                    cms_info=cms_info,
                    cause=ex,
                ), stacklevel=2)

    except Exception as ex:
        print(f"{ex} on {path}")

    if img.has_transparency_data:
        if img.mode != "RGBA":
            try:
                img = img.convert("RGBA")
            except ValueError:
                img = img.convert("RGBa").convert("RGBA")
    elif img.mode != "RGB":
        img = img.convert("RGB")

    return img

def put_srgb(img: Image.Image, tensor: Tensor) -> None:
    if img.mode not in ("RGB", "RGBA", "RGBa"):
        raise ValueError(f"Image has non-RGB mode {img.mode}.")

    np.copyto(tensor.numpy(), np.asarray(img)[:, :, :3], casting="no")
