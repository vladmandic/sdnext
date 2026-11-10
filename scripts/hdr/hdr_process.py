import importlib
import os
import shutil
import struct
import subprocess
import tempfile
import zlib

import numpy as np
from modules.logger import log


HDR_FORMATS = {
    'jpg': 'jpeg-gain-map',
    'jpeg': 'jpeg-gain-map',
    'png': 'png-cicp',
    'tif': 'tiff16',
    'tiff': 'tiff16',
    'avif': 'avif',
    'jxl': 'jxl',
    'exr': 'exr',
}

_GAMUT_MATRICES = {
    'rec.2020': np.array([
        [0.627404, 0.329283, 0.043313],
        [0.069097, 0.919540, 0.011362],
        [0.016391, 0.088013, 0.895595],
    ], dtype=np.float32),
    'dci-p3': np.array([
        [0.822592, 0.177534, 0.0],
        [0.033200, 0.966784, 0.0],
        [0.017085, 0.072396, 0.910302],
    ], dtype=np.float32),
}


def resolve_hdr_format(samples_format: str | None) -> str | None:
    """Map the configured sample extension to its HDR encoding."""
    if not isinstance(samples_format, str):
        return None
    return HDR_FORMATS.get(samples_format.lower().strip().lstrip('.'))


def srgb_to_linear(rgb: np.ndarray) -> np.ndarray:
    return np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)


def algorithmic_hdr(image, target_peak_nits: float = 1000.0, target_gamut: str = 'Rec.2020') -> np.ndarray:
    """Expand an sRGB image into linear HDR RGB, normalized to 100-nit diffuse white."""
    if target_peak_nits < 100.0 or target_peak_nits > 10000.0:
        raise ValueError(f'target peak must be between 100 and 10000 nits, got {target_peak_nits}')
    gamut = target_gamut.lower()
    if gamut not in _GAMUT_MATRICES:
        raise ValueError(f'unsupported HDR gamut: {target_gamut}')

    rgb = np.asarray(image.convert('RGB'), dtype=np.float32) / 255.0
    linear = srgb_to_linear(rgb)
    luminance = np.sum(linear * np.array([0.2126, 0.7152, 0.0722], dtype=np.float32), axis=2, keepdims=True)
    mask = np.clip((luminance - 0.5) / 0.5, 0.0, 1.0)
    mask = mask * mask * (3.0 - 2.0 * mask)
    expanded = linear * (1.0 + mask * (target_peak_nits / 100.0 - 1.0))
    converted = expanded @ _GAMUT_MATRICES[gamut].T
    return np.nan_to_num(np.clip(converted, 0.0, target_peak_nits / 100.0), copy=False)


def linear_to_pq(rgb: np.ndarray) -> np.ndarray:
    """Encode linear RGB normalized to 100-nit diffuse white as SMPTE ST 2084."""
    m1 = 2610.0 / 16384.0
    m2 = 2523.0 / 32.0
    c1 = 3424.0 / 4096.0
    c2 = 2413.0 / 128.0
    c3 = 2392.0 / 128.0
    luminance = np.clip(rgb * 100.0 / 10000.0, 0.0, 1.0) ** m1
    return ((c1 + c2 * luminance) / (1.0 + c3 * luminance)) ** m2


def _gamut_to_cicp(gamut: str) -> int:
    return 9 if gamut.lower() == 'rec.2020' else 12


def _write_encoded(path: str, extension: str, rgb: np.ndarray) -> None:
    import cv2
    bgr = np.ascontiguousarray(rgb[:, :, ::-1])
    ok, encoded = cv2.imencode(extension, bgr)
    if not ok:
        raise OSError(f'OpenCV could not encode {extension}')
    with open(path, 'wb') as file:
        file.write(encoded.tobytes())


def _get_imagecodecs():
    try:
        import imagecodecs
    except ImportError:
        from installer import install
        install('imagecodecs', 'imagecodecs', quiet=True)
        importlib.invalidate_caches()
        import imagecodecs
    return imagecodecs


def _add_png_cicp(path: str, primaries: int, transfer: int, matrix: int) -> None:
    with open(path, 'rb') as file:
        data = file.read()
    if not data.startswith(b'\x89PNG\r\n\x1a\n'):
        raise ValueError('not a PNG image')
    payload = bytes((primaries, transfer, matrix, 1))
    chunk_type = b'cICP'
    chunk = struct.pack('>I', len(payload)) + chunk_type + payload
    chunk += struct.pack('>I', zlib.crc32(chunk_type + payload) & 0xffffffff)
    with open(path, 'wb') as file:
        file.write(data[:33] + chunk + data[33:])


def _save_ultrahdr(path: str, hdr: np.ndarray, image, target_peak_nits: float, target_gamut: str) -> bool:
    executable = shutil.which('ultrahdr_app')
    if not executable:
        log.warning('HDR: JPEG gain map skipped: ultrahdr_app is not installed')
        return False
    height, width = hdr.shape[:2]
    gamut_code = 2 if target_gamut.lower() == 'rec.2020' else 1
    pq = np.clip(linear_to_pq(hdr), 0.0, 1.0)
    quantized = np.rint(pq * 1023).astype(np.uint32)
    rgba1010102 = quantized[:, :, 0] | (quantized[:, :, 1] << 10) | (quantized[:, :, 2] << 20) | (3 << 30)
    with tempfile.TemporaryDirectory(prefix='sdnext-hdr-') as temp_dir:
        sdr_path = os.path.join(temp_dir, 'sdr.jpg')
        hdr_path = os.path.join(temp_dir, 'hdr.rgba1010102')
        output_path = os.path.join(temp_dir, 'gain-map.jpg')
        image.convert('RGB').save(sdr_path, format='JPEG', quality=95)
        rgba1010102.tofile(hdr_path)
        command = [
            executable, '-m', '0', '-p', hdr_path, '-i', sdr_path,
            '-w', str(width), '-h', str(height), '-a', '5', '-C', str(gamut_code),
            '-t', '2', '-q', '95', '-K', str(max(1.0, target_peak_nits / 100.0)),
            '-z', output_path,
        ]
        subprocess.run(command, check=True, capture_output=True, timeout=600)
        os.replace(output_path, path)
    return True


def _save_avif(path: str, hdr: np.ndarray, target_gamut: str) -> None:
    imagecodecs = _get_imagecodecs()
    pq10 = np.rint(np.clip(linear_to_pq(hdr), 0.0, 1.0) * 1023).astype(np.uint16)
    primaries = imagecodecs.AVIF.COLOR_PRIMARIES.BT2020 if target_gamut.lower() == 'rec.2020' else imagecodecs.AVIF.COLOR_PRIMARIES.SMPTE432
    encoded = imagecodecs.avif_encode(
        pq10,
        level=100,
        bitspersample=10,
        primaries=primaries,
        transfer=imagecodecs.AVIF.TRANSFER_CHARACTERISTICS.PQ,
        matrix=imagecodecs.AVIF.MATRIX_COEFFICIENTS.IDENTITY,
    )
    with open(path, 'wb') as file:
        file.write(encoded)


def _save_jxl(path: str, hdr: np.ndarray, target_gamut: str) -> None:
    imagecodecs = _get_imagecodecs()
    pq16 = np.rint(np.clip(linear_to_pq(hdr), 0.0, 1.0) * 65535).astype(np.uint16)
    primaries = imagecodecs.JPEGXL.PRIMARIES.BT2100 if target_gamut.lower() == 'rec.2020' else imagecodecs.JPEGXL.PRIMARIES.P3
    encoded = imagecodecs.jpegxl_encode(
        pq16,
        lossless=True,
        bitspersample=16,
        primaries=primaries,
        transfer=imagecodecs.JPEGXL.TRANSFER_FUNCTION.PQ,
    )
    with open(path, 'wb') as file:
        file.write(encoded)


def save_hdr_image(path: str, hdr: np.ndarray, image, hdr_format: str, target_peak_nits: float, target_gamut: str) -> bool:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    if hdr_format == 'jpeg-gain-map':
        return _save_ultrahdr(path, hdr, image, target_peak_nits, target_gamut)
    if hdr_format == 'png-cicp':
        pq16 = np.rint(np.clip(linear_to_pq(hdr), 0.0, 1.0) * 65535).astype(np.uint16)
        _write_encoded(path, '.png', pq16)
        primaries = _gamut_to_cicp(target_gamut)
        _add_png_cicp(path, primaries, 16, 0)
        return True
    if hdr_format == 'tiff16':
        pq16 = np.rint(np.clip(linear_to_pq(hdr), 0.0, 1.0) * 65535).astype(np.uint16)
        import tifffile
        tifffile.imwrite(path, pq16, photometric='rgb', metadata={
            'transfer': 'SMPTE ST 2084',
            'primaries': target_gamut,
            'target_peak_nits': target_peak_nits,
        })
        return True
    if hdr_format == 'avif':
        _save_avif(path, hdr, target_gamut)
        return True
    if hdr_format == 'jxl':
        _save_jxl(path, hdr, target_gamut)
        return True
    if hdr_format == 'exr':
        from modules.image.exr import save_exr
        save_exr(path, hdr, target_gamut)
        return True
    return False
