import numpy as np


_CHROMATICITIES = {
    'rec.709': ((0.640, 0.330), (0.300, 0.600), (0.150, 0.060), (0.3127, 0.3290)),
    'rec.2020': ((0.708, 0.292), (0.170, 0.797), (0.131, 0.046), (0.3127, 0.3290)),
    'dci-p3': ((0.680, 0.320), (0.265, 0.690), (0.150, 0.060), (0.3127, 0.3290)),
}


def _write_with_openexr(path: str, rgb: np.ndarray, gamut: str) -> None:
    import Imath
    import OpenEXR
    height, width = rgb.shape[:2]
    pixel_type = Imath.PixelType(Imath.PixelType.FLOAT)
    header_factory = getattr(OpenEXR, 'Header')  # noqa: B009
    header = header_factory(width, height)
    header['channels'] = {name: Imath.Channel(pixel_type) for name in ('R', 'G', 'B')}
    primaries = _CHROMATICITIES[gamut.lower()]
    header['chromaticities'] = Imath.Chromaticities(*(Imath.V2f(*point) for point in primaries))
    channels = {name: np.ascontiguousarray(rgb[:, :, index], dtype=np.float32).tobytes() for index, name in enumerate(('R', 'G', 'B'))}
    output_file = getattr(OpenEXR, 'OutputFile')  # noqa: B009
    output = output_file(path, header)
    try:
        output.writePixels(channels)
    finally:
        output.close()


def save_exr(path: str, rgb: np.ndarray, gamut: str = 'Rec.2020') -> None:
    rgb = np.asarray(rgb, dtype=np.float32)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f'expected HxWx3 RGB image, got {rgb.shape}')
    try:
        _write_with_openexr(path, rgb, gamut)

    except ImportError:
        from installer import install
        install('OpenEXR', 'OpenEXR', quiet=True)
        import importlib
        importlib.invalidate_caches()
        _write_with_openexr(path, rgb, gamut)
