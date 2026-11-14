registered = False


def register_extensions():
    global registered # pylint: disable=global-statement
    if registered:
        return
    try:
        from installer import install
    except Exception:
        pass
    try:
        install('pillow-heif==1.8.1', quiet=True)
    except Exception:
        pass
    try:
        import pillow_heif
        pillow_heif.register_heif_opener()
    except Exception:
        pass
    try:
        install('pillow-avif-plugin==1.6.0', quiet=True)
    except Exception:
        pass
    try:
        import pillow_avif # pylint: disable=unused-import
    except Exception:
        pass
    try:
        install('pillow-jxl-plugin==1.3.8', quiet=True)
    except Exception:
        pass
    try:
        import pillow_jxl # pylint: disable=unused-import
    except Exception:
        pass
    registered = True


if not registered:
    register_extensions()
