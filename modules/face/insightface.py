import os
from modules.shared import log, opts
from modules import devices


insightface_app = None
instightface_mp = None


def get_app(mp_name, threshold=0.5, resolution=640):
    global insightface_app, instightface_mp # pylint: disable=global-statement

    from installer import install, installed, install_pydantic
    if not installed('insightface', reload=False, quiet=True):
        install('git+https://github.com/deepinsight/insightface@554a05561cb71cfebb4e012dfea48807f845a0c2#subdirectory=python-package', 'insightface') # insightface==0.7.3 with patches
        install('albumentations==1.4.3', ignore=False, reinstall=True)
        install_pydantic()
    if not installed('ip_adapter', reload=False, quiet=True):
        install('git+https://github.com/tencent-ailab/IP-Adapter.git', 'ip_adapter', ignore=False)

    if insightface_app is None or mp_name != instightface_mp:
        import insightface
        from insightface.model_zoo import model_zoo
        from insightface.app import face_analysis
        model_zoo.print = lambda *args, **kwargs: None
        face_analysis.print = lambda *args, **kwargs: None
        import huggingface_hub as hf
        import zipfile
        log.debug(f"InsightFace: version={insightface.__version__} mp={mp_name} provider={devices.onnx}")
        root_dir = os.path.join(opts.diffusers_dir, 'models--vladmandic--insightface-faceanalysis')
        local_dir = os.path.join(root_dir, 'models')
        extract_dir = os.path.join(local_dir, mp_name)
        model_path = os.path.join(local_dir, f'{mp_name}.zip')
        if not os.path.exists(model_path):
            model_path = hf.hf_hub_download(
                repo_id='vladmandic/insightface-faceanalysis',
                filename=f'{mp_name}.zip',
                local_dir_use_symlinks=False,
                cache_dir=opts.hfcache_dir,
                local_dir=local_dir
            )
        if not os.path.exists(extract_dir):
            log.debug(f'InsightFace extract: folder="{extract_dir}"')
            os.makedirs(extract_dir)
            with zipfile.ZipFile(model_path) as zf:
                zf.extractall(local_dir)
        kwargs = {
            'root': root_dir,
            'download': False,
            'download_zip': False,
        }
        insightface_app = face_analysis.FaceAnalysis(name=mp_name, providers=devices.onnx, **kwargs)
        instightface_mp = mp_name
        insightface_app.prepare(ctx_id=0, det_thresh=threshold, det_size=(resolution, resolution))
    return insightface_app
