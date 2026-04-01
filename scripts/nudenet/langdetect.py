from modules.logger import log
repo_id = "facebook/fasttext-language-identification"
model = None


def lang_detect(text:str, top:int=1, threshold:float=0.25) -> str:
    try:
        global model # pylint: disable=global-statement
        from modules import shared
        if model is None:
            from installer import install
            install('fasttext')
            import fasttext
            from huggingface_hub import hf_hub_download
            model_path = hf_hub_download(repo_id, filename="model.bin", cache_dir=shared.opts.hfcache_dir)
            log.info(f'NudeNet load: model="{repo_id}"')
            model = fasttext.load_model(model_path)
        text = text.replace('\n', '. ')
        lang, score = model.predict(text, k=top, threshold=threshold, on_unicode_error="ignore")
        result = [f'{l.replace("__label__", "").lower()}:{s:.2f}' for l, s in zip(lang, score) if s > threshold][:top]
        log.debug(f'NudeNet LangDetect: {result}')
        return result
    except Exception as e:
        log.error(f'NudeNet LangDetect: {e}')
        return str(e)
