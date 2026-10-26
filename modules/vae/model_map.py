from modules.logger import log


last_vae = None
vae_map = {
    'sd': ['sd',
           'sdxs'],
    'sdxl': ['sdxl',
             'ldm',
             'pixartalpha',
             'pixartsigma',
             'hunyuandit',
             'omnigen',
             'auraflow',
             'bria',
             'kolors',
             'lumina'],
    'f1': ['f1',
           'h1',
           'zimage',
           'lumina2',
           'chroma',
           'longcat',
           'omnigen2',
           'flite',
           'ovis',
           'kandinsky5',
           'glmimage',
           'ultraflux',
           'boogu',
           'step1x_edit'],
    'f2': ['f2',
           'f3',
           'ernieimage',
           'lens',
           'ideogram4',
           'lladaimage',
           'mossimage',
           'sefi',
           'fibo2'],
    'sd3': ['sd3'],
    'wan21': ['wanai',
              'qwen',
              'chrono',
              'cosmos',
              'anima',
              'fibo',
              'joy',
              'krea2',
              'mingimage',
              'nucleusimage'], # TODO FIBO?
    'wan22': ['wan22'],
    'minimaxh3': ['minimaxh3'],
    'qwen21': ['qwen21'],
    'dcae': ['sana'],
    'mage': ['mageflow'],
    'cog': ['cogview3',
            'cogview4'],
    'hy': ['hunyuanimage'],
    None: ['allegrovideo',
           'cogvideo',
           'hunyuanimage3',
           'hunyuanvideo',
           'hunyuanvideo15',
           'kandinsky',
           'kandinsky3',
           'kandinsky5video',
           'ltxvideo',
           'luminadimoo',
           'meissonic',
           'mochivideo',
           'nanobanana',
           'nextstep',
           'o1',
           'prx',
           'prxpixel',
           'sc',
           'svd',
           'veo3',
           'x-omni',
           'zetachroma'],
}


def get_vae_type():
    global last_vae # pylint: disable=global-statement
    from modules import shared
    if not shared.sd_loaded:
        return None
    if hasattr(shared.sd_model, 'vae_variant'):
        vae_variant = shared.sd_model.vae_variant
    else:
        vae_variant = shared.sd_model_type
    for k, v in vae_map.items():
        if vae_variant in v:
            if last_vae != k:
                log.debug(f'Decode: model={shared.sd_model_type} vae={vae_variant} mapped={k}')
                last_vae = k
            return k
    return None
