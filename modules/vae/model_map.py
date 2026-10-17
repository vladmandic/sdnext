vae_map = {
    'sd': ['sd', 'sdxs'],
    'sdxl': ['sdxl', 'ldm', 'pixartalpha', 'pixartsigma', 'hunyuandit', 'omnigen', 'auraflow', 'bria', 'kolors', 'lumina'],
    'f1': ['f1', 'h1', 'zimage', 'lumina2', 'chroma', 'longcat', 'omnigen2', 'flite', 'ovis', 'kandinsky5', 'glmimage', 'cogview3', 'cogview4', 'ultraflux', 'boogu', 'step1x_edit'],
    'f2': ['f2', 'f3', 'ernieimage', 'lens', 'ideogram4', 'lladaimage', 'mossimage', 'sefi'],
    'sd3': ['sd3'],
    'wan21': ['wanai', 'qwen', 'chrono', 'cosmos', 'anima', 'fibo', 'joy', 'krea2', 'mingimage', 'nucleusimage'],
    'minimaxh3': ['minimaxh3'],
    'qwen21': ['qwen21'],
    # No shared MicroDecoder family; model-specific TAESD fallbacks still apply.
    None: ['allegrovideo', 'cogvideo', 'hunyuanimage', 'hunyuanimage3', 'hunyuanvideo', 'hunyuanvideo15', 'kandinsky', 'kandinsky3', 'kandinsky5video', 'ltxvideo', 'luminadimoo', 'mageflow', 'meissonic', 'mochivideo', 'nanobanana', 'nextstep', 'o1', 'prx', 'prxpixel', 'sana', 'sc', 'svd', 'veo3', 'x-omni', 'zetachroma']
}

def get_vae_type():
    from modules import shared
    if not shared.sd_loaded:
        return None
    for k, v in vae_map.items():
        if shared.sd_model_type in v:
            return k
    return None
