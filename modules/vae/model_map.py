vae_map = {
    'sd': ['sd'],
    'sdxl': ['sdxl', 'ldm', 'pixartalpha', 'pixartsigma', 'hunyuandit', 'omnigen', 'auraflow'],
    'f1': ['f1', 'h1', 'zimage', 'lumina2', 'chroma', 'longcat', 'omnigen2', 'flite', 'ovis', 'kandinsky5', 'glmimage', 'cogview3', 'cogview4', 'ultraflux'],
    'f2': ['f2', 'ernieimage', 'lens', 'ideogram4', 'lladaimage'],
    'sd3': ['sd3'],
    'wan21': ['wanai', 'qwen', 'chrono', 'cosmos', 'anima', 'fibo', 'joy', 'krea2'],
    'minimaxh3': ['minimaxh3'],
    'qwen21': ['qwen21']
}

def get_vae_type():
    from modules import shared
    if not shared.sd_loaded:
        return None
    for k, v in vae_map.items():
        if shared.sd_model_type in v:
            return k
    return None
