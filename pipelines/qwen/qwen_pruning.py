def check_qwen_pruning(repo_id, subfolder):
    from modules.shared import log
    if 'pruning' not in repo_id.lower():
        return repo_id, subfolder
    if '2509' in (repo_id or '') or '2509' in (subfolder or ''):
        repo_id, subfolder = "Qwen/Qwen-Image-Edit-2509", None
    elif 'Edit' in (repo_id or '') or 'Edit' in (subfolder or ''):
        repo_id, subfolder = "Qwen/Qwen-Image-Edit", None
    else:
        repo_id, subfolder = "Qwen/Qwen-Image", None
    log.debug(f'Load model: variant=pruning target="{repo_id}"')
    return repo_id, subfolder
