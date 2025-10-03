def check_qwen_pruning(repo_id):
    from modules.shared import log
    if 'pruning' not in repo_id.lower():
        return repo_id
    if '2509' in repo_id:
        repo_id = "Qwen/Qwen-Image-Edit-2509"
    elif 'Edit' in repo_id:
        repo_id = "Qwen/Qwen-Image-Edit"
    else:
        repo_id = "Qwen/Qwen-Image"
    log.debug(f'Load model: variant=pruning target="{repo_id}"')
    return repo_id
