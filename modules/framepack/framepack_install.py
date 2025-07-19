import os
import shutil
import git as gitpython
from installer import install, git
from modules.shared import log


def rename(src:str, dst:str):
    import errno
    try:
        os.rename(src, dst)
    except OSError as e:
        if e.errno == errno.EXDEV: # cross-device
            shutil.move(src, dst)
        else:
            raise e


def install_requirements(attention:str='SDPA'):
    install('av')
    import av
    import torchvision
    torchvision.io.video.av = av
    if attention == 'Xformers':
        log.debug('FramePack install: xformers')
        install('xformers')
    elif attention == 'FlashAttention':
        log.debug('FramePack install: flash-attn')
        install('flash-attn')
    elif attention == 'SageAttention':
        log.debug('FramePack install: sageattention')
        install('sageattention')


def git_clone(git_repo:str, git_dir:str, tmp_dir:str):
    if os.path.exists(git_dir):
        return
    try:
        shutil.rmtree(tmp_dir, True)
        args = {
            'url': git_repo,
            'to_path': tmp_dir,
            'allow_unsafe_protocols': True,
            'allow_unsafe_options': True,
            'filter': ['blob:none'],
        }
        ssh = os.environ.get('GIT_SSH_COMMAND', None)
        if ssh:
            args['env'] = {'GIT_SSH_COMMAND':ssh}
        log.info(f'FramePack install: url={args} path={git_repo}')
        with gitpython.Repo.clone_from(**args) as repo:
            repo.remote().fetch(verbose=True)
            for submodule in repo.submodules:
                submodule.update()
        rename(tmp_dir, git_dir)
    except Exception as e:
        log.error(f'FramePack install: {e}')
    shutil.rmtree(tmp_dir, True)


def git_update(git_dir:str, git_commit:str):
    if not os.path.exists(git_dir):
        return
    try:
        with gitpython.Repo(git_dir) as repo:
            commit = repo.commit()
            if f'{commit}' != git_commit:
                log.info(f'FramePack update: path={repo.git_dir} current={commit} target={git_commit}')
                repo.git.fetch(all=True)
                repo.git.reset('origin', hard=True)
                git(f'checkout {git_commit}', folder=git_dir, ignore=True, optional=True)
            else:
                log.debug(f'FramePack version: sha={commit}')
    except Exception as e:
        log.error(f'FramePack update: {e}')
