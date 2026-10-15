#!/usr/bin/env python

import os
import json
import argparse
from rich import print as rp


api = None
default_target = "SD-Next"


def hf_init(token: str | None = None, silent: bool = False):
    token = token or load_token()
    import huggingface_hub as hf
    global api # pylint: disable=global-statement
    api = hf.HfApi(token=token)
    user = api.whoami()
    if not silent:
        rp(f'User: {user}')

def error(e: Exception):
    return str(e).splitlines()[0]


def load_token() -> str | None:
    secrets_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "secrets.json")
    if not os.path.isfile(secrets_path):
        return None
    try:
        with open(secrets_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        hf_token = data.get("huggingface_token")
        return hf_token if isinstance(hf_token, str) and hf_token.strip() else None
    except Exception:
        return None


def validate_model(repo: str, silent: bool = False) -> bool:
    try:
        model = api.model_info(repo_id=repo, files_metadata=True)
        if not silent:
            rp(model)
        return model
    except Exception as e:
        rp(f"Repo={repo}: {error(e)}")
        return False


def validate_user(repo: str, silent: bool = False) -> bool:
    user = repo.split("/")[0]
    err_org = ''
    err_user = ''
    try:
        info = api.get_organization_overview(user)
        if not silent:
            rp(info)
        return user
    except Exception as e:
        err_org = f"Organization={user}: {error(e)}"
    try:
        info = api.get_user_overview(user)
        if not silent:
            rp(info)
        return user
    except Exception as e:
        err_user = f"User={user}: {error(e)}"
    rp(err_org)
    rp(err_user)
    return False


def duplicate_repo(repo: str, target: str, overwrite: bool = False, private: bool = False, pretend: bool = False, silent: bool = False):
    model = validate_model(repo, silent=silent)
    user = validate_user(target, silent=silent)
    if not model or not user:
        rp('Aborting...')
        return

    if "/" not in target:
        name = model.id.split("/")[-1]
        target = f'{target}/{name}'
    kwargs = {
        "from_id": model.id,
        "to_id": target,
        "repo_type": "model",
        "exist_ok": overwrite,
        "private": private,
    }
    if not silent:
        rp(f"Args: {kwargs}")

    if pretend:
        url = 'dummy'
        if not silent:
            rp("Pretend...")
        return
    try:
        url = api.duplicate_repo(**kwargs)
        rp(f'Duplicate: {url}')
    except Exception as e:
        rp(f"Duplicate: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Duplicate HuggingFace model")
    parser.add_argument("--repo", type=str, help="Source repository", required=True)
    parser.add_argument("--target", type=str, default=default_target, help="Target username or repo", required=False)
    parser.add_argument("--token", type=str, help="HuggingFace API token", required=False)
    parser.add_argument("--overwrite", action="store_true", help="Overwrite the target repository if it exists")
    parser.add_argument("--private", action="store_true", help="Make the duplicated repository private")
    parser.add_argument("--silent", action="store_true", help="Suppress output")
    parser.add_argument("--pretend", action="store_true", help="Do not actually perform any actions, just simulate")
    args = parser.parse_args()
    if not args.silent:
        rp('HF-Copy')
        rp(args)
    hf_init(token=args.token, silent=args.silent)
    duplicate_repo(
        args.repo,
        args.target,
        overwrite=args.overwrite,
        private=args.private,
        pretend=args.pretend,
        silent=args.silent,
    )
