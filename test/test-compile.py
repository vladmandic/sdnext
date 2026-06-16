#!/usr/bin/env python3

import io
import os
import sys
import ast
import logging
import contextlib
import py_compile
import importlib.util
import importlib.machinery
from pathlib import Path


includes = [
    { 'folder': '.', 'recursive': False, 'includes': True },
    { 'folder': 'pipelines', 'recursive': True, 'includes': True },
    { 'folder': 'modules', 'recursive': True, 'includes': False },
    { 'folder': 'scripts', 'recursive': True, 'local': True, 'includes': False },
]
excludes = [
    'node_modules',
    '__pycache__',
]
ignores = [
    'torch_directml',
    'intel_extension_for_pytorch',
    'torch_xla.core.xla_model',
    'flash_attn',
    'flash_attn_interface',
    'openai',
    'rembg',
    'controlnet_aux',
    'image_gen_aux',
    'torchsde',
    'ligo.segments',
    'torchdiffeq',
    'insightface',
    'pynvml',
]
output = '/tmp/pycompile'
root = Path('.')


def test_compile(folder: str, recursive: bool):
    stats = { 'ok': [], 'failed': [], 'errors': [] }
    for entry in os.scandir(folder):
        if not any(exclude in entry.path for exclude in excludes) and not entry.name.startswith('.'):
            if entry.is_file() and entry.name.endswith('.py'):
                try:
                    cfile = os.path.join(output, os.path.relpath(entry.path, start='.')) + 'c'
                    py_compile.compile(entry.path, cfile=cfile, doraise=True)
                    stats['ok'].append(entry.path)
                except Exception as e:
                    print(f'fail: file={entry.path} error={e}')
                    stats['failed'].append(entry.path)
                    stats['errors'].append(str(e))
            elif entry.is_dir() and recursive:
                nested_stats = test_compile(entry.path, recursive)
                stats['ok'].extend(nested_stats['ok'])
                stats['failed'].extend(nested_stats['failed'])
                stats['errors'].extend(nested_stats['errors'])
    if len(stats["ok"]) > 0 or len(stats["failed"]) > 0:
        print(f'Compile: folder={folder} ok={len(stats["ok"])} failed={len(stats["failed"])}')
    return stats


def list_imports(path: Path):
    imports = []
    tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append((alias.name, 0, path))
        elif isinstance(node, ast.ImportFrom):
            module = node.module
            if node.level > 0:
                if module:
                    imports.append((module, node.level, path))
                else:
                    for alias in node.names:
                        if alias.name != '*':
                            imports.append((alias.name, node.level, path))
            elif module:
                imports.append((module, 0, path))
    return imports


def find_import(module: str | None, level: int, path: Path):
    package_parts = path.relative_to(root).with_suffix('').parts
    package_parts = package_parts[:-1]
    if level > len(package_parts) + 1:
        return None
    if level == 0:
        base = list(package_parts)
    else:
        base = list(package_parts[: -level + 1]) if level > 1 else list(package_parts)
    if module:
        base.extend(module.split('.'))
    return '.'.join(base) if base else None


def local_import(path: Path) -> bool: # modules that modify sys.path to allow local imports
    text = path.read_text(encoding='utf-8')
    return 'sys.path.append' in text or 'sys.path.insert' in text


def install_import(path: Path, module_name: str): # modules that install packages at runtime and import them
    text = path.read_text(encoding='utf-8')
    if f'install("{module_name}' in text or f'install(\'{module_name}' in text:
        return True
    return False


def test_import(module: str | None, level: int, path: Path, local: bool = False):
    module_name = find_import(module, level, path) if level > 0 else module
    if not module_name:
        return True
    old_disable = logging.root.manager.disable
    error = None
    try:
        logging.disable(logging.CRITICAL)
        if install_import(path, module_name):
            return True
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            try:
                spec = importlib.util.find_spec(module_name)
            except ModuleNotFoundError as e:
                spec = None
                error = e
            if spec is not None:
                return True
            if local or local_import(path):
                for extra in [str(path.parent.parent), str(path.parent)]:
                    spec = importlib.machinery.PathFinder.find_spec(module_name, [extra] + sys.path)
                    if spec is not None:
                        return True
            if error is not None and any(ignore in str(error) for ignore in ignores):
                return True
            return False
    except Exception as e:
        if any(ignore in str(e) for ignore in ignores):
            return True
        return False
    finally:
        logging.disable(old_disable)


def verify_imports(folder: str, recursive: bool, local: bool = False):
    stats = { 'ok': [], 'failed': [] }
    for entry in os.scandir(folder):
        if not any(exclude in entry.path for exclude in excludes) and not entry.name.startswith('.'):
            if entry.is_file() and entry.name.endswith('.py'):
                file_path = Path(entry.path)
                has_failure = False
                for module, level, _path in list_imports(file_path):
                    if module in ignores:
                        continue
                    if not test_import(module, level, file_path, local):
                        stats['failed'].append(f'{entry.path}: module={module} level={level}')
                        has_failure = True
                if not has_failure:
                    stats['ok'].append(entry.path)
            elif entry.is_dir() and recursive:
                nested_stats = verify_imports(entry.path, recursive)
                stats['ok'].extend(nested_stats['ok'])
                stats['failed'].extend(nested_stats['failed'])
    if len(stats['ok']) > 0 or len(stats['failed']) > 0:
        print(f'Imports: folder={folder} ok={len(stats["ok"])} failed={len(stats["failed"])}')
    if len(stats['failed']) > 0:
        for fail in stats['failed']:
            print(f'  {fail}')
    return stats


if __name__ == '__main__':
    os.makedirs(output, exist_ok=True)
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / 'modules' / 'control'))
    for item in includes:
        print(f"Test {item['folder']}")
        test_compile(item['folder'], item['recursive'])
        if item.get('includes', False):
            verify_imports(item['folder'], item.get('recursive', False), item.get('local', False))
