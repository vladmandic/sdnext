#!/usr/bin/env python
"""
Offline tests for the native LoRA loader key mapping (issue #5174).

``lora_load.load_safetensors()`` classifies every state-dict key into a module
key plus a network part. A diffusers PEFT adapter that keeps its pipeline prefix
(``transformer.transformer_blocks.0.attn.to_q.lora_A.weight``) takes the prefixed
branch, which used to consume only ``weight`` and leave ``lora_A``/``lora_B`` on
the module key, so the module never resolved and the whole adapter was dropped
with nothing but an unmatched-keys warning, while generation still returned an
image built from the base weights.

Covers:

- prefixed PEFT keys resolving to the module and normalizing lora_A/lora_B to
  lora_down/lora_up, both directions of the pair
- prefixed lora_down/lora_up keys (ai-toolkit / converted kohya) still mapping
  exactly as before, alpha included
- a key whose module is genuinely absent from the model still reporting nothing
  matched instead of picking up a near-miss module

The loader's sibling modules are stubbed so the test stays offline and runs on
any checkout, the same way test-civitai-names.py builds the state it needs; only
modules/lora/lora_load.py and modules/lora/lora_convert.py are exercised for real.

Usage:
    python test/test-lora-native-peft-keys.py
"""

import importlib.util
import os
import sys
import tempfile
import types

import torch

script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, script_dir)
os.chdir(script_dir)

lora_dir = os.path.join(script_dir, 'modules', 'lora')
passed = 0
failed = 0


def check(label: str, got, want):
    global passed, failed  # pylint: disable=global-statement
    if got == want:
        passed += 1
    else:
        failed += 1
        print(f'FAIL: {label}: got={got!r} want={want!r}')


def register(mod):
    """Bind the stand-in into sys.modules and onto its parent package."""
    sys.modules[mod.__name__] = mod
    parent, _, leaf = mod.__name__.rpartition('.')
    if parent in sys.modules:
        setattr(sys.modules[parent], leaf, mod)
    return mod


def package(name: str, path: str):
    mod = types.ModuleType(name)
    mod.__path__ = [path]
    return register(mod)


def stub(name: str, **attrs):
    mod = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(mod, key, value)
    return register(mod)


class LogCapture:
    def __init__(self):
        self.lines = []

    def __getattr__(self, kind):
        def _f(msg, *_a, **_k):
            self.lines.append((kind, str(msg)))
        return _f


class FakeTransformer:
    """Stand-in for a diffusers transformer: only named_modules() is consumed."""

    MODULES = (
        'transformer_blocks.0.attn.to_q',
        'transformer_blocks.0.attn.to_k',
        'transformer_blocks.1.attn.to_q',
    )

    def named_modules(self):
        yield '', self
        for name in self.MODULES:
            yield name, self


class FakePipe:
    def __init__(self):
        self.transformer = FakeTransformer()


log = LogCapture()
state = {'sd': {}}

package('modules', os.path.join(script_dir, 'modules'))
package('modules.lora', lora_dir)
stub('modules.errors')
stub('modules.logger', log=log)
stub('modules.shared', opts=types.SimpleNamespace(extra_networks_default_multiplier=1.0, lora_fuse_diffusers=False, lora_add_hashes_to_infotext=False), sd_loaded=True, sd_model=None, sd_model_type='f1')
stub('modules.sd_models', state=state, read_state_dict=lambda filename, what=None: dict(state['sd']), get_diffusers_task=lambda pipe=None: 'txt2img')
stub('modules.sd_models_compile')
stub('modules.files_cache')
stub('modules.lora.network',
     Network=lambda name, network_on_disk=None: types.SimpleNamespace(name=name, network_on_disk=network_on_disk, modules={}, bundle_embeddings={}, mtime=0),
     NetworkWeights=lambda network_key=None, sd_key=None, w=None, sd_module=None: types.SimpleNamespace(network_key=network_key, sd_key=sd_key, w={} if w is None else w, sd_module=sd_module))
stub('modules.lora.lora_overrides', fuse_native=lambda: False)
stub('modules.lora.lora_common', debug=False, module_types=[], loaded_networks=[], errors={})
stub('modules.lora.lora_diffusers')


def load_real(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    register(mod)
    spec.loader.exec_module(mod)
    return mod


lora_convert = load_real('modules.lora.lora_convert', os.path.join(lora_dir, 'lora_convert.py'))
lora_load = load_real('modules.lora.lora_load', os.path.join(lora_dir, 'lora_load.py'))

matched = []
calls = 0


class FakeType:
    """Records what the loader resolved, in place of a real LoRA module type."""

    @staticmethod
    def create_module(_net, weights):
        matched.append((weights.sd_key, sorted(weights.w), weights.sd_module is not None))
        return object()


sys.modules['modules.lora.lora_common'].module_types = [FakeType]


def w(shape, value=1.0):
    return torch.full(shape, value, dtype=torch.float32)


def load(sd: dict):
    """Run the real loader over ``sd`` against a model exposing the flux block names."""
    from modules import shared
    from modules.lora import lora_common as l  # noqa: F401  pylint: disable=unused-import
    matched.clear()
    global calls  # pylint: disable=global-statement
    calls += 1  # the loader caches by name, so every case needs its own
    shared.sd_model = FakePipe()
    handle = tempfile.NamedTemporaryFile(suffix='.safetensors', delete=False)
    handle.close()
    state['sd'] = sd
    try:
        net = lora_load.load_safetensors(f'unit-test-peft-{calls}', types.SimpleNamespace(filename=handle.name, shorthash='00000000'))
    finally:
        os.unlink(handle.name)
    return net, list(matched)


def test_prefixed_peft_keys_resolve():
    """A PEFT pair under the transformer. prefix must reach the module it names."""
    net, got = load({
        'transformer.transformer_blocks.0.attn.to_q.lora_A.weight': w((4, 32)),
        'transformer.transformer_blocks.0.attn.to_q.lora_B.weight': w((32, 4)),
        'transformer.transformer_blocks.1.attn.to_q.lora_A.weight': w((4, 32)),
        'transformer.transformer_blocks.1.attn.to_q.lora_B.weight': w((32, 4)),
    })
    check('prefixed peft: network built', net is not None, True)
    check('prefixed peft: modules matched', sorted(got), [
        ('lora_transformer_transformer_blocks_0_attn_to_q', ['lora_down.weight', 'lora_up.weight'], True),
        ('lora_transformer_transformer_blocks_1_attn_to_q', ['lora_down.weight', 'lora_up.weight'], True),
    ])
    check('prefixed peft: network carries both layers', sorted(net.modules) if net is not None else None, [
        'lora_transformer_transformer_blocks_0_attn_to_q',
        'lora_transformer_transformer_blocks_1_attn_to_q',
    ])


def test_prefixed_kohya_keys_unchanged():
    """lora_down/lora_up keys under the same prefix keep mapping as before."""
    net, got = load({
        'transformer.transformer_blocks.0.attn.to_k.lora_down.weight': w((4, 32)),
        'transformer.transformer_blocks.0.attn.to_k.lora_up.weight': w((32, 4)),
        'transformer.transformer_blocks.0.attn.to_k.alpha': torch.tensor(4.0),
    })
    check('prefixed kohya: network built', net is not None, True)
    check('prefixed kohya: module and weights', got, [
        ('lora_transformer_transformer_blocks_0_attn_to_k', ['alpha', 'lora_down.weight', 'lora_up.weight'], True),
    ])


def test_unknown_module_still_unmatched():
    """A name the model does not have must not resolve to a near-miss module."""
    net, got = load({
        'transformer.transformer_blocks.9.attn.to_q.lora_A.weight': w((4, 32)),
        'transformer.transformer_blocks.9.attn.to_q.lora_B.weight': w((32, 4)),
    })
    check('unknown module: nothing loaded', net, None)
    check('unknown module: nothing matched', got, [])


if __name__ == '__main__':
    for test in [test_prefixed_peft_keys_resolve, test_prefixed_kohya_keys_unchanged, test_unknown_module_still_unmatched]:
        test()
    print(f'Total: {passed} passed, {failed} failed')
    sys.exit(1 if failed else 0)
