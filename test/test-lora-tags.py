#!/usr/bin/env python
"""
Offline tests for *LoRA auto-apply tags* (``extra_networks_lora.prompt``).

Networks activate after the batch's prompts were sliced from ``all_prompts`` and stripped of extra-network references,
so the tags have to reach ``p.prompts`` as well as ``p.prompt`` and ``p.all_prompts``. Activation runs once per batch
and again for hires, so applying the tags twice must change nothing.

Covers:

- the tags reach the batch that loads the LoRA (``p.prompts``)
- the tagged raw prompt parses to the batch prompt, so later batches and a restore encode the same text; appended tags
  go before trailing references
- a tag that is part of the LoRA's file name is added: the reference ``<lora:name:1.0>`` does not count as the tag
- a second activation adds nothing, no bare ``, ``, and the infotext keeps the applied tags
- several tags keep the order of the LoRA's metadata
- a LoRA without tags in its metadata adds nothing
- finding the trailing references stays linear in a long run of spaces
- a tag already in the prompt text is not repeated (control)

The loaded network and the LoRA page are stubs. No model, no running server.

Usage:
    python test/test-lora-tags.py
"""

import os
import sys
import time
from types import SimpleNamespace

script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, script_dir)
os.chdir(script_dir)

os.environ['SD_INSTALL_QUIET'] = '1'

# Bootstrap cmd_args before any module that pulls in shared.py.
import modules.cmd_args  # pylint: disable=wrong-import-position
import installer  # pylint: disable=wrong-import-position
orig_argv = sys.argv
sys.argv = [sys.argv[0]]
try:
    modules.cmd_args.parse_args()
finally:
    sys.argv = orig_argv
installer.add_args(modules.cmd_args.parser)
modules.cmd_args.parsed, _ = modules.cmd_args.parser.parse_known_args([])

from modules import shared, extra_networks                      # pylint: disable=wrong-import-position
from modules.lora import extra_networks_lora                    # pylint: disable=wrong-import-position
from modules.lora import lora_common                            # pylint: disable=wrong-import-position
from modules.errors import log                                  # pylint: disable=wrong-import-position

results: dict[str, dict] = {}
LORA = 'modare-anima-lokrC12'
REFERENCE = f'<lora:{LORA}:1.0>'


def category(name: str):
    if name not in results:
        results[name] = {'passed': 0, 'failed': 0, 'skipped': 0, 'tests': []}
    return name


def record(cat: str, status: str, name: str, detail: str = ''):
    results[cat][{'PASS': 'passed', 'FAIL': 'failed', 'SKIP': 'skipped'}[status]] += 1
    results[cat]['tests'].append((status, name))
    msg = f'  {status}: {name}'
    if detail:
        msg += f' ({detail})'
    if status == 'PASS':
        log.info(msg)
    elif status == 'SKIP':
        log.warning(msg)
    else:
        log.error(msg)


def run_test(cat: str, fn):
    name = fn.__name__
    try:
        ok = fn()
        if isinstance(ok, str):
            record(cat, 'SKIP', name, ok)
        else:
            record(cat, 'FAIL' if ok is False else 'PASS', name)
    except AssertionError as e:
        record(cat, 'FAIL', name, str(e))
    except Exception as e:  # pylint: disable=broad-except
        record(cat, 'FAIL', name, f'exception: {type(e).__name__}: {e}')


class LoraPage:
    name = 'lora'

    def __init__(self, tags: dict):
        self.tags = tags

    def create_item(self, name):  # pylint: disable=unused-argument
        return {'tags': dict(self.tags)}


def batch(prompts: list[str]):
    """A processing object as activation sees it: the batch's prompts already parsed, which strips extra network references."""
    return SimpleNamespace(
        prompt=prompts[0],
        all_prompts=list(prompts),
        prompts=[extra_networks.parse_prompt(prompt)[0] for prompt in prompts],
        extra_generation_params={},
    )


def apply(p, tags: dict, max_tags: int = -1, calls: int = 1):
    saved_networks = list(lora_common.loaded_networks)
    saved_pages = shared.extra_networks
    saved_max = shared.opts.data.get('lora_apply_tags', None)
    lora_common.loaded_networks[:] = [SimpleNamespace(name=LORA, tags=[])]
    shared.extra_networks = [LoraPage(tags)]
    shared.opts.data['lora_apply_tags'] = max_tags
    try:
        for _ in range(calls):
            extra_networks_lora.prompt(p)
        return p
    finally:
        lora_common.loaded_networks[:] = saved_networks
        shared.extra_networks = saved_pages
        if saved_max is None:
            shared.opts.data.pop('lora_apply_tags', None)
        else:
            shared.opts.data['lora_apply_tags'] = saved_max


def test_tags_reach_current_batch():
    """Red before the fix: the tags went into all_prompts after the batch had been sliced from it."""
    p = apply(batch([f'a cat {REFERENCE}', f'a dog {REFERENCE}']), {'painterly': 3})
    assert p.prompts == ['a cat, painterly', 'a dog, painterly'], f'batch prompts {p.prompts}'
    assert p.all_prompts == [f'a cat, painterly {REFERENCE}', f'a dog, painterly {REFERENCE}'], f'all prompts {p.all_prompts}'
    return True


def test_tagged_prompt_parses_to_batch_prompt():
    """Red before the fix: later batches and a restore parse the tagged prompt, which differed from what the loading batch encoded."""
    prompts = [f'a cat {REFERENCE}', f'a cat, {REFERENCE}', REFERENCE, f'a {REFERENCE} cat', f'a cat {REFERENCE}, soft light', 'a cat,', f'a cat\n{REFERENCE}', f'a cat {REFERENCE}\n<lora:other:0.5>']
    p = apply(batch(prompts), {'painterly': 3})
    parsed = [extra_networks.parse_prompt(prompt)[0] for prompt in p.all_prompts]
    assert parsed == p.prompts, f'parsed tagged prompts {parsed} against batch prompts {p.prompts}'
    assert all('painterly' in prompt for prompt in p.prompts), f'batch prompts {p.prompts}'
    return True


def test_tag_in_lora_name_is_added():
    """Red before the fix: `modare` occurs inside `<lora:modare-...>`, so it was skipped and a bare `, ` appended."""
    p = apply(batch([f'a cat {REFERENCE}']), {'modare': 3})
    assert p.prompt == f'a cat, modare {REFERENCE}', f'prompt {p.prompt!r}'
    assert p.prompts == ['a cat, modare'], f'batch prompts {p.prompts}'
    return True


def test_second_activation_adds_nothing():
    """Red before the fix: once the tags were in the prompt, each later activation appended `, ` and blanked the infotext."""
    p = apply(batch([f'a cat {REFERENCE}']), {'painterly': 3, 'chibi': 1}, calls=3)
    assert p.prompt == f'a cat, painterly, chibi {REFERENCE}', f'prompt {p.prompt!r}'
    assert p.all_prompts == [f'a cat, painterly, chibi {REFERENCE}'], f'all prompts {p.all_prompts}'
    assert p.prompts == ['a cat, painterly, chibi'], f'batch prompts {p.prompts}'
    assert p.extra_generation_params.get('LoRA tags') == 'painterly, chibi', f'infotext {p.extra_generation_params}'
    return True


def test_tags_keep_metadata_order():
    """Red before the fix: a set ordered the tags by string hash, which changes from one process to the next."""
    tags = {f'tag{i}': 10 - i for i in range(8)}
    p = apply(batch(['a cat']), tags)
    assert p.prompt == 'a cat, ' + ', '.join(tags), f'prompt {p.prompt!r}'
    return True


def test_lora_without_tags_adds_nothing():
    """Red before the fix: the LoRA's name stood in as its tag, was found inside its own reference and left a bare `, `."""
    p = apply(batch([f'a cat {REFERENCE}']), {})
    assert p.prompt == f'a cat {REFERENCE}', f'prompt {p.prompt!r}'
    assert p.prompts == ['a cat'], f'batch prompts {p.prompts}'
    assert 'LoRA tags' not in p.extra_generation_params, f'infotext {p.extra_generation_params}'
    return True


def test_long_separator_run_stays_linear():
    """A trailing-reference search that starts with [\\s,]* backtracks across a long run of spaces, quadratic in its length."""
    text = 'a' + ' ' * 40000 + 'b'
    t0 = time.perf_counter()
    tagged = extra_networks_lora.add_tags(text, ['painterly'])
    elapsed = time.perf_counter() - t0
    assert tagged == f'{text}, painterly', f'tagged text ends {tagged[-20:]!r}'
    assert elapsed < 0.5, f'{elapsed:.2f}s for a 40000 character run of spaces'
    return True


def test_tag_already_in_prompt_not_repeated():
    p = apply(batch(['a painterly cat']), {'painterly': 3, 'chibi': 1})
    assert p.prompt == 'a painterly cat, chibi', f'prompt {p.prompt!r}'
    return True


def run_all():
    log.warning('=== LoRA auto-apply tags ===')
    cat = category('tags')
    for fn in [
        test_tags_reach_current_batch,
        test_tagged_prompt_parses_to_batch_prompt,
        test_tag_in_lora_name_is_added,
        test_second_activation_adds_nothing,
        test_tags_keep_metadata_order,
        test_lora_without_tags_adds_nothing,
        test_long_separator_run_stays_linear,
        test_tag_already_in_prompt_not_repeated,
    ]:
        run_test(cat, fn)

    log.warning('=== Results ===')
    total = {'passed': 0, 'failed': 0, 'skipped': 0}
    for cat_name, info in results.items():
        status = 'PASS' if info['failed'] == 0 else 'FAIL'
        log.info(f"  {cat_name}: {info['passed']} passed, {info['failed']} failed, {info['skipped']} skipped [{status}]")
        for key in total:
            total[key] += info[key]
    log.warning(f"Total: {total['passed']} passed, {total['failed']} failed, {total['skipped']} skipped")
    return total['failed'] == 0


if __name__ == '__main__':
    t0 = time.time()
    ok = run_all()
    log.warning(f'Total time: {time.time() - t0:.2f}s')
    sys.exit(0 if ok else 1)
