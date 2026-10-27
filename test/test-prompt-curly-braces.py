#!/usr/bin/env python
"""
Offline tests for the curly-brace choice syntax in ``modules.styles``.

A structured JSON caption keeps its braces, including when extra-network tags
follow it; a choice list still resolves to one of its options.

No running server required.

Usage:
    python test/test-prompt-curly-braces.py
"""

import os
import sys
import time

script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, script_dir)
os.chdir(script_dir)

os.environ['SD_INSTALL_QUIET'] = '1'

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

from modules.errors import log  # pylint: disable=wrong-import-position
from modules import shared  # pylint: disable=wrong-import-position,unused-import
from modules import styles  # pylint: disable=wrong-import-position

CAPTION = '{"short_description":"A red fox.","objects":[{"description":"a red fox","location":"center foreground"}],"lighting":{"conditions":"soft light","direction":"low"}}'


def test_json_caption_untouched():
    assert styles.apply_curly_braces_to_prompt(CAPTION, seed=1) == CAPTION


def test_json_caption_with_network_tag_untouched():
    prompt = f'{CAPTION} <lora:anime:0.7>'
    assert styles.apply_curly_braces_to_prompt(prompt, seed=1) == prompt


def test_json_caption_with_comma_and_two_tags_untouched():
    prompt = f'{CAPTION}, <lora:anime:0.7> <lora:detail:0.3>'
    assert styles.apply_curly_braces_to_prompt(prompt, seed=1) == prompt


def test_choice_list_resolves():
    out = styles.apply_curly_braces_to_prompt('woman with {red|blue} dress <lora:x:1.0>', seed=1)
    assert out in ('woman with red dress <lora:x:1.0>', 'woman with blue dress <lora:x:1.0>'), out


def test_text_before_json_is_not_a_caption():
    out = styles.apply_curly_braces_to_prompt('photo of {"a":1} <lora:x:1.0>', seed=1)
    assert '{' not in out and '}' not in out, out


def run_tests():
    t0 = time.time()
    passed, failed = 0, 0
    for fn in [
        test_json_caption_untouched,
        test_json_caption_with_network_tag_untouched,
        test_json_caption_with_comma_and_two_tags_untouched,
        test_choice_list_resolves,
        test_text_before_json_is_not_a_caption,
    ]:
        try:
            fn()
            log.info(f'  PASS: {fn.__name__}')
            passed += 1
        except AssertionError as e:
            log.error(f'  FAIL: {fn.__name__} ({e})')
            failed += 1
    log.warning(f'Total: {passed} passed, {failed} failed in {time.time() - t0:.2f}s')
    return failed == 0


if __name__ == '__main__':
    sys.exit(0 if run_tests() else 1)
