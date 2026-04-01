#!/usr/bin/env python

import sys
import os
from collections import Counter

script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(script_dir)

# --- test defition -------------------------------
# library
fn = r'./modules/styles.py'
# tested function
funcname = 'select_from_weighted_list'
# random needed
ns = {'Dict': dict, 'random': __import__('random')}
# number of samples to test
tries = 2000
# allowed deviation in percentage points
tolerance_pct = 5
# tests
tests = [
    # - empty
    ["", { '': 100 } ],
    # - no weights
    [ "red|blonde|black", { 'black': 33, 'red': 33, 'blonde': 33 } ],
    # - full weights <= 1
    [ "red:0.1|blonde:0.9", { 'blonde': 90, 'red': 10 } ],
    # - weights > 1 to test normalization
    [ "red:1|blonde:2|black:5", { 'blonde': 25, 'red': 12.5, 'black': 62.5 } ],
    # - disabling 0 weights to force one result
    [ "red:0|blonde|black:0", { 'blonde': 100 } ],
    # - weights <= 1 with distribution of the leftover
    [ "red:0.5|blonde|black:0.3|brown", { 'red': 50, 'black': 30, 'brown': 10, 'blonde': 10 } ],
    # - weights > 1, unweightes should get default of 1
    [ "red:2|blonde|black", { 'red': 50, 'blonde': 25, 'black': 25 } ],
    # - ignore content of ()
    [ "red:0.5|(blonde:1.3)", { 'red': 50, '(blonde:1.3)': 50 } ],
    # - ignore content of []
    [ "red:0.5|[stuff:1.3]", { '[stuff:1.3]': 50, 'red': 50 } ],
    # - ignore content of <>
    [ "red:0.5|<lora:1.0>", { '<lora:1.0>': 50, 'red': 50 } ]
]

# -------------------------------------------------

with open(fn, 'r', encoding='utf-8') as f:
    src = f.read()
    start = src.find('def ' + funcname)
    if start == -1:
        print('Function not found')
        sys.exit(1)
    # find next top-level def or class after start
    next_def = src.find('\ndef ', start+1)
    next_class = src.find('\nclass ', start+1)
    end_candidates = [i for i in (next_def, next_class) if i != -1]
    end = min(end_candidates) if end_candidates else len(src)
    func_src = src[start:end]

    exec(func_src, ns)
    func = ns.get(funcname)
    if func is None:
        print('Failed to extract function')
        sys.exit(1)

    print('Running' , tries, 'isolated quick tests for ' + funcname + ':\n')

    """Print test summary."""
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    for t in tests:
        print('INPUT:', t)
        samples = [func(t[0]) for _ in range(tries)]
        c = Counter(samples)
        print("SAMPLES: ", dict(c))

        # validation
        expected_pct = t[1]
        expected_keys = set(expected_pct.keys())
        actual_keys = set(c.keys())
        missing = expected_keys - actual_keys
        unexpected = actual_keys - expected_keys

        if missing or unexpected:
            if missing:
                print("MISSING: ", sorted(missing))
            if unexpected:
                print("UNEXPECTED: ", sorted(unexpected))
            print("RESULT: FAILED (keys)")
            print('')
            continue

        failures = []
        for k, pct in expected_pct.items():
            expected_count = tries * (pct / 100.0)
            actual_count = c.get(k, 0)
            actual_pct = (actual_count / tries) * 100.0
            if abs(actual_pct - pct) > tolerance_pct:
                failures.append(
                    f"{k}: expected {pct:.1f}%, got {actual_pct:.1f}% "
                    f"({actual_count}/{tries})"
                )

        if failures:
            print("OUT OF RANGE: ")
            for line in failures:
                print(" - " + line)
            print("RESULT: FAILED (distribution)")
        else:
            print("RESULT: PASSED")
        print('')
