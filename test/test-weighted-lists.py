#!/usr/bin/env python

import sys
import os
from collections import Counter


script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(script_dir)


# library
fn = r'./modules/styles.py'
# tested function
funcname = 'select_from_weighted_list'
# random needed
ns = {'Dict': dict, 'random': __import__('random')}
# number of samples to test
tries = 10000
# allowed deviation in percentage points
tolerance_pct = 2.0
# tests
tests = [
    # - empty
    ["", { '': 100 } ],
    # - no weights
    [ "red|blonde|black", { 'black': 33, 'red': 33, 'blonde': 33 } ],
    # - list with empty entry
    [ "red||black", { 'black': 33, 'red': 33, '': 33 } ],
    # - list with repeated entry
    [ "red|blonde|blonde", { 'red': 33, 'blonde': 66 } ],
    # - full weights <= 1
    [ "red:0.1|blonde:0.9", { 'red': 10, 'blonde': 90 } ],
    # - weights > 1 to test normalization
    [ "red:1|blonde:2|black:5", { 'red': 12.5, 'blonde': 25, 'black': 62.5 } ],
    # - disabling 0 weights to force one result
    [ "red:0|blonde|black:0", { 'blonde': 100 } ],
    # - weights <= 1 with distribution of the leftover
    [ "red:0.5|blonde|black:0.3|brown", { 'red': 0.5, 'blonde': 1.0, 'black': 0.3, 'brown': 1.0 } ],
    # - weights > 1, unweightes should get default of 1
    [ "red:2|blonde|black", { 'red': 50, 'blonde': 25, 'black': 25 } ],
    # - ignore content of ()
    [ "red:0.5|(blonde:1.3)", { 'red': 50, '(blonde:1.3)': 100 } ],
    # - ignore content of ()
    [ "red:0.5|(blonde:1.3):0.5", { 'red': 50, '(blonde:1.3)': 50 } ],
    # - ignore content of []
    [ "red:0.5|[stuff:1.3]", { 'red': 50, '[stuff:1.3]': 100 } ],
    # - ignore content of <>
    [ "red:0.5|<lora:1.0>", { 'red': 50, '<lora:1.0>': 100 } ],
    # - simple list, 1 entry with lora with weights
    [ "red|<lora:test:1.0>|black", { 'black': 33, 'red': 33, '<lora:test:1.0>': 33 } ],
    # - simple list, 1 entry with loraand comma
    [ "red|blonde <lora:test:1.0>|black", { 'black': 33, 'red': 33, 'blonde <lora:test:1.0>': 33 } ],
    # - simple list, 1 entry with lora and comma
    [ "red|blonde, <lora:test:1.0>|black", { 'black': 33, 'red': 33, 'blonde, <lora:test:1.0>': 33 } ],
    # - simple list, 1 entry with lora and comma
    [ "red|blonde, <lora:test:1.0>|black:2", { 'black': 50, 'red': 25, 'blonde, <lora:test:1.0>': 25 } ],
]

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

    exec(func_src, ns) # pylint: disable=exec-used
    func = ns.get(funcname)
    if func is None:
        print('Failed to extract function')
        sys.exit(1)

    print('Running' , tries, 'isolated quick tests for ' + funcname + ':\n')

    for t in tests:
        print('Input:', t[0])
        samples = [func(t[0]) for _ in range(tries)]
        c = Counter(samples)
        print("  Expected:", dict(t[1]))
        print("  Samples:", dict(c))

        # validation
        expected_pct = t[1]
        expected_keys = set(expected_pct.keys())
        actual_keys = set(c.keys())
        missing = expected_keys - actual_keys
        unexpected = actual_keys - expected_keys

        if missing or unexpected:
            if missing:
                print("  Missing: ", sorted(missing))
            if unexpected:
                print("  Unexpected: ", sorted(unexpected))
            print(" Result: FAIL keys")
            continue

        failures = []
        W = sum(expected_pct.values())
        deviations = {}
        for k, pct in expected_pct.items():
            pct = pct / W # normalize
            expected_count = tries * pct
            actual_count = c.get(k, 0)
            actual_pct = actual_count / tries
            deviation_pct = abs(actual_pct - pct)
            deviations[k] = deviation_pct
            if deviation_pct > tolerance_pct / 100.0:
                failures.append(
                    f"{k}: expected {pct:.1f}%, got {actual_pct:.1f}% "
                    f"({actual_count}/{tries})"
                )

        print("  Deviations:", {k: f"{v*100:.2f}%" for k, v in deviations.items()})
        if failures:
            for line in failures:
                print("    " + line)
            print("  Result: FAIL distribution")
        else:
            print("  Result: PASS")
