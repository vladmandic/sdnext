#!/usr/bin/env python
"""
API integration tests for UNET/DiT override architecture-mismatch self-heal.

Verifies the reactive behavior end-to-end against a running SD.Next instance:
when a UNET/DiT override does not match the base model's architecture, the load
drops the override, resets the UNET dropdown to ``Default``, and the base model
still generates, instead of crashing inside an arch-specific converter.

Covers:
- GET /sdapi/v1/unets, /sdapi/v1/sd-models  (discovery / sanity)
- baseline: base model with no override loads and generates
- mismatch self-heal: base model + wrong-arch override -> override dropped,
  sd_unet reset to Default, generation still succeeds
- match preserved (optional): a correct-arch override stays applied

Requires a running SD.Next instance with the relevant models on disk. Model and
UNET names are environment-specific, so pass them explicitly. Run with no model
args to just list what the server has available.

Usage:
    python test/test-override-mismatch-api.py \
        --url http://127.0.0.1:7860 \
        --base-model "Diffusers/lodestones/Chroma1-HD [0e0c60ece1]" \
        --mismatch-unet "novaOrangeAM_v15" \
        [--match-unet "<a correct-arch transformer single-file name>"] \
        [--steps 4]
"""

import sys
import time
import argparse
import requests
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


class OverrideMismatchAPITest:
    """Drives the override-mismatch self-heal scenarios over the HTTP API."""

    def __init__(self, base_url, base_model=None, mismatch_unet=None, match_unet=None, steps=4):
        self.base_url = base_url.rstrip('/')
        self.base_model = base_model
        self.mismatch_unet = mismatch_unet
        self.match_unet = match_unet
        self.steps = steps
        self.load_timeout = 900
        self.gen_timeout = 600
        self.passed = 0
        self.failed = 0
        self.skipped = 0

    # ---- low-level helpers -------------------------------------------------

    def _get(self, endpoint):
        r = requests.get(f'{self.base_url}{endpoint}', timeout=60, verify=False)
        r.raise_for_status()
        return r.json()

    def _post(self, endpoint, data=None, params=None, timeout=60):
        r = requests.post(f'{self.base_url}{endpoint}', json=data, params=params, timeout=timeout, verify=False)
        return r

    def record(self, ok, name, detail=''):
        tag = 'PASS' if ok else 'FAIL'
        self.passed += 1 if ok else 0
        self.failed += 0 if ok else 1
        line = f'  {tag}: {name}'
        if detail:
            line += f'  ({detail})'
        print(line, flush=True)

    def skip(self, name, reason):
        self.skipped += 1
        print(f'  SKIP: {name}  ({reason})', flush=True)

    # ---- mid-level operations ----------------------------------------------

    def set_options(self, **kwargs):
        r = self._post('/sdapi/v1/options', data=kwargs, timeout=self.load_timeout)
        return r.status_code == 200, (r.text[:200] if r.status_code != 200 else '')

    def reload(self, force=True):
        r = self._post('/sdapi/v1/reload-checkpoint', params={'force': str(force).lower()}, timeout=self.load_timeout)
        return r.status_code == 200, (r.text[:200] if r.status_code != 200 else '')

    def get_sd_unet(self):
        return self._get('/sdapi/v1/options').get('sd_unet')

    def generate(self):
        payload = {'prompt': 'a photo of a cat', 'steps': self.steps, 'width': 512, 'height': 512, 'save_images': False}
        t0 = time.time()
        r = self._post('/sdapi/v1/txt2img', data=payload, timeout=self.gen_timeout)
        elapsed = time.time() - t0
        if r.status_code != 200:
            return False, f'http {r.status_code}: {r.text[:160]}'
        body = r.json()
        images = body.get('images') or []
        if not images:
            return False, f'no images returned ({elapsed:.1f}s)'
        return True, f'{elapsed:.1f}s'

    # ---- scenarios ---------------------------------------------------------

    def test_discovery(self):
        print('=== discovery ===', flush=True)
        try:
            unets = self._get('/sdapi/v1/unets')
            self.record(isinstance(unets, list), 'GET /sdapi/v1/unets', f'{len(unets)} unets')
            models = self._get('/sdapi/v1/sd-models')
            self.record(isinstance(models, list), 'GET /sdapi/v1/sd-models', f'{len(models)} models')
            if not (self.base_model and self.mismatch_unet):
                print('  available UNET names:', flush=True)
                for u in unets:
                    print(f'    - {u.get("name")}', flush=True)
                print('  available model titles:', flush=True)
                for m in models[:40]:
                    print(f'    - {m.get("title")}', flush=True)
        except Exception as e:
            self.record(False, 'discovery', f'exception: {e}')

    def test_baseline(self):
        print('=== baseline (base model, no override) ===', flush=True)
        if not self.base_model:
            self.skip('baseline', 'no --base-model')
            return False
        ok, err = self.set_options(sd_unet='Default', sd_model_checkpoint=self.base_model)
        if not ok:
            self.record(False, 'set base model + Default unet', err)
            return False
        ok, err = self.reload(force=True)
        if not ok:
            self.record(False, 'reload base model', err)
            return False
        ok, detail = self.generate()
        self.record(ok, 'generate with base model', detail)
        return ok

    def test_mismatch_self_heal(self):
        print('=== mismatch self-heal ===', flush=True)
        if not (self.base_model and self.mismatch_unet):
            self.skip('mismatch self-heal', 'needs --base-model and --mismatch-unet')
            return
        # configure base model with the wrong-arch override, then force a clean reload
        ok, err = self.set_options(sd_model_checkpoint=self.base_model, sd_unet=self.mismatch_unet)
        if not ok:
            self.record(False, 'set base model + mismatch override', err)
            return
        ok, err = self.reload(force=True)
        # the reload itself must not error out (the whole point of the fix)
        self.record(ok, 'reload does not error on mismatched override', err)
        # override must self-heal back to Default
        healed = self.get_sd_unet()
        self.record(healed == 'Default', 'sd_unet reset to Default', f'sd_unet={healed!r}')
        # base model must still be usable
        gen_ok, detail = self.generate()
        self.record(gen_ok, 'generate after self-heal', detail)

    def test_match_preserved(self):
        print('=== match preserved (no false drop) ===', flush=True)
        if not (self.base_model and self.match_unet):
            self.skip('match preserved', 'no --match-unet')
            return
        ok, err = self.set_options(sd_model_checkpoint=self.base_model, sd_unet=self.match_unet)
        if not ok:
            self.record(False, 'set base model + matching override', err)
            return
        ok, err = self.reload(force=True)
        self.record(ok, 'reload with matching override', err)
        kept = self.get_sd_unet()
        self.record(kept == self.match_unet, 'matching override kept (not dropped)', f'sd_unet={kept!r}')
        gen_ok, detail = self.generate()
        self.record(gen_ok, 'generate with matching override', detail)

    def cleanup(self):
        self.set_options(sd_unet='Default')

    def run(self):
        try:
            self.test_discovery()
            self.test_baseline()
            self.test_mismatch_self_heal()
            self.test_match_preserved()
        finally:
            self.cleanup()
        print('=== results ===', flush=True)
        print(f'  passed={self.passed} failed={self.failed} skipped={self.skipped}', flush=True)
        return self.failed == 0


def main():
    ap = argparse.ArgumentParser(description='Override arch-mismatch self-heal API tests')
    ap.add_argument('--url', default='http://127.0.0.1:7860', help='SD.Next base URL')
    ap.add_argument('--base-model', default=None, help='checkpoint title to load as the base (DiT arch)')
    ap.add_argument('--mismatch-unet', default=None, help='UNET name whose arch does NOT match the base')
    ap.add_argument('--match-unet', default=None, help='optional UNET name whose arch DOES match the base')
    ap.add_argument('--steps', type=int, default=4)
    args = ap.parse_args()

    try:
        requests.get(f'{args.url.rstrip("/")}/sdapi/v1/sd-models', timeout=10, verify=False)
    except Exception as e:
        print(f'cannot reach SD.Next at {args.url}: {e}', flush=True)
        return 2

    ok = OverrideMismatchAPITest(
        args.url, base_model=args.base_model, mismatch_unet=args.mismatch_unet,
        match_unet=args.match_unet, steps=args.steps,
    ).run()
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
