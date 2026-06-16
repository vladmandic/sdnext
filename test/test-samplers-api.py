#!/usr/bin/env python3
"""
Sampler differential tester (API).

Drives /sdapi/v1/txt2img and /sdapi/v1/img2img with the full per-sampler option matrix
against a live server: every value of every scheduler option (sigma method, prediction
type, timestep spacing, beta schedule, solver order, shift, low order, thresholding,
dynamic shift, rescale betas), one option at a time over a fully pinned baseline. Each
case is measured against the default and reported as applied / inert / marginal /
rejected / fallback. The tester pins schedulers_fallback off, so a selection the
sampler does not support fails server-side and is reported as rejected.

A fallback-semantics group asserts the sampler resolution edge cases as hard
expectations: sampler_name "Default" and an omitted sampler_name run the model default
scheduler, an unknown name is rejected by the API in both fallback modes, and invalid
selections with schedulers_fallback enabled fall back or redirect instead of failing
the request.

Modes:
    --sampler "ER-SDE"            deep dive: full matrix for one sampler
    --sampler "ER-SDE,UniPC"      deep dive for each listed sampler
    --sampler all                 every sampler the server reports
    --sweep                       reduced per-sampler matrix + capability report instead
                                  of the full deep matrix

The sweep emits an empirical capability report (which knobs actually change the output
per sampler), useful for auditing declared-but-inert preset keys and capability-table
drift. The infotext Scheduler class is recorded for every case: the Sampler field only
echoes the request, so the class is the ground truth for detecting silent fallback to
the model default. Laplacian variance is reported as a sharpness proxy.

The img2img pass exercises the scheduler noise-injection path: VP models use add_noise,
flow models use scale_noise. Every img2img case returning an image is the support signal.

Requires a running server with a model loaded. Scheduler settings are passed as direct
payload fields (schedulers_sigma, schedulers_shift, ...), the same path the UI uses.

    python test/test-samplers-api.py --arch flux --sampler ER-SDE
    python test/test-samplers-api.py --arch sdxl --sampler all --sweep --mode txt2img
    python test/test-samplers-api.py --url http://127.0.0.1:7860 --sampler "DPM++ 2M" --mode img2img --denoise 0.6

Paste the printed SUMMARY block (or the written JSON) back for interpretation. Generated
images are saved to --outdir with a banner stating endpoint/sampler/case/settings, so a
folder full of test output stays attributable; case montages are saved alongside.
"""
import argparse
import base64
import io
import json
import os
import re
import tempfile
import textwrap
import time

import numpy as np
import requests
from PIL import Image, ImageDraw, ImageFont

PROMPT = "close-up portrait of an elderly fisherman, deep weathered skin texture, individual silver beard hairs, sharp catchlight in the eyes, hand-knitted wool sweater with visible fibers, soft window light"
NEG = "blurry, smooth, plastic, low detail"

# one-factor-at-a-time axes: (label prefix, payload field, values to run)
# values are the UI choice lists verbatim; sliders use representative points
AXES = [
    ("sigma", "schedulers_sigma", ["karras", "betas", "exponential", "lambdas", "flowmatch"]),
    ("pred", "schedulers_prediction_type", ["default", "epsilon", "sample", "v_prediction", "flow_prediction"]),
    ("spacing", "schedulers_timestep_spacing", ["linspace", "leading", "trailing"]),
    ("beta", "schedulers_beta_schedule", ["linear", "scaled_linear", "squaredcos_cap_v2", "sigmoid"]),
    ("order", "schedulers_solver_order", [1, 2, 3, 4, 5]),
    ("shift", "schedulers_shift", [1, 6]),
    ("loworder", "schedulers_use_loworder", [True, False]),
    ("thresholding", "schedulers_use_thresholding", [True, False]),
    ("dynamic", "schedulers_dynamic_shift", [True, False]),
    ("rescale", "schedulers_rescale_betas", [True, False]),
]


# Pin every overridable scheduler knob to its option default so the server's persisted
# config cannot leak into the un-overridden baseline; each case flips exactly one knob.
# On flow models non-FlowMatch samplers must be pinned to flow_prediction, else
# create_sampler rejects them (a Flux pipe is not "flexible" like SDXL) and silently
# restores the model default.
# schedulers_fallback must be off: with it on, create_sampler redirects plain <-> FlowMatch
# variants by name on mismatched model types, so a case would no longer test the sampler
# it names (plain ER-SDE on Flux would silently run ER-SDE FlowMatch).
def baseline_for(arch):
    return {
        "schedulers_sigma": "default",
        "schedulers_prediction_type": "flow_prediction" if arch == "flux" else "default",
        "schedulers_timestep_spacing": "default",
        "schedulers_beta_schedule": "default",
        "schedulers_solver_order": 0,
        "schedulers_shift": 3,
        "schedulers_dynamic_shift": False,
        "schedulers_base_shift": 0.5,
        "schedulers_max_shift": 1.15,
        "schedulers_use_loworder": True,
        "schedulers_use_thresholding": False,
        "schedulers_rescale_betas": False,
        "schedulers_beta_start": 0,
        "schedulers_beta_end": 0,
        "schedulers_timesteps_range": 1000,
        "schedulers_timesteps": "",
        "uni_pc_variant": "bh2",
        "override_settings": {"schedulers_fallback": False},
    }


def post(url, path, payload, timeout=900):
    r = requests.post(f"{url}{path}", json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()


def get(url, path, timeout=60):
    r = requests.get(f"{url}{path}", timeout=timeout)
    r.raise_for_status()
    return r.json()


def decode(b64):
    raw = base64.b64decode(b64.split(',', 1)[-1])
    img = Image.open(io.BytesIO(raw)).convert('RGB')
    return np.asarray(img, dtype=np.float64)


def lap_var(arr):
    """Variance of the discrete Laplacian = sharpness proxy (higher is sharper)."""
    g = arr.mean(axis=2)
    lap = 4 * g[1:-1, 1:-1] - g[:-2, 1:-1] - g[2:, 1:-1] - g[1:-1, :-2] - g[1:-1, 2:]
    return float(lap.var())


def mad(a, b):
    """Mean absolute pixel difference on a 0-255 scale; nan if shapes differ."""
    if a.shape != b.shape:
        return float('nan')
    return float(np.abs(a - b).mean())


def norm_name(s):
    return re.sub(r'[^a-z0-9]', '', s.lower())


# name qualifiers that do not appear in scheduler class names
QUALIFIER_TOKENS = {"flowmatch", "a", "sgm", "edm", "1s", "2s", "3s", "2m", "3m", "sde", "inverse", "parallel", "solver"}


def scheduler_class(info):
    """The Scheduler class that actually ran. The response info JSON exposes Processed
    attributes at the top level; the scheduler class only appears inside the infotext
    parameter string, so it is extracted from there."""
    texts = info.get("infotexts") or []
    m = re.search(r'Scheduler: (\w+)', texts[0]) if texts and texts[0] else None
    return m.group(1) if m else ""


def ran_requested(requested, actual_cls):
    """Heuristic match of the requested sampler name against the Scheduler class that
    actually ran. A mismatch usually means silent fallback to the model default; reported
    as a warning, not a failure, since some short names cannot be matched reliably."""
    if not requested or requested == "Default":
        return True  # any scheduler class is a correct answer for the model default
    if not actual_cls:
        return True
    a = norm_name(actual_cls)
    tokens = [t for t in re.split(r'[^a-zA-Z0-9]+', requested.lower()) if t]
    family = ''.join(t for t in tokens if t not in QUALIFIER_TOKENS)
    for cand in (norm_name(requested), norm_name(requested.replace("FlowMatch", "")), family):
        if len(cand) >= 3 and cand in a:
            return True
    return bool(tokens) and len(tokens[0]) >= 3 and tokens[0] in a


def case_label(prefix, value):
    if value is True:
        return f"{prefix}_on"
    if value is False:
        return f"{prefix}_off"
    return f"{prefix}_{value}"


def build_matrix(sampler, arch, all_names, deep, baseline):
    """Case list + assertions for one sampler. Cases: (label, sampler_name, payload
    overrides). Assertions: (label, a, b, relation) with relation same | probe.

    Each AXES value becomes one case flipping a single option over the pinned baseline,
    measured against the default case as a probe. Values matching the baseline pin are
    covered by the default case itself. Sweep mode keeps only default and sigma=karras.
    Hard "same" assertions cover known invariants: a plain sampler matches its FlowMatch
    sibling on flow models, and explicit epsilon matches default on epsilon models."""
    cases = [("default", sampler, {})]
    assertions = []

    for prefix, field, values in AXES:
        for value in values:
            if value == baseline.get(field):
                continue  # the default case already runs this exact request
            label = case_label(prefix, value)
            cases.append((label, sampler, {field: value}))
            if not deep and (prefix, value) != ("sigma", "karras"):
                continue
            assertions.append((f"{prefix}={value}", label, "default", "probe"))
    if not deep:
        cases = [c for c in cases if c[0] in ("default", "sigma_karras")]

    if deep and arch == "flux" and 'FlowMatch' not in sampler and f"{sampler} FlowMatch" in all_names:
        cases.append(("flowmatch_variant", f"{sampler} FlowMatch", {}))
        assertions.append(("plain == FlowMatch (shift parity)", "default", "flowmatch_variant", "same"))
    if deep and arch == "sdxl":
        assertions.append(("explicit epsilon == default", "pred_epsilon", "default", "same"))

    if deep:
        fc, fa = fallback_matrix(sampler, arch, all_names)
        cases += fc
        assertions += fa

    return cases, assertions


def fallback_matrix(sampler, arch, all_names):
    """Sampler resolution and fallback semantics, asserted as hard expectations:
    sampler_name "Default" and an omitted sampler_name both run the model default
    scheduler; an unknown name is rejected by the API (404) in both fallback modes;
    with schedulers_fallback enabled, invalid selections (unsupported sigma method,
    out-of-range solver order, mismatched prediction type, plain name on a flow model)
    still generate by falling back or redirecting instead of failing the request."""
    fb_on = {"override_settings": {"schedulers_fallback": True}}
    cases = [
        ("fb_default_name", "Default", {}),
        ("fb_unspecified", None, {}),
        ("fb_unknown_strict", "__no_such_sampler__", {}),
        ("fb_unknown_loose", "__no_such_sampler__", dict(fb_on)),
        ("fb_on_sigma_lambdas", sampler, {"schedulers_sigma": "lambdas", **fb_on}),
        ("fb_on_order_5", sampler, {"schedulers_solver_order": 5, **fb_on}),
    ]
    assertions = [
        ('sampler "Default" generates', "fb_default_name", None, "ok"),
        ("omitted sampler_name generates", "fb_unspecified", None, "ok"),
        ("unknown sampler rejected, fallback off", "fb_unknown_strict", None, "rejected"),
        ("unknown sampler rejected, fallback on", "fb_unknown_loose", None, "rejected"),
        ("fallback on: unsupported sigma generates", "fb_on_sigma_lambdas", None, "ok"),
        ("fallback on: invalid solver order generates", "fb_on_order_5", None, "ok"),
    ]
    if arch == "flux":
        cases.append(("fb_on_pred_epsilon", sampler, {"schedulers_prediction_type": "epsilon", **fb_on}))
        assertions.append(("fallback on: prediction mismatch generates", "fb_on_pred_epsilon", None, "ok"))
        if 'FlowMatch' not in sampler and f"{sampler} FlowMatch" in all_names:
            cases.append(("fb_on_redirect", sampler, {"schedulers_prediction_type": "default", **fb_on}))
            assertions.append(("fallback on: plain name redirects and generates", "fb_on_redirect", None, "ok"))
    return cases, assertions


def generate(url, endpoint, sampler, overrides, args, init_b64=None):
    payload = {
        "prompt": PROMPT, "negative_prompt": NEG,
        "seed": args.seed, "steps": args.steps, "cfg_scale": args.cfg,
        "width": args.width, "height": args.height,
        "sampler_name": sampler, "batch_size": 1, "n_iter": 1,
        "save_images": bool(args.save_images), "send_images": True,
    }
    if endpoint == "img2img":
        payload["init_images"] = [init_b64]
        payload["denoising_strength"] = args.denoise
    payload.update(args.baseline)
    payload.update(overrides)
    if sampler is None:  # exercise the API default for an omitted sampler_name
        payload.pop("sampler_name", None)
    t0 = time.time()
    d = post(url, f"/sdapi/v1/{endpoint}", payload)
    dt = time.time() - t0
    if not d.get("images"):
        raise RuntimeError(f"no image returned for {endpoint} {sampler} {overrides}")
    b64 = d["images"][0]
    info = {}
    try:
        info = json.loads(d.get("info", "{}"))
    except (ValueError, TypeError):
        pass
    return decode(b64), b64, dt, info


def classify_error(e):
    """Compact error string. HTTP rejections carry the server's own reason from the JSON
    error body (detail for HTTPExceptions, errors for exceptions caught by middleware)."""
    if isinstance(e, requests.HTTPError) and e.response is not None:
        reason = ""
        try:
            body = e.response.json()
            reason = str(body.get("detail") or body.get("errors") or body.get("error") or "").strip()
        except ValueError:
            pass
        return f"http {e.response.status_code}" + (f": {reason[:160]}" if reason else "")
    return str(e)


def save_montage(results, tag, args):
    """Labeled contact sheet of every case for one-glance visual inspection; failed
    cases appear as their black error placeholders."""
    items = [(k, v) for k, v in results.items() if "arr" in v]
    if not items:
        return
    thumb, cols, pad, labelh = 360, 4, 12, 20
    rows = (len(items) + cols - 1) // cols
    cellw, cellh = thumb + pad, thumb + labelh + pad
    canvas = Image.new("RGB", (cols * cellw + pad, rows * cellh + pad), (25, 25, 25))
    draw = ImageDraw.Draw(canvas)
    f = font(14)
    for i, (label, v) in enumerate(items):
        r, c = divmod(i, cols)
        x, y = pad + c * cellw, pad + r * cellh
        thumbimg = Image.fromarray(v["arr"].astype(np.uint8)).resize((thumb, thumb))
        canvas.paste(thumbimg, (x, y + labelh))
        txt = f"{label}  lap={v['lap_var']:.0f}" if v.get("ok") else f"{label}  ERROR"
        draw.text((x, y + 3), txt, fill=(230, 230, 230) if v.get("ok") else (255, 120, 120), font=f)
    out = os.path.join(args.outdir, f"montage_{tag}.png")
    canvas.save(out)
    print(f"  montage: {out}")


def run_case(url, endpoint, label, sampler, overrides, args, init_b64=None):
    outpath = os.path.join(args.outdir, f"{endpoint}_{slug(sampler) if sampler else 'unspecified'}_{label}.png")
    try:
        arr, b64, dt, info = generate(url, endpoint, sampler, overrides, args, init_b64)
    except (requests.RequestException, RuntimeError) as e:
        err = classify_error(e)
        placeholder = np.asarray(error_image(args.width, args.height, f"Error: {err}"), dtype=np.float64)
        lines = banner_lines(endpoint, label, sampler, overrides, args, f"error={err}  seed={args.seed} steps={args.steps} cfg={args.cfg}")
        annotate(placeholder, lines).save(outpath)
        return {"sampler": sampler, "overrides": overrides, "ok": False, "error": err, "arr": placeholder}
    actual_cls = scheduler_class(info)
    res = {
        "sampler": sampler, "overrides": overrides,
        "lap_var": round(lap_var(arr), 2),
        "mean": round(float(arr.mean()), 2), "std": round(float(arr.std()), 2),
        "time": round(dt, 1),
        "actual_cls": actual_cls,
        "ran_ok": ran_requested(sampler, actual_cls),
        "degenerate": float(arr.std()) < 2.0,
        "arr": arr, "b64": b64, "ok": True,
    }
    lines = banner_lines(endpoint, label, sampler, overrides, args, f"cls={actual_cls or '?'}  lap_var={res['lap_var']}  std={res['std']}  seed={args.seed} steps={args.steps} cfg={args.cfg}")
    annotate(arr, lines).save(outpath)
    return res


def slug(s):
    return re.sub(r'[^A-Za-z0-9]+', '_', s).strip('_')


def font(size):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1 has no size parameter
        return ImageFont.load_default()


def error_image(width, height, message):
    """Black placeholder at the case resolution with the error centered in white, so a
    failed case still has a visual slot on disk and in the montage."""
    img = Image.new("RGB", (width, height), (0, 0, 0))
    draw = ImageDraw.Draw(img)
    f = font(max(24, width // 24))
    lines = textwrap.wrap(message, width=36) or [message]
    boxes = [draw.textbbox((0, 0), line, font=f) for line in lines]
    spacing = 8
    block = sum(b[3] - b[1] for b in boxes) + spacing * (len(lines) - 1)
    y = (height - block) // 2
    for line, b in zip(lines, boxes):
        draw.text(((width - (b[2] - b[0])) // 2, y), line, fill=(255, 255, 255), font=f)
        y += (b[3] - b[1]) + spacing
    return img


def banner_lines(endpoint, label, sampler, overrides, args, tail):
    """Annotation banner: requested configuration (baseline + this case's override)."""
    eff = {**args.baseline, **overrides}
    return [
        f"{endpoint}  {sampler or '(unspecified)'}  case={label}",
        f"pred={eff['schedulers_prediction_type']}  sigma={eff['schedulers_sigma']}  spacing={eff['schedulers_timestep_spacing']}  beta={eff['schedulers_beta_schedule']}",
        f"order={eff['schedulers_solver_order']}  shift={eff['schedulers_shift']}  loworder={eff['schedulers_use_loworder']}  thresh={eff['schedulers_use_thresholding']}  dynamic={eff['schedulers_dynamic_shift']}  rescale={eff['schedulers_rescale_betas']}",
        tail,
    ]


def annotate(arr, lines):
    """Banner-annotate an image for disk inspection; the pixels under test stay untouched."""
    img = Image.fromarray(arr.astype(np.uint8))
    lineh, padx, pady = 22, 8, 5
    barh = lineh * len(lines) + 2 * pady
    canvas = Image.new("RGB", (img.width, img.height + barh), (15, 15, 15))
    canvas.paste(img, (0, barh))
    draw = ImageDraw.Draw(canvas)
    f = font(16)
    for i, line in enumerate(lines):
        draw.text((padx, pady + lineh * i), line, fill=(235, 235, 235), font=f)
    return canvas


def run_pass(endpoint, url, cases, args, tag, init_b64=None):
    print(f"\n--- {endpoint} pass: {len(cases)} generations ---")
    results = {}
    for label, sampler, overrides in cases:
        res = run_case(url, endpoint, label, sampler, overrides, args, init_b64)
        results[label] = res
        if res["ok"]:
            sigma = overrides.get("schedulers_sigma", "-")
            flags = "".join([" RAN?" if not res["ran_ok"] else "", " DEGENERATE" if res["degenerate"] else ""])
            print(f"  [ok]   {label:18s} sampler='{sampler}' sigma={sigma:11s} lap_var={res['lap_var']:8.1f} cls={res['actual_cls']} t={res['time']:.1f}s{flags}")
        else:
            # rejection is data, not a test failure: probe cases run with fallback off so
            # an unsupported selection is supposed to be refused, and the fb_unknown cases
            # assert the refusal; pass/fail judgment happens in the assertions table
            tag = "rej" if res["error"].startswith("http ") else "err"
            print(f"  [{tag}]  {label:18s} sampler='{sampler}' {res['error']}")
    save_montage(results, tag, args)
    return results


def probe_verdict(res, default_res, args):
    """Classify one case against the default: rejected (server refused), fallback (a
    different scheduler class ran), or applied/inert/marginal by MAD."""
    if not res.get("ok"):
        return "rejected", None
    if not res.get("ran_ok"):
        return "fallback", None
    if not default_res.get("ok"):
        return "no-baseline", None
    m = mad(res["arr"], default_res["arr"])
    return ("applied" if m > args.diff_thresh else "inert" if m < args.same_thresh else "marginal"), m


def report(endpoint, results, assertions, arch, args):
    present = {k for k, v in results.items() if v.get("ok")}
    print(f"\n=== {endpoint} comparisons (MAD 0-255; same<{args.same_thresh}, diff>{args.diff_thresh}) ===")
    if endpoint == "img2img":
        path = "add_noise (VP)" if arch == "sdxl" else "scale_noise (flow)" if arch == "flux" else "noise-injection"
        print(f"  (every case returning an image exercises the sampler's {path} path)")
    comparisons = []
    counts = {}
    npass = 0
    total = 0
    for label, a, b, rel in assertions:
        if rel in ("ok", "rejected"):
            total += 1
            res = results.get(a, {})
            passed = bool(res.get("ok")) if rel == "ok" else not res.get("ok")
            npass += passed
            verdict = "PASS" if passed else "FAIL"
            if rel == "ok":
                detail = res.get("actual_cls") or "generated" if passed else res.get("error", "no result")
            else:
                detail = res.get("error", "no result") if passed else "unexpectedly generated an image"
            comparisons.append({"check": label, "a": a, "b": None, "expect": rel, "verdict": verdict, "detail": detail})
            print(f"  [{verdict}] {label:34s} {a}: {detail}")
            continue
        if rel == "probe":
            verdict, m = probe_verdict(results.get(a, {}), results.get(b, {}), args)
            counts[verdict] = counts.get(verdict, 0) + 1
            detail = results.get(a, {}).get("error") if verdict == "rejected" else results.get(a, {}).get("actual_cls") if verdict == "fallback" else None
            comparisons.append({"check": label, "a": a, "b": b, "expect": rel, "mad": round(m, 2) if m is not None else None, "verdict": verdict, "detail": detail})
            madtxt = f"MAD={m:7.2f}" if m is not None else f"({detail})"
            print(f"  [probe] {label:34s} {madtxt} -> {verdict}")
            continue
        if a not in present or b not in present:
            continue
        # a no-op comparison is meaningless when either side silently fell back to default
        if rel == "same" and not (results[a]["ran_ok"] and results[b]["ran_ok"]):
            print(f"  [skip] {label:34s} (sampler fallback detected, parity not meaningful)")
            continue
        m = mad(results[a]["arr"], results[b]["arr"])
        total += 1
        ok = (m < args.same_thresh) if rel == "same" else (m > args.diff_thresh)
        npass += ok
        verdict = "PASS" if ok else "FAIL"
        comparisons.append({"check": label, "a": a, "b": b, "expect": rel, "mad": round(m, 2), "verdict": verdict})
        print(f"  [{verdict}] {label:34s} {a} vs {b}: MAD={m:7.2f} (expect {rel})")
    for label in sorted(present):
        if not results[label]["ran_ok"]:
            print(f"  [warn] {label}: Scheduler class '{results[label]['actual_cls']}' does not match requested '{results[label]['sampler']}' (silent fallback?)")
        if results[label]["degenerate"]:
            print(f"  [warn] {label}: degenerate output (std={results[label]['std']}), likely black/blank image")
    if {"shift_1", "shift_6"} <= present:
        s1, s6 = results["shift_1"]["lap_var"], results["shift_6"]["lap_var"]
        print(f"  sharpness: shift_1 lap_var={s1:.1f} vs shift_6 lap_var={s6:.1f}")
    countstxt = "  ".join(f"{k}={v}" for k, v in sorted(counts.items()))
    print(f"  {endpoint}: invariants {npass}/{total} passed | probes: {countstxt}")
    return {"comparisons": comparisons, "passed": npass, "total": total, "probes": counts}


def strip(results):
    return {k: {kk: vv for kk, vv in v.items() if kk not in ("arr", "b64")} for k, v in results.items()}


def run_deep(url, sampler, arch, all_names, args, dump):
    grand_pass = grand_total = 0
    cases, assertions = build_matrix(sampler, arch, all_names, deep=True, baseline=args.baseline)
    if args.mode in ("txt2img", "both"):
        res = run_pass("txt2img", url, cases, args, tag=f"txt2img_{slug(sampler)}")
        rep = report("txt2img", res, assertions, arch, args)
        dump["txt2img"] = {"cases": strip(res), **rep}
        grand_pass += rep["passed"]
        grand_total += rep["total"]
    if args.mode in ("img2img", "both"):
        print(f"\nGenerating img2img base image ({sampler} default)...")
        _, base_b64, _, _ = generate(url, "txt2img", sampler, {}, args)
        res = run_pass("img2img", url, cases, args, tag=f"img2img_{slug(sampler)}", init_b64=base_b64)
        rep = report("img2img", res, assertions, arch, args)
        dump["img2img"] = {"cases": strip(res), **rep}
        grand_pass += rep["passed"]
        grand_total += rep["total"]
    return grand_pass, grand_total


def run_sweep(url, samplers, arch, all_names, args, dump):
    """Reduced matrix per sampler (default + karras + img2img); emits an empirical
    capability table instead of hard pass/fail."""
    rows = []
    defaults_montage = {}
    do_img = args.mode in ("img2img", "both")
    for sampler in samplers:
        cases, assertions = build_matrix(sampler, arch, all_names, deep=False, baseline=args.baseline)
        print(f"\n##### {sampler}")
        res = run_pass("txt2img", url, cases, args, tag=f"sweep_{slug(sampler)}")
        row = {"sampler": sampler, "karras": None, "karras_mad": None,
               "txt2img": False, "img2img": None, "actual_cls": "", "notes": []}
        d = res.get("default", {})
        if "arr" in d:
            defaults_montage[sampler] = d
        if d.get("ok"):
            row["txt2img"] = True
            row["actual_cls"] = d["actual_cls"]
            if not d["ran_ok"]:
                row["notes"].append("class mismatch (fallback?)")
            if d["degenerate"]:
                row["notes"].append("degenerate output")
        else:
            row["notes"].append(d.get("error", "generation failed"))
        for _label, a, _b, rel in assertions:
            if rel != "probe":
                continue
            verdict, m = probe_verdict(res.get(a, {}), d, args)
            row["karras"] = verdict
            row["karras_mad"] = round(m, 2) if m is not None else None
            if verdict != "applied":
                row["notes"].append(f"karras: {verdict}" + (f" (MAD={m:.2f})" if m is not None else ""))
        if do_img and d.get("ok"):
            ires = run_case(url, "img2img", "default", sampler, {}, args, init_b64=d["b64"])
            row["img2img"] = bool(ires.get("ok"))
            if not ires.get("ok"):
                row["notes"].append(f"img2img: {ires.get('error')}")
        rows.append(row)
        dump.setdefault("sweep", {})[sampler] = {"cases": strip(res), "row": row}
    save_montage(defaults_montage, "sweep_defaults", args)

    print(f"\n=== sweep capability report ({arch}) ===")
    lines = ["| Sampler | txt2img | Scheduler class | Karras | img2img | Notes |",
             "| --- | --- | --- | --- | --- | --- |"]
    for r in rows:
        img = {True: "yes", False: "NO", None: "-"}[r["img2img"]]
        notes = "; ".join(r["notes"]) or "-"
        lines.append(f"| {r['sampler']} | {'yes' if r['txt2img'] else 'NO'} | {r['actual_cls'] or '-'} | {r['karras'] or '-'} | {img} | {notes} |")
    table = "\n".join(lines)
    print(table)
    rpath = os.path.join(args.outdir, f"capability_report_{arch}.md")
    with open(rpath, "w", encoding="utf-8") as f:
        f.write(f"# Sampler capability report ({arch})\n\n{table}\n")
    print(f"\nreport: {rpath}")
    issues = sum(1 for r in rows if r["notes"])
    return len(rows) - issues, len(rows)


def main():
    ap = argparse.ArgumentParser(description="Sampler differential tester (API)")
    ap.add_argument("--url", default="http://127.0.0.1:7860", help="server URL (default SD.Next port 7860)")
    ap.add_argument("--sampler", required=True, help="sampler name, comma-separated list, or 'all'")
    ap.add_argument("--sweep", action="store_true", help="reduced per-sampler matrix + empirical capability report instead of the full deep matrix")
    ap.add_argument("--arch", choices=["sdxl", "flux", "auto"], default="auto")
    ap.add_argument("--mode", choices=["txt2img", "img2img", "both"], default="both")
    ap.add_argument("--model", default=None, help="checkpoint name to load first (see /sdapi/v1/sd-models)")
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--cfg", type=float, default=None, help="cfg/guidance (default 6 sdxl, 4 flux)")
    ap.add_argument("--denoise", type=float, default=0.6, help="img2img denoising strength")
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--outdir", default=os.path.join(tempfile.gettempdir(), "samplers_api"), help="output dir for images + montage (default: <system temp>/samplers_api)")
    ap.add_argument("--save-images", action="store_true", help="also save originals server-side into the standard output folders (off by default to keep automated runs out of the gallery)")
    ap.add_argument("--same-thresh", type=float, default=1.0, help="MAD below this = identical")
    ap.add_argument("--diff-thresh", type=float, default=3.0, help="MAD above this = different")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    if args.model:
        print(f"Loading model: {args.model}")
        post(args.url, "/sdapi/v1/options", {"sd_model_checkpoint": args.model})
        time.sleep(2)

    arch = args.arch
    model_name = ""
    try:
        model_name = str(get(args.url, "/sdapi/v1/options").get("sd_model_checkpoint", ""))
    except requests.RequestException as e:
        print(f"warn: could not read options: {e}")
    if arch == "auto":
        low = model_name.lower()
        arch = "flux" if "flux" in low else "sdxl" if "xl" in low else "generic"
        print(f"auto-detected arch={arch} from model='{model_name}' (override with --arch)")
    if args.cfg is None:
        args.cfg = 4.0 if arch == "flux" else 6.0
    case_arch = arch if arch in ("sdxl", "flux") else "generic"
    args.baseline = baseline_for(case_arch)

    server_samplers = get(args.url, "/sdapi/v1/samplers")
    all_names = {s["name"] for s in server_samplers}
    if args.sampler.strip().lower() == "all":
        targets = sorted(all_names)
    else:
        targets = [s.strip() for s in args.sampler.split(",") if s.strip()]
        for name in targets:
            if name not in all_names:
                ap.error(f"sampler '{name}' not in server list; see /sdapi/v1/samplers")

    print(f"\nModel: {model_name}")
    print(f"Arch: {arch}  mode={args.mode}  samplers={len(targets)} ({'sweep' if args.sweep else 'deep'})  baseline_prediction={args.baseline['schedulers_prediction_type']}  steps={args.steps} seed={args.seed} cfg={args.cfg} denoise={args.denoise} res={args.width}x{args.height}")
    if not args.sweep and len(targets) > 3:
        print(f"warn: deep mode runs the full matrix for each of {len(targets)} samplers; --sweep is the reduced audit")

    dump = {"model": model_name, "arch": arch, "mode": args.mode,
            "samplers": targets, "baseline": {k: v for k, v in args.baseline.items() if k != "override_settings"},
            "params": {"steps": args.steps, "seed": args.seed, "cfg": args.cfg, "denoise": args.denoise, "res": [args.width, args.height]}}

    if args.sweep:
        clean, total = run_sweep(args.url, targets, case_arch, all_names, args, dump)
        summary = f"{clean}/{total} samplers clean"
    else:
        grand_pass = grand_total = 0
        for name in targets:
            if len(targets) > 1:
                print(f"\n##### {name}")
            sub = dump.setdefault("deep", {}).setdefault(name, {})
            npass, total = run_deep(args.url, name, case_arch, all_names, args, sub)
            grand_pass += npass
            grand_total += total
        summary = f"{grand_pass}/{grand_total} invariants passed"

    print(f"\n=== SUMMARY: {summary} | arch={arch} mode={args.mode} | images in {args.outdir} ===")
    jpath = os.path.join(args.outdir, "samplers_api_results.json")
    with open(jpath, "w", encoding="utf-8") as f:
        json.dump(dump, f, indent=2)
    print(f"JSON: {jpath}")


if __name__ == "__main__":
    main()
