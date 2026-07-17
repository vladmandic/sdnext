#!/usr/bin/env python
"""LoRA fidelity analyzer for quantized base models.

Measures, in weight space, how faithfully a LoRA lands on an SDNQ-quantized
model. For every LoRA-targeted module it reports where the delta sits relative
to the quantization grid and what each apply path preserves:

- requantize path (dequantize + add + requantize, the fallback for
  non-factorable families): retention ``rho`` of the intended delta. On-grid
  rounding erases sub-step deltas down to a ``2/group_size`` floor, so low-bit
  formats (<=6 bits) typically show rho ~= 0.02-0.03.
- factor path (plain LoRA riding the svd side-channel): exact by construction;
  the tool verifies each module qualifies and flags families that fall back.
- unquantized modules: the LoRA applies exactly regardless.

Works offline against a pre-quantized SDNQ repo (stored tensors + config) or
a bf16 repo with simulated quantization settings, so a combination can be
assessed before committing to a quantized checkpoint.

Examples:
    python cli/lora-quant-fidelity.py --model vladmandic/Krea-2-Base-sdnq-hadamard-uint4 --arch krea2 --lora "~/models/Lora/Krea 2/krea2_turbo_distill_r256.safetensors"
    python cli/lora-quant-fidelity.py --model CalamitousFelicitousness/Krea-2-Base-Diffusers --arch krea2 --dtype uint4 --lora lora.safetensors --json report.json
"""

import os
import sys
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('SD_INSTALL_QUIET', '1')


def parse_cli():
    parser = argparse.ArgumentParser(description='lora-quant-fidelity')
    parser.add_argument('--model', required=True, help='model dir, transformer dir, or org/name repo id')
    parser.add_argument('--arch', default='generic', help='lora key resolver: a native arch (e.g. krea2, zimage, f2) or generic')
    parser.add_argument('--lora', required=True, nargs='+', help='lora safetensors file(s)')
    parser.add_argument('--dtype', default=None, help='simulate quantization of a bf16 repo at this sdnq dtype (e.g. uint4, int8)')
    parser.add_argument('--group', type=int, default=0, help='sdnq group_size for simulation')
    parser.add_argument('--hadamard-group', type=int, default=256, help='sdnq hadamard group for simulation')
    parser.add_argument('--sample', type=int, default=40, help='max modules analyzed per lora (evenly sampled)')
    parser.add_argument('--full', action='store_true', help='analyze every matched module')
    parser.add_argument('--json', default=None, help='write full report to this json file')
    parser.add_argument('--fail-under', type=float, default=None, help='exit 2 when effective fidelity of any lora is below this')
    return parser.parse_args()


cli_args = parse_cli()
sys.argv = [sys.argv[0]] # sdnext arg parsing during imports must not see tool args (prefix matching eats --model/--lora)

import modules.cmd_args # pylint: disable=wrong-import-position
import installer # pylint: disable=wrong-import-position
modules.cmd_args.parse_args()
installer.add_args(modules.cmd_args.parser)
modules.cmd_args.parsed, _unknown = modules.cmd_args.parser.parse_known_args([])

import torch # pylint: disable=wrong-import-position
from safetensors import safe_open # pylint: disable=wrong-import-position
from rich import print as rprint # pylint: disable=wrong-import-position

from modules.lora import native_adapter # pylint: disable=wrong-import-position
from modules.lora.lora_load import NATIVE_DISPATCH # pylint: disable=wrong-import-position
from sdnq.quantizer import sdnq_quantize_layer_weight # pylint: disable=wrong-import-position
from sdnq.quant_utils import rotate_hadamard # pylint: disable=wrong-import-position


MODEL_ROOTS = [
    os.path.expanduser('~/database/models/huggingface'),
    os.path.expanduser('~/database/models/Diffusers'),
]
FACTORABLE_SUFFIX = 'lora' # only plain lora groups are factor-path eligible
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def resolve_model_dir(spec):
    """Return the transformer directory for a local path or org/name repo id."""
    candidates = [spec, os.path.join(spec, 'transformer')]
    cache_name = 'models--' + spec.replace('/', '--')
    for root in MODEL_ROOTS:
        snap_root = os.path.join(root, cache_name, 'snapshots')
        if os.path.isdir(snap_root):
            for snap in sorted(os.listdir(snap_root), reverse=True):
                candidates.append(os.path.join(snap_root, snap, 'transformer'))
                candidates.append(os.path.join(snap_root, snap))
    for c in candidates:
        if os.path.isfile(os.path.join(c, 'config.json')):
            return c
    raise SystemExit(f'model not found: {spec}')


def resolve_arch(name):
    """Return the arch lora module for key resolution, or None for generic matching."""
    if name == 'generic':
        return None
    path = NATIVE_DISPATCH.get({'flux2': 'f2', 'ernie': 'ernieimage'}.get(name, name))
    if path is None:
        raise SystemExit(f'unknown arch {name}; choices: {sorted(NATIVE_DISPATCH)} or generic')
    import importlib
    return importlib.import_module(path)


def map_lora_modules(lora_path, arch_mod):
    """Return {model_module_path: (down, up, alpha)} for the file's plain-lora groups plus a family census."""
    with safe_open(lora_path, framework='pt', device='cpu') as f:
        state_dict = {k: f.get_tensor(k) for k in f.keys()}
    prefixes = getattr(arch_mod, 'KNOWN_PREFIXES', native_adapter.KNOWN_PREFIXES_DEFAULT)
    bare = getattr(arch_mod, 'BARE_DIFFUSERS_PREFIXES', ())
    resolve = getattr(arch_mod, 'resolve_targets', None) or (lambda prefix, base: [(base, None)])
    families = {}
    for fam, suffixes in (('lora', native_adapter.LORA_SUFFIXES), ('lokr', native_adapter.LOKR_SUFFIXES), ('loha', native_adapter.LOHA_SUFFIXES), ('oft', native_adapter.OFT_SUFFIXES)):
        groups = native_adapter.group_by_suffixes(state_dict, suffixes, prefixes=prefixes, bare_diffusers_prefixes=bare)
        if fam == 'lora':
            groups = {k: w for k, w in groups.items() if 'lora_down.weight' in w and 'lora_up.weight' in w}
        else:
            groups = {k: w for k, w in groups.items() if native_adapter.has_marker({f'x.{s}': None for s in w}, getattr(native_adapter, f'{fam.upper()}_MARKERS'))}
        families[fam] = groups
    mapped = {}
    for (prefix, base), w in families['lora'].items():
        for path, chunk in native_adapter.resolve_group_targets(resolve, prefix, base):
            if chunk is not None:
                continue # fused-split groups are arch-handled; out of scope here
            alpha = w.get('alpha')
            mapped[path] = (w['lora_down.weight'], w['lora_up.weight'], float(alpha) if alpha is not None else None)
    return mapped, {fam: len(g) for fam, g in families.items() if fam != 'lora' and len(g) > 0}


def resolve_transformer_cls(arch, class_name):
    """Prefer an sdnext-owned transformer class over the upstream diffusers one.

    Arches like krea2 keep checkpoint-style module names in their own class;
    the diffusers class of the same name expects diffusers-style keys and
    cannot load these state dicts.
    """
    if arch and class_name:
        try:
            import importlib
            pkg = importlib.import_module(f'pipelines.{ {"zimage": "z_image", "f2": "flux"}.get(arch, arch) }')
            for attr in dir(pkg):
                if attr.endswith('_SPEC'):
                    cls = getattr(getattr(pkg, attr), 'cls', None)
                    if cls is not None and cls.__name__ == class_name:
                        return cls
        except Exception:
            pass
    return None


def load_quantized_model(model_dir, arch=None, class_name=None):
    from sdnq.loader import load_sdnq_model
    model = load_sdnq_model(model_dir, model_cls=resolve_transformer_cls(arch, class_name), dtype=torch.bfloat16, device='cpu')
    layers = {}
    for name, module in model.named_modules():
        if getattr(module, 'sdnq_dequantizer', None) is not None:
            layers[name] = module
        elif module.__class__.__name__ == 'Linear' and getattr(module, 'weight', None) is not None:
            layers[name] = module
    del model # layer modules own their tensors; the dict keeps them alive
    return layers


class Bf16Repo:
    """Lazy per-module weight access for a sharded bf16 transformer repo."""

    def __init__(self, model_dir):
        self.model_dir = model_dir
        index = os.path.join(model_dir, 'diffusion_pytorch_model.safetensors.index.json')
        if os.path.isfile(index):
            with open(index, encoding='utf-8') as f:
                self.weight_map = json.load(f)['weight_map']
        else:
            single = os.path.join(model_dir, 'diffusion_pytorch_model.safetensors')
            with safe_open(single, framework='pt', device='cpu') as f:
                self.weight_map = dict.fromkeys(f.keys(), 'diffusion_pytorch_model.safetensors')

    def get(self, key):
        shard = self.weight_map.get(key)
        if shard is None:
            return None
        with safe_open(os.path.join(self.model_dir, shard), framework='pt', device='cpu') as f:
            return f.get_tensor(key)


def analyze_module(W_dq, deq_params, down, up, alpha):
    """Return fidelity metrics for one quantized module and one lora delta."""
    rank = down.shape[0]
    scale = (alpha / rank) if alpha is not None else 1.0
    D = (up.to(device, torch.float32) @ down.to(device, torch.float32)) * scale
    kw = dict(layer_class_name='Linear', torch_dtype=torch.bfloat16, group_size=deq_params['group_size'],
              hadamard_group_size=deq_params['hadamard_group_size'], use_hadamard=deq_params['use_hadamard'],
              weights_dtype=deq_params['weights_dtype'], use_svd=False, use_quantized_matmul=False, dequantize_fp32=False)
    deq2, data2 = sdnq_quantize_layer_weight(W_dq + D, **kw)
    W2 = deq2(data2['weight'], data2['scale'], zero_point=data2['zero_point'], svd_up=None, svd_down=None, dtype=torch.float32, skip_compile=True)
    E = W2 - W_dq
    nD = D.norm()
    rho = float(E.flatten() @ D.flatten() / nD.square())
    resid = float((E - D).norm() / nD)
    if deq_params['use_hadamard']:
        Dh = rotate_hadamard(D, group_size=deq_params['hadamard_group_size'])
    else:
        Dh = D
    step = data2['scale'].float()
    Dg = Dh.unflatten(-1, (step.shape[1], -1)) if step.ndim == 3 else Dh
    step_ratio = float((Dg.abs() / step).mean())
    crossers = float((Dg.abs() > step / 2).float().mean())
    return dict(rank=rank, rms_delta=float(D.pow(2).mean().sqrt()), rms_weight=float(W_dq.pow(2).mean().sqrt()),
                step_ratio=step_ratio, crossers=crossers, requant_rho=rho, requant_resid=resid)


def main():
    args = cli_args
    model_dir = resolve_model_dir(args.model)
    arch_mod = resolve_arch(args.arch)
    with open(os.path.join(model_dir, 'config.json'), encoding='utf-8') as f:
        model_config = json.load(f)
    pre_quantized = model_config.get('quantization_config') is not None

    quant_layers, bf16_repo = {}, None
    if pre_quantized:
        rprint(f'model: "{model_dir}" pre-quantized={pre_quantized}')
        quant_layers = load_quantized_model(model_dir, arch=args.arch, class_name=model_config.get('_class_name'))
    else:
        bf16_repo = Bf16Repo(model_dir)
        if args.dtype is None:
            rprint('model is not quantized and no --dtype given: loras apply exactly, nothing to analyze')
            return 0
        rprint(f'model: "{model_dir}" simulating dtype={args.dtype} group={args.group} hadamard={args.hadamard_group}')

    report = {'model': model_dir, 'pre_quantized': pre_quantized, 'loras': []}
    worst_effective = 1.0
    for lora_path in args.lora:
        lora_path = os.path.expanduser(lora_path)
        mapped, other_families = map_lora_modules(lora_path, arch_mod)
        rows, unquantized, unmatched = [], [], []
        keys = sorted(mapped)
        if not args.full and len(keys) > args.sample:
            keys = keys[::max(1, len(keys) // args.sample)][:args.sample]
        for path in keys:
            down, up, alpha = mapped[path]
            if down.ndim != 2 or up.ndim != 2:
                continue
            if pre_quantized:
                layer = quant_layers.get(path)
                if layer is None:
                    unmatched.append(path)
                    continue
                deq = getattr(layer, 'sdnq_dequantizer', None)
                if deq is None:
                    unquantized.append(path)
                    continue
                W_dq = deq(layer.weight, layer.scale, zero_point=layer.zero_point, svd_up=layer.svd_up, svd_down=layer.svd_down,
                           skip_quantized_matmul=deq.use_quantized_matmul, dtype=torch.float32, skip_compile=True).to(device)
                params = dict(weights_dtype=deq.weights_dtype, group_size=deq.group_size, hadamard_group_size=deq.hadamard_group_size, use_hadamard=deq.use_hadamard)
            else:
                W = bf16_repo.get(f'{path}.weight')
                if W is None:
                    unmatched.append(path)
                    continue
                deq0, data0 = sdnq_quantize_layer_weight(W.to(device, torch.float32), layer_class_name='Linear', weights_dtype=args.dtype,
                                                         group_size=args.group, hadamard_group_size=args.hadamard_group, use_hadamard=args.hadamard_group > 0,
                                                         use_svd=False, use_quantized_matmul=False, dequantize_fp32=False, torch_dtype=torch.bfloat16)
                W_dq = deq0(data0['weight'], data0['scale'], zero_point=data0['zero_point'], svd_up=None, svd_down=None, dtype=torch.float32, skip_compile=True)
                params = dict(weights_dtype=args.dtype, group_size=deq0.group_size, hadamard_group_size=deq0.hadamard_group_size, use_hadamard=deq0.use_hadamard)
            row = analyze_module(W_dq, params, down, up, alpha)
            row['module'] = path
            row['dtype'] = params['weights_dtype']
            rows.append(row)
            del W_dq
            if device.type == 'cuda':
                torch.cuda.empty_cache()

        rhos = sorted(r['requant_rho'] for r in rows)
        median_rho = rhos[len(rhos) // 2] if rhos else 1.0
        effective = 1.0 if len(other_families) == 0 else median_rho # factor path covers plain lora exactly
        worst_effective = min(worst_effective, effective)
        rprint(f'\nlora: "{os.path.basename(lora_path)}" targets={len(mapped)} analyzed={len(rows)} unquantized={len(unquantized)} unmatched={len(unmatched)} other_families={other_families or "none"}')
        rprint(f'  requantize path: median rho={median_rho:.3f} (fallback families would land at this fidelity)')
        rprint(f'  factor path:     {"exact (plain lora, all analyzed modules eligible)" if len(other_families) == 0 else "partial: non-lora families fall back to requantize"}')
        if rows:
            worst = sorted(rows, key=lambda r: r['requant_rho'])[:5]
            rprint('  lowest-retention modules (requantize path):')
            for r in worst:
                rprint(f'    {r["module"]:52s} dtype={r["dtype"]} step-ratio={r["step_ratio"]:.3f} crossers={r["crossers"]*100:5.1f}% rho={r["requant_rho"]:.3f}')
        report['loras'].append({'file': lora_path, 'targets': len(mapped), 'unquantized': unquantized, 'unmatched': unmatched,
                                'other_families': other_families, 'median_requant_rho': median_rho, 'effective_fidelity': effective, 'modules': rows})

    if args.json:
        with open(args.json, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)
        rprint(f'\nreport: "{args.json}"')
    if args.fail_under is not None and worst_effective < args.fail_under:
        rprint(f'FAIL: effective fidelity {worst_effective:.3f} < {args.fail_under}')
        return 2
    return 0


if __name__ == '__main__':
    with torch.inference_mode():
        sys.exit(main())
