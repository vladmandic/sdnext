#!/usr/bin/env python
"""LoRA fidelity analyzer for quantized base models.

Measures, in weight space, how faithfully a LoRA lands on an SDNQ-quantized
model. Every targeted module is rebuilt with the loader's own module class and
its delta taken from the production ``calc_updown``, so all adapter families
(LoRA, LoKR, LoHA, OFT, full, IA3, GLoRA, norm, plus DoRA and bias variants)
are measured as they would actually apply:

- factor path (plain additive LoRA riding the svd side-channel): storage is
  lossless; the reported figure is the delta realized through the result-dtype
  materialize, the same bf16 rounding an unquantized model applies.
  Eligibility is decided by the loader's own predicate.
- hosted path (non-factorable families on sub-8-bit formats): the seeded svd
  truncation at ``--host-rank``, realized the same way.
- requantize path (all other fallbacks): retention ``rho`` of the intended
  delta. On-grid rounding erases sub-step deltas down to a ``2/group_size``
  floor, so low-bit formats (<=6 bits) typically show rho ~= 0.02-0.03.
- unquantized modules: the LoRA applies exactly regardless.

Reported fidelity is per-module ``applied_rho`` (the measured figure for
whichever path the loader would take), summarized as a median and an
energy-weighted mean over the file's modules; ``requant_rho`` always carries
the if-merged figure.

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
    parser.add_argument('--dtype', default=None, help='simulate quantization of a bf16 repo at this sdnq dtype (e.g. uint4, int8); bf16 measures the unquantized reference')
    parser.add_argument('--group', type=int, default=0, help='sdnq group_size for simulation')
    parser.add_argument('--hadamard-group', type=int, default=256, help='sdnq hadamard group for simulation')
    parser.add_argument('--sample', type=int, default=40, help='max modules analyzed per lora (evenly sampled)')
    parser.add_argument('--full', action='store_true', help='analyze every matched module')
    parser.add_argument('--json', default=None, help='write full report to this json file')
    parser.add_argument('--host-rank', type=int, default=256, help='svd hosting cap for non-factorable modules on sub-8-bit formats, mirroring lora_sdnq_host_rank; 0 scores the requantize path instead')
    parser.add_argument('--calib', default=None, help='activation statistics file (data/sdnq-calib/*.safetensors): hosting truncation is then channel-weighted as with lora_sdnq_host_calib, and hosted rho is measured in the activation-weighted norm (the output-error proxy)')
    parser.add_argument('--fail-under', type=float, default=None, help='exit 2 when median applied fidelity of any lora is below this')
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

from modules.lora import native_adapter, network, network_lora, network_lokr, network_hada, network_oft, network_full, network_ia3, network_glora, network_norm, lora_sdnq # pylint: disable=wrong-import-position
from modules.lora.lora_load import NATIVE_DISPATCH # pylint: disable=wrong-import-position
from modules.sdnq.quantizer import sdnq_quantize_layer_weight # pylint: disable=wrong-import-position
from modules.sdnq.quant_utils import rotate_hadamard # pylint: disable=wrong-import-position


MODEL_ROOTS = [
    os.path.expanduser('~/database/models/huggingface'),
    os.path.expanduser('~/database/models/Diffusers'),
]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# every adapter family the native loader can build, with the module class that owns its
# apply-time math. deltas are taken from the production calc_updown so the tool cannot
# drift from the loader, and eligibility is decided by the production predicate itself.
FAMILY_SPECS = (
    ('lora',  network_lora.NetworkModuleLora,  native_adapter.LORA_SUFFIXES,  native_adapter.LORA_MARKERS),
    ('lokr',  network_lokr.NetworkModuleLokr,  native_adapter.LOKR_SUFFIXES,  native_adapter.LOKR_MARKERS),
    ('loha',  network_hada.NetworkModuleHada,  native_adapter.LOHA_SUFFIXES,  native_adapter.LOHA_MARKERS),
    ('oft',   network_oft.NetworkModuleOFT,    native_adapter.OFT_SUFFIXES,   native_adapter.OFT_MARKERS),
    ('full',  network_full.NetworkModuleFull,  native_adapter.FULL_SUFFIXES,  native_adapter.FULL_MARKERS),
    ('ia3',   network_ia3.NetworkModuleIa3,    native_adapter.IA3_SUFFIXES,   native_adapter.IA3_MARKERS),
    ('glora', network_glora.NetworkModuleGLora, native_adapter.GLORA_SUFFIXES, native_adapter.GLORA_MARKERS),
    ('norm',  network_norm.NetworkModuleNorm,  native_adapter.NORM_SUFFIXES,  native_adapter.NORM_MARKERS),
)


class StubOnDisk:
    def __init__(self, path):
        self.filename = path
        self.name = os.path.splitext(os.path.basename(path))[0]
        self.shorthash = ''
        self.sd_version = 'unknown'


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
    """Return {model_module_path: (family, weights)} across every adapter family, plus a census.

    Grouping mirrors the native loader: a family is only considered when its
    marker is present, and groups resolve to model paths through the arch's own
    resolver. Fused-split chunks are counted but not analyzed (their apply-time
    math is arch-owned).
    """
    with safe_open(lora_path, framework='pt', device='cpu') as f:
        state_dict = {k: f.get_tensor(k) for k in f.keys()}
    prefixes = getattr(arch_mod, 'KNOWN_PREFIXES', native_adapter.KNOWN_PREFIXES_DEFAULT)
    bare = getattr(arch_mod, 'BARE_DIFFUSERS_PREFIXES', ())
    resolve = getattr(arch_mod, 'resolve_targets', None) or (lambda prefix, base: [(base, None)])
    mapped, census, chunked = {}, {}, 0
    for fam, _cls, suffixes, markers in FAMILY_SPECS:
        if not native_adapter.has_marker(state_dict, markers):
            continue
        groups = native_adapter.group_by_suffixes(state_dict, suffixes, prefixes=prefixes, bare_diffusers_prefixes=bare)
        if fam == 'lora':
            groups = {k: w for k, w in groups.items() if 'lora_down.weight' in w and 'lora_up.weight' in w}
        else:
            groups = {k: w for k, w in groups.items() if native_adapter.has_marker({f'x.{s}': None for s in w}, markers)}
        if not groups:
            continue
        census[fam] = len(groups)
        for (prefix, base), w in groups.items():
            for path, chunk in native_adapter.resolve_group_targets(resolve, prefix, base):
                if chunk is not None:
                    chunked += 1
                    continue
                mapped.setdefault(path, []).append((fam, w)) # a module can carry several families; the loader applies each
    return mapped, census, chunked


def stamp_index(paths):
    """Map each module path to its stamped form, the way the loader matches.

    The loader compares ``network_prefix + path.replace('.', '_')`` against each
    module's stamped ``network_layer_name``, so kohya-style ``lora_unet_`` keys
    (whose base arrives already underscored) resolve fine there. Matching on the
    stamped form reproduces that and keeps dotted bases working unchanged.
    """
    return {p.replace('.', '_'): p for p in paths}


def make_stub(shape, dtype=torch.bfloat16):
    """Minimal sd_module standing in for a bf16 repo weight: the module classes key off its type and shape."""
    if len(shape) == 2:
        return torch.nn.Linear(shape[1], shape[0], bias=False, dtype=dtype, device='meta')
    return torch.nn.Conv2d(shape[1], shape[0], shape[2:], bias=False, dtype=dtype, device='meta')


def build_module(fam, path, w, net, sd_module):
    """Instantiate the family's production NetworkModule for one target."""
    cls = next(c for f, c, _s, _m in FAMILY_SPECS if f == fam)
    weights = network.NetworkWeights(network_key=path, sd_key=path, w=w, sd_module=sd_module)
    return cls(net, weights)


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
    from modules.sdnq.loader import load_sdnq_model
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
        self.handles = {} # reopening a multi-gb shard per module dominates runtime over many loras
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
        f = self.handles.get(shard)
        if f is None:
            f = safe_open(os.path.join(self.model_dir, shard), framework='pt', device='cpu')
            self.handles[shard] = f
        return f.get_tensor(key)


def analyze_module(W_dq, deq_params, mods, calib_rms=None, step_live=None):
    """Return fidelity metrics for one quantized module and the adapters targeting it.

    Deltas come from each module's production calc_updown and sum the way the
    loader stacks them, so every family (and dora / dense-bias / diff_b variant)
    is measured as applied. A module is factor-path eligible only when every
    contribution is a plain additive lora. With ``calib_rms``, hosting mirrors
    the calibrated production path and its rho is scored in the weighted norm.
    With ``step_live`` (the layer's own pre-add scale), the production routing
    rule applies: a delta fat against the grid whose truncation capture is low
    reports the requantize path, the way the loader would route it.
    """
    D = None
    for mod in mods:
        d = mod.calc_updown(W_dq)[0].to(device, torch.float32).reshape(W_dq.shape)
        D = d if D is None else D + d
    nD = D.norm()
    control = deq_params['weights_dtype'] == 'bf16' # unquantized reference: the delta just rounds into bf16
    factor_eligible = (not control) and all(lora_sdnq.get_module_factors(m, device, torch.bfloat16) is not None for m in mods)
    if float(nD) == 0.0: # an all-zero delta (some full-rank extractions carry empty .diff): retention is undefined, not erased
        return dict(rank=getattr(mods[0], 'dim', None), rms_delta=0.0, rms_weight=float(W_dq.pow(2).mean().sqrt()),
                    step_ratio=None, crossers=None, requant_rho=None, requant_resid=None,
                    factor_eligible=factor_eligible, hosted=False, applied_rho=None, delta_energy=0.0)
    step_ratio, crossers = None, None
    if control:
        W2 = (W_dq + D).to(torch.bfloat16).float()
    else:
        # mirror network_add_weights: it requantizes with the layer's own svd setting and rank,
        # and an svd checkpoint's dequantized weight is not on the plain integer grid
        use_svd = deq_params.get('use_svd', False)
        kw = dict(layer_class_name='Linear', torch_dtype=torch.bfloat16, group_size=deq_params['group_size'],
                  hadamard_group_size=deq_params['hadamard_group_size'], use_hadamard=deq_params['use_hadamard'],
                  weights_dtype=deq_params['weights_dtype'], use_svd=use_svd, svd_rank=deq_params.get('svd_rank', 32),
                  svd_steps=deq_params.get('svd_steps', 8), use_quantized_matmul=False, dequantize_fp32=False)
        deq2, data2 = sdnq_quantize_layer_weight(W_dq + D, **kw)
        W2 = deq2(data2['weight'], data2['scale'], zero_point=data2['zero_point'],
                  svd_up=data2['svd_up'], svd_down=data2['svd_down'], dtype=torch.float32, skip_compile=True)
        Dh = rotate_hadamard(D, group_size=deq_params['hadamard_group_size']) if deq_params['use_hadamard'] else D
        step = data2['scale'].float()
        Dg = Dh.unflatten(-1, (step.shape[1], -1)) if step.ndim == 3 else Dh
        step_ratio = float((Dg.abs() / step).mean())
        crossers = float((Dg.abs() > step / 2).float().mean())
    E = W2 - W_dq
    rho = float(E.flatten() @ D.flatten() / nD.square())
    resid = float((E - D).norm() / nD)
    hosted = False
    if factor_eligible:
        # the factor path stores the delta losslessly, but the dequantizer materializes
        # base + factors in the result dtype (bf16 here), so realized fidelity floors at
        # the same ULP rounding an unquantized bf16 model applies to a merged delta
        base16 = W_dq.to(torch.bfloat16).float()
        realized = (W_dq.to(torch.bfloat16) + D.to(torch.bfloat16)).float() - base16
        applied_rho = float(realized.flatten() @ D.flatten() / nD.square())
    else:
        applied_rho = rho
        if (not control) and cli_args.host_rank > 0:
            from modules.sdnq.common import dtype_dict
            if dtype_dict[deq_params['weights_dtype']]['num_bits'] < 8:
                # mirror lora_sdnq.apply_hosted: seeded svd truncation, realized through the bf16 materialize
                q = min(cli_args.host_rank, *D.shape)
                rms = None
                if calib_rms is not None and calib_rms.shape[-1] == D.shape[-1]:
                    rms = calib_rms.to(D.device, torch.float32).clamp(min=1e-8)
                Dw = D * rms if rms is not None else D
                with torch.random.fork_rng(devices=[D.device] if D.device.type == 'cuda' else []):
                    torch.manual_seed(0)
                    U, S, V = torch.svd_lowrank(Dw, q=min(q + 64, *D.shape), niter=8)
                energy = float(S[:q].square().sum() / Dw.square().sum().clamp(min=1e-30))
                routed = False
                if step_live is not None and not deq_params.get('use_svd', False):
                    sr = float(D.square().mean().sqrt() / step_live.float().mean())
                    routed = sr > lora_sdnq.REQUANT_RATIO and energy < lora_sdnq.REQUANT_ENERGY
                if not routed: # the loader routes fat, genuinely-truncated deltas back to requantize
                    Dk = (U[:, :q] * S[:q]) @ V[:, :q].t()
                    if rms is not None:
                        Dk = Dk / rms
                    base16 = W_dq.to(torch.bfloat16).float()
                    realized = (W_dq.to(torch.bfloat16) + Dk.to(torch.bfloat16)).float() - base16
                    if rms is not None: # weighted norm: the diagonal-covariance output-error proxy the calibrated truncation optimizes
                        Dr = D * rms
                        applied_rho = float((realized * rms).flatten() @ Dr.flatten() / Dr.square().sum())
                    else:
                        applied_rho = float(realized.flatten() @ D.flatten() / nD.square())
                    hosted = True
    return dict(rank=getattr(mods[0], 'dim', None), rms_delta=float(D.pow(2).mean().sqrt()), rms_weight=float(W_dq.pow(2).mean().sqrt()),
                step_ratio=step_ratio, crossers=crossers, requant_rho=rho, requant_resid=resid,
                factor_eligible=factor_eligible, hosted=hosted, applied_rho=applied_rho,
                delta_energy=float(nD.square()))


def main():
    args = cli_args
    model_dir = resolve_model_dir(args.model)
    arch_mod = resolve_arch(args.arch)
    with open(os.path.join(model_dir, 'config.json'), encoding='utf-8') as f:
        model_config = json.load(f)
    pre_quantized = model_config.get('quantization_config') is not None

    quant_layers, bf16_repo = {}, None
    quant_stamps, bf16_stamps = {}, {}
    if pre_quantized:
        rprint(f'model: "{model_dir}" pre-quantized={pre_quantized}')
        quant_layers = load_quantized_model(model_dir, arch=args.arch, class_name=model_config.get('_class_name'))
        quant_stamps = stamp_index(quant_layers)
    else:
        bf16_repo = Bf16Repo(model_dir)
        bf16_stamps = stamp_index(k[:-len('.weight')] for k in bf16_repo.weight_map if k.endswith('.weight'))
        if args.dtype is None:
            rprint('model is not quantized and no --dtype given: loras apply exactly, nothing to analyze')
            return 0
        rprint(f'model: "{model_dir}" simulating dtype={args.dtype} group={args.group} hadamard={args.hadamard_group}')

    calib_stats = {}
    if args.calib:
        with safe_open(os.path.expanduser(args.calib), framework='pt', device='cpu') as f:
            calib_stats = {k: f.get_tensor(k) for k in f.keys()}
        rprint(f'calib: "{args.calib}" layers={len(calib_stats)}')

    report = {'model': model_dir, 'pre_quantized': pre_quantized, 'loras': []}
    worst_effective = 1.0
    def write_report():
        if args.json:
            with open(args.json, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2)

    for lora_path in args.lora:
        lora_path = os.path.expanduser(lora_path)
        try:
            mapped, census, chunked = map_lora_modules(lora_path, arch_mod)
            net = network.Network(os.path.basename(lora_path), StubOnDisk(lora_path))
            rows, unquantized, unmatched, failed, non_matrix = [], [], [], [], []
            keys = sorted(mapped)
            if not args.full and len(keys) > args.sample:
                keys = keys[::max(1, len(keys) // args.sample)][:args.sample]
            for path in keys:
                entries = mapped[path]
                lname = path
                if pre_quantized:
                    if path not in quant_layers:
                        lname = quant_stamps.get(path.replace('.', '_'), '')
                    layer = quant_layers.get(lname)
                    if layer is None:
                        unmatched.append(path)
                        continue
                    deq = getattr(layer, 'sdnq_dequantizer', None)
                    if deq is None:
                        unquantized.append(path)
                        continue
                    if len(deq.original_shape) != 2:
                        non_matrix.append(path)
                        continue
                    W_dq = deq(layer.weight, layer.scale, zero_point=layer.zero_point, svd_up=layer.svd_up, svd_down=layer.svd_down,
                               skip_quantized_matmul=deq.use_quantized_matmul, dtype=torch.float32, skip_compile=True).to(device)
                    params = dict(weights_dtype=deq.weights_dtype, group_size=deq.group_size, hadamard_group_size=deq.hadamard_group_size,
                                  use_hadamard=deq.use_hadamard, use_svd=layer.svd_up is not None, svd_rank=deq.svd_rank, svd_steps=deq.svd_steps)
                    step_live = layer.scale.detach().to(device)
                    sd_module = layer
                else:
                    W = bf16_repo.get(f'{path}.weight')
                    if W is None:
                        W = bf16_repo.get(f'{bf16_stamps.get(path.replace(".", "_"), "")}.weight')
                    if W is None:
                        unmatched.append(path)
                        continue
                    if W.ndim != 2: # norm/scale targets (e.g. adaLN_modulation) are 1-D; the quantizer and the stub both expect a matrix
                        non_matrix.append(path)
                        continue
                    if args.dtype == 'bf16':
                        W_dq = W.to(device, torch.bfloat16).float()
                        params = dict(weights_dtype='bf16', group_size=0, hadamard_group_size=0, use_hadamard=False)
                        step_live = None
                    else:
                        deq0, data0 = sdnq_quantize_layer_weight(W.to(device, torch.float32), layer_class_name='Linear', weights_dtype=args.dtype,
                                                                 group_size=args.group, hadamard_group_size=args.hadamard_group, use_hadamard=args.hadamard_group > 0,
                                                                 use_svd=False, use_quantized_matmul=False, dequantize_fp32=False, torch_dtype=torch.bfloat16)
                        W_dq = deq0(data0['weight'], data0['scale'], zero_point=data0['zero_point'], svd_up=None, svd_down=None, dtype=torch.float32, skip_compile=True)
                        params = dict(weights_dtype=args.dtype, group_size=deq0.group_size, hadamard_group_size=deq0.hadamard_group_size, use_hadamard=deq0.use_hadamard)
                        step_live = data0['scale'].detach()
                    sd_module = make_stub(W.shape)
                try:
                    mods = [build_module(fam, path, w, net, sd_module) for fam, w in entries]
                    row = analyze_module(W_dq, params, mods, calib_rms=calib_stats.get(lname), step_live=step_live)
                except Exception as e: # a family the tool cannot rebuild must not read as a clean module
                    failed.append(f'{path}: {type(e).__name__}: {e}')
                    del W_dq
                    continue
                row.update(module=path, dtype=params['weights_dtype'], family='+'.join(f for f, _w in entries))
                rows.append(row)
                del W_dq # the caching allocator reuses these; emptying it per module costs more than it saves

            scored = [r for r in rows if r['applied_rho'] is not None] # zero-delta modules have no retention to report
            applied = sorted(r['applied_rho'] for r in scored)
            median_applied = applied[len(applied) // 2] if applied else None
            energy = sum(r['delta_energy'] for r in scored)
            weighted = (sum(r['applied_rho'] * r['delta_energy'] for r in scored) / energy) if energy > 0 else None
            n_exact = sum(1 for r in scored if r['factor_eligible'])
            fb = [r['requant_rho'] for r in scored if not r['factor_eligible']]
            fb_median = sorted(fb)[len(fb) // 2] if fb else None
            if median_applied is not None:
                worst_effective = min(worst_effective, median_applied)
            report['loras'].append({'file': lora_path, 'families': census, 'targets': len(mapped), 'unquantized': unquantized,
                                    'unmatched': unmatched, 'non_matrix': non_matrix, 'chunked': chunked, 'failed': failed,
                                    'exact_modules': n_exact, 'fallback_modules': len(fb), 'fallback_median_rho': fb_median,
                                    'median_applied_rho': median_applied, 'weighted_applied_rho': weighted, 'modules': rows})
            write_report() # rewrite per file so a crash keeps completed work
            rprint(f'\nlora: "{os.path.basename(lora_path)}" families={census or "none"} targets={len(mapped)} analyzed={len(rows)} scored={len(scored)} exact={n_exact} fallback={len(fb)} unquantized={len(unquantized)} unmatched={len(unmatched)} non_matrix={len(non_matrix)} chunked={chunked} failed={len(failed)}')
            if median_applied is None:
                rprint('  no analyzable modules: nothing measured')
            else:
                rprint(f'  applied fidelity: median={median_applied:.3f} energy-weighted={weighted:.3f}' + (f'  (fallback modules land at median rho={fb_median:.3f})' if fb_median is not None else ''))
            for f in failed[:3]:
                rprint(f'  [red]could not rebuild[/red]: {f}')
            if fb:
                worst = sorted((r for r in scored if not r['factor_eligible']), key=lambda r: r['requant_rho'])[:5]
                rprint('  lowest-retention modules:')
                for r in worst:
                    grid = f'step-ratio={r["step_ratio"]:.3f} crossers={r["crossers"]*100:5.1f}%' if r['step_ratio'] is not None else 'unquantized reference'
                    rprint(f'    {r["module"]:48s} fam={r["family"]:5s} dtype={r["dtype"]} {grid} rho={r["requant_rho"]:.3f}')
            del mapped, net
        except KeyboardInterrupt:
            raise
        except Exception as e: # one broken file must not cost the rest of the batch
            rprint(f'\n[red]lora failed[/red]: "{os.path.basename(lora_path)}" {type(e).__name__}: {e}')
            report['loras'].append({'file': lora_path, 'error': f'{type(e).__name__}: {e}'})
            write_report()
        if device.type == 'cuda':
            torch.cuda.empty_cache() # once per file, after its modules are done

    report['complete'] = True
    write_report()
    if args.json:
        rprint(f'\nreport: "{args.json}"')
    if args.fail_under is not None and worst_effective < args.fail_under:
        rprint(f'FAIL: effective fidelity {worst_effective:.3f} < {args.fail_under}')
        return 2
    return 0


if __name__ == '__main__':
    with torch.inference_mode():
        sys.exit(main())
