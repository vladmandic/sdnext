from typing import Dict, Any
import time
import warnings
import torch
import torch.nn.functional as F

warnings.filterwarnings("ignore", category=UserWarning)

warmup = 2
repeats = 50
dtypes = [torch.bfloat16] # , torch.float16]
backends = [
    "sdpa_math",
    "sdpa_mem_efficient",
    "sdpa_flash",
    "sdpa_all",
    "flex_attention",
    "xformers",
    "flash_attn",
    "sage_attn",
]
profiles = {
    "sdxl": {"l_q": 4096, "l_k": 4096, "h": 32, "d": 128},
    "flux.1": {"l_q": 16717, "l_k": 16717, "h": 24, "d": 128},
    "sd35": {"l_q": 16538, "l_k": 16538, "h": 24, "d": 128},
    "qwen-image": {"l_q": 16384, "l_k": 16384, "h": 24, "d": 128},
    "z-image": {"l_q": 4096, "l_k": 4096, "h": 32, "d": 120},
    "wan2.1": {"l_q": 16384, "l_k": 16384, "h": 40, "d": 128},
}

def get_stats(reset: bool = False):
    torch.cuda.synchronize()
    if reset:
        with torch.no_grad():
            torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    m = torch.cuda.max_memory_allocated()
    t = time.perf_counter()
    return m / (1024 ** 2), t

def print_gpu_info():
    if not torch.cuda.is_available():
        print("GPU: Not available")
        return

    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    total_mem = props.total_memory / (1024**3)
    free_mem, _ = torch.cuda.mem_get_info(device)
    free_mem = free_mem / (1024**3)
    major, minor = torch.cuda.get_device_capability(device)

    print(f"gpu: {torch.cuda.get_device_name(device)}")
    print(f"vram: total={total_mem:.2f}GB free={free_mem:.2f}GB")
    print(f"cuda: capability={major}.{minor} version={torch.version.cuda}")
    print(f"torch: {torch.__version__}")

def benchmark_attention(
    backend: str,
    dtype: torch.dtype,
    b: int = 1,
    l_q: int = 4096,
    l_k: int = 4096,
    h: int = 32,
    d: int = 128,
) -> Dict[str, Any]:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Initialize tensors
    q = torch.randn(b, h, l_q, d, device=device, dtype=torch.float16 if dtype.is_floating_point and dtype.itemsize == 1 else dtype, requires_grad=False).to(dtype)
    k = torch.randn(b, h, l_k, d, device=device, dtype=torch.float16 if dtype.is_floating_point and dtype.itemsize == 1 else dtype, requires_grad=False).to(dtype)
    v = torch.randn(b, h, l_k, d, device=device, dtype=torch.float16 if dtype.is_floating_point and dtype.itemsize == 1 else dtype, requires_grad=False).to(dtype)

    results = {
        "backend": backend,
        "dtype": str(dtype),
        "status": "pass",
        "latency_ms": 0.0,
        "memory_mb": 0.0,
        "version": "N/A",
        "error": ""
    }
    try:
        if backend.startswith("sdpa_"):
            from torch.nn.attention import sdpa_kernel, SDPBackend
            sdp_type = backend[len("sdpa_"):]
            # Map friendly names to new SDPA backends
            backend_map = {
                "math": [SDPBackend.MATH],
                "flash": [SDPBackend.FLASH_ATTENTION],
                "mem_efficient": [SDPBackend.EFFICIENT_ATTENTION],
                "all": [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]
            }
            if sdp_type not in backend_map:
                raise ValueError(f"Unknown SDPA type: {sdp_type}")

            results["version"] = torch.__version__

            with sdpa_kernel(backend_map[sdp_type]):
                # Warmup
                for _ in range(warmup):
                    _ = F.scaled_dot_product_attention(q, k, v)

                start_mem, start_time = get_stats(True)

                for _ in range(repeats):
                    _ = F.scaled_dot_product_attention(q, k, v)

                end_mem, end_time = get_stats()

                results["latency_ms"] = (end_time - start_time) / repeats * 1000
                results["memory_mb"] = end_mem - start_mem

        elif backend == "flash_attn":
            from flash_attn import flash_attn_func, __version__ as fa_version
            results["version"] = fa_version
            # Flash attention usually expects (B, L, H, D)
            q_fa = q.transpose(1, 2)
            k_fa = k.transpose(1, 2)
            v_fa = v.transpose(1, 2)

            for _ in range(warmup):
                _ = flash_attn_func(q_fa, k_fa, v_fa)

            start_mem, start_time = get_stats(True)

            for _ in range(repeats):
                _ = flash_attn_func(q_fa, k_fa, v_fa)

            end_mem, end_time = get_stats()

            results["latency_ms"] = (end_time - start_time) / repeats * 1000
            results["memory_mb"] = end_mem - start_mem

        elif backend == "xformers":
            from xformers.ops import memory_efficient_attention
            from xformers import __version__ as xf_version
            results["version"] = xf_version
            # xformers also usually prefers (B, L, H, D)
            q_xf = q.transpose(1, 2)
            k_xf = k.transpose(1, 2)
            v_xf = v.transpose(1, 2)

            for _ in range(warmup):
                _ = memory_efficient_attention(q_xf, k_xf, v_xf)

            start_mem, start_time = get_stats(True)

            for _ in range(repeats):
                _ = memory_efficient_attention(q_xf, k_xf, v_xf)

            end_mem, end_time = get_stats()

            results["latency_ms"] = (end_time - start_time) / repeats * 1000
            results["memory_mb"] = end_mem - start_mem

        elif backend == "sage_attn":
            from sageattention import sageattn
            import sageattention
            # Attempt to get version from package metadata or a common attribute
            try:
                import importlib.metadata
                results["version"] = importlib.metadata.version("sageattention")
            except Exception:
                results["version"] = getattr(sageattention, "__version__", "N/A")

            # SageAttention expects (B, H, L, D) logic
            for _ in range(warmup):
                _ = sageattn(q, k, v)

            start_mem, start_time = get_stats(True)

            for _ in range(repeats):
                _ = sageattn(q, k, v)

            end_mem, end_time = get_stats()

            results["latency_ms"] = (end_time - start_time) / repeats * 1000
            results["memory_mb"] = end_mem - start_mem

        elif backend == "flex_attention":
            from torch.nn.attention.flex_attention import flex_attention
            results["version"] = torch.__version__

            # flex_attention requires torch.compile for performance
            flex_attention_compiled = torch.compile(flex_attention, dynamic=False)

            # Warmup (important to trigger compilation)
            for _ in range(warmup):
                _ = flex_attention_compiled(q, k, v)

            start_mem, start_time = get_stats(True)

            for _ in range(repeats):
                _ = flex_attention_compiled(q, k, v)

            end_mem, end_time = get_stats()

            results["latency_ms"] = (end_time - start_time) / repeats * 1000
            results["memory_mb"] = end_mem - start_mem
    except Exception as e:
        results["status"] = "fail"
        results["error"] = str(e)[:49]
        print(e)

    return results

def main():

    all_results = []

    print_gpu_info()
    print(f'config: warmup={warmup} repeats={repeats} dtypes={dtypes}')
    for name, config in profiles.items():
        print(f"profile: {name} (L_q={config['l_q']}, L_k={config['l_k']}, H={config['h']}, D={config['d']})")
        for dtype in dtypes:
            print(f"  dtype: {dtype}")
            print(f"    {'backend':<20} | {'version':<12} | {'status':<8} | {'latency':<10} | {'memory':<12} | ")
            for backend in backends:
                res = benchmark_attention(
                    backend,
                    dtype,
                    l_q=config["l_q"],
                    l_k=config["l_k"],
                    h=config["h"],
                    d=config["d"],
                )
                all_results.append(res)

                latency = f"{res['latency_ms']:.4f} ms"
                memory = f"{res['memory_mb']:.2f} MB"

                print(f"    {res['backend']:<20} | {res['version']:<12} | {res['status']:<8} | {latency:<10} | {memory:<12} | {res['error']}")

if __name__ == "__main__":
    main()
