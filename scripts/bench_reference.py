import torch
import torch.nn.functional as F
import triton

from pm_env.reference_linear_ce import fused_linear_cross_entropy, IGNORE_INDEX


def bench(B: int, D: int, V: int) -> None:
    torch.manual_seed(0)
    hidden = torch.randn(B, D, device="cuda", dtype=torch.float16, requires_grad=True)
    weight = torch.randn(V, D, device="cuda", dtype=torch.float16, requires_grad=True)
    targets = torch.randint(0, V, (B,), device="cuda", dtype=torch.int64)

    def run_pytorch():
        hidden.grad = None
        weight.grad = None
        h = hidden.to(torch.float32)
        w = weight.to(torch.float32)
        logits = h @ w.T
        loss = F.cross_entropy(logits, targets, reduction="mean")
        loss.backward()

    def run_ours():
        hidden.grad = None
        weight.grad = None
        loss = fused_linear_cross_entropy(hidden, weight, targets)
        loss.backward()

    t_pt = triton.testing.do_bench(run_pytorch)
    t_ours = triton.testing.do_bench(run_ours)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    run_pytorch()
    mem_pt = torch.cuda.max_memory_allocated()

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    run_ours()
    mem_ours = torch.cuda.max_memory_allocated()

    logits_bytes = B * V * 2
    print(
        f"B={B:3d} D={D:4d} V={V:6d}  "
        f"time: pt={t_pt:6.3f}ms ours={t_ours:6.3f}ms speedup={t_pt/t_ours:5.2f}x  "
        f"mem: pt={mem_pt/1e6:6.2f}MB ours={mem_ours/1e6:6.2f}MB  "
        f"logits-matrix={logits_bytes/1e6:.2f}MB"
    )


def main() -> None:
    shapes = [
        (8, 256, 4096),
        (32, 512, 8192),
        (16, 512, 32768),
        (4, 1024, 131072),
    ]
    for B, D, V in shapes:
        bench(B, D, V)


if __name__ == "__main__":
    main()
