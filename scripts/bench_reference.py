import torch
import torch.nn.functional as F
import triton

from pm_env.reference_fused_loss import fused_cross_entropy


def bench(B: int, V: int) -> None:
    torch.manual_seed(0)
    logits = torch.randn(B, V, device="cuda", dtype=torch.float32, requires_grad=True)
    targets = torch.randint(0, V, (B,), device="cuda", dtype=torch.int64)

    def run_ref():
        logits.grad = None
        loss = F.cross_entropy(logits, targets, reduction="mean")
        loss.backward()

    def run_ours():
        logits.grad = None
        loss = fused_cross_entropy(logits, targets)
        loss.backward()

    t_ref = triton.testing.do_bench(run_ref)
    t_ours = triton.testing.do_bench(run_ours)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    run_ref()
    mem_ref = torch.cuda.max_memory_allocated()

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    run_ours()
    mem_ours = torch.cuda.max_memory_allocated()

    logits_bytes = B * V * 4
    print(
        f"B={B:4d} V={V:5d}  "
        f"time: ref={t_ref:6.3f}ms ours={t_ours:6.3f}ms speedup={t_ref/t_ours:5.2f}x  "
        f"mem: ref={mem_ref/1e6:6.2f}MB ours={mem_ours/1e6:6.2f}MB  "
        f"ours/(2*logits)={mem_ours/(2*logits_bytes):4.2f}"
    )


def main() -> None:
    shapes = [
        (32, 1024),
        (32, 2048),
        (32, 4096),
        (32, 8192),
        (128, 4096),
        (256, 4096),
        (16, 8192),
    ]
    for B, V in shapes:
        bench(B, V)


if __name__ == "__main__":
    main()
