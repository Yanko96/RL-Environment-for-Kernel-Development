"""Verify the fused linear-CE reference matches a fp64 PyTorch ground
truth across multiple dtypes, distributions, and ignore_index patterns.
"""

import torch
import torch.nn.functional as F

from pm_env.reference_linear_ce import fused_linear_cross_entropy, IGNORE_INDEX


def check(
    B: int, D: int, V: int,
    dtype: torch.dtype,
    targets_pattern: str,
    seed: int = 0,
) -> None:
    torch.manual_seed(seed)
    hidden_data = (torch.randn(B, D, device="cuda", dtype=torch.float32) * 0.5).to(dtype)
    weight_data = (torch.randn(V, D, device="cuda", dtype=torch.float32) * 0.5).to(dtype)

    if targets_pattern == "all_valid":
        targets = torch.randint(0, V, (B,), device="cuda", dtype=torch.int64)
    elif targets_pattern == "some_ignored":
        targets = torch.randint(0, V, (B,), device="cuda", dtype=torch.int64)
        # ignore roughly the first 1/3 of rows
        n_ignored = max(1, B // 3)
        targets[:n_ignored] = IGNORE_INDEX
    else:
        raise ValueError(targets_pattern)

    # Ground truth: fp64 reference materializing logits.
    hidden_ref = hidden_data.to(torch.float64).clone().requires_grad_(True)
    weight_ref = weight_data.to(torch.float64).clone().requires_grad_(True)
    logits_ref = hidden_ref @ weight_ref.T
    loss_ref = F.cross_entropy(
        logits_ref, targets, reduction="mean", ignore_index=IGNORE_INDEX
    )
    loss_ref.backward()

    hidden_ours = hidden_data.detach().clone().requires_grad_(True)
    weight_ours = weight_data.detach().clone().requires_grad_(True)
    loss_ours = fused_linear_cross_entropy(hidden_ours, weight_ours, targets)
    loss_ours.backward()

    fwd_diff = (loss_ref.float() - loss_ours.float()).abs().item()
    grad_h_diff = (hidden_ref.grad.float() - hidden_ours.grad.float()).abs().max().item()
    grad_w_diff = (weight_ref.grad.float() - weight_ours.grad.float()).abs().max().item()

    print(
        f"B={B:3d} D={D:4d} V={V:6d} dt={str(dtype).replace('torch.', ''):8s} "
        f"tgt={targets_pattern:13s} "
        f"fwd={fwd_diff:.2e} gh={grad_h_diff:.2e} gw={grad_w_diff:.2e}"
    )

    # Tolerance: bf16 has ~7 mantissa bits → ~1e-2 absolute precision for
    # values in O(1); fp16 has 10 mantissa bits → ~1e-3. We measure at fp32
    # but the agent's output is rounded to bf16/fp16, so this is the floor.
    fwd_tol = 5e-2 if dtype == torch.bfloat16 else 5e-3
    grad_tol = 5e-2 if dtype == torch.bfloat16 else 5e-3
    assert fwd_diff < fwd_tol, f"fwd diff too large: {fwd_diff}"
    assert grad_h_diff < grad_tol, f"grad_hidden diff too large: {grad_h_diff}"
    assert grad_w_diff < grad_tol, f"grad_weight diff too large: {grad_w_diff}"


def main() -> None:
    shapes = [
        (8, 256, 4096),
        (32, 512, 8192),
        (16, 256, 16384),
        (4, 128, 4096),
        (1, 128, 4096),
    ]
    for B, D, V in shapes:
        for dtype in (torch.float16, torch.bfloat16):
            for tgt in ("all_valid", "some_ignored"):
                # B=1 + some_ignored makes the entire batch ignored
                # (n_valid=0). PyTorch returns NaN, we return 0; the
                # reduction is undefined either way. Skip the degenerate.
                if B == 1 and tgt == "some_ignored":
                    continue
                check(B, D, V, dtype, tgt)
    print("\nall correctness checks passed")


if __name__ == "__main__":
    main()
