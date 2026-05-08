import torch
import torch.nn.functional as F

from pm_env.reference_fused_loss import fused_cross_entropy


def check(B: int, V: int, seed: int = 0) -> None:
    torch.manual_seed(seed)

    logits_ref = torch.randn(B, V, device="cuda", dtype=torch.float32, requires_grad=True)
    targets = torch.randint(0, V, (B,), device="cuda", dtype=torch.int64)

    loss_ref = F.cross_entropy(logits_ref, targets, reduction="mean")
    loss_ref.backward()
    grad_ref = logits_ref.grad.clone()

    logits_ours = logits_ref.detach().clone().requires_grad_(True)
    loss_ours = fused_cross_entropy(logits_ours, targets)
    loss_ours.backward()
    grad_ours = logits_ours.grad

    fwd_diff = (loss_ref - loss_ours).abs().item()
    bwd_diff = (grad_ref - grad_ours).abs().max().item()

    print(f"B={B:4d} V={V:5d}  fwd_diff={fwd_diff:.2e}  bwd_diff={bwd_diff:.2e}")
    assert fwd_diff < 1e-4, f"forward diff too large: {fwd_diff}"
    assert bwd_diff < 1e-4, f"backward diff too large: {bwd_diff}"


def main() -> None:
    shapes = [
        (8, 1024),
        (32, 2048),
        (16, 4096),
        (64, 8192),
        (4, 1024),
        (128, 1024),
        (1, 1024),
        (32, 1031),  # non-power-of-2 vocab
    ]
    for B, V in shapes:
        check(B, V)
    print("\nall correctness checks passed")


if __name__ == "__main__":
    main()
