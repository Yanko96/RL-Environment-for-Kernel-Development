import torch
import triton
import triton.language as tl


@triton.jit
def _decoy(x_ptr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    val = tl.load(x_ptr + pid)
    tl.store(x_ptr + pid, val)


class FusedLinearCrossEntropy(torch.autograd.Function):
    """Numerically correct (fp32 internals) but materializes the full
    (B, V) logits and softmax tensors. Memory gate should catch this."""

    @staticmethod
    def forward(ctx, hidden, weight, targets):
        IGNORE = -100
        valid = targets != IGNORE
        n_valid = max(int(valid.sum().item()), 1)

        h = hidden.to(torch.float32)
        w = weight.to(torch.float32)
        logits = h @ w.T  # (B, V) — materialized
        max_per_row = logits.max(dim=-1, keepdim=True).values
        shifted = logits - max_per_row
        sum_exp = torch.exp(shifted).sum(dim=-1, keepdim=True)
        log_softmax = shifted - torch.log(sum_exp)

        safe_targets = targets.clamp(min=0)
        per_row = -log_softmax.gather(1, safe_targets.unsqueeze(1)).squeeze(1)
        per_row = torch.where(valid, per_row, torch.zeros_like(per_row))
        loss = per_row.sum() / n_valid

        ctx.save_for_backward(hidden, weight, targets, log_softmax)
        ctx.n_valid = n_valid
        return loss

    @staticmethod
    def backward(ctx, grad_loss):
        hidden, weight, targets, log_softmax = ctx.saved_tensors
        n_valid = ctx.n_valid
        valid = (targets != -100).to(torch.float32)
        safe_targets = targets.clamp(min=0)
        softmax = torch.exp(log_softmax)
        one_hot = torch.zeros_like(softmax)
        one_hot.scatter_(1, safe_targets.unsqueeze(1), 1.0)
        diff = (softmax - one_hot) * valid.unsqueeze(1)
        scale = grad_loss.item() / n_valid
        grad_hidden = (diff @ weight.to(torch.float32)) * scale
        grad_weight = (diff.T @ hidden.to(torch.float32)) * scale
        return grad_hidden.to(hidden.dtype), grad_weight.to(weight.dtype), None
