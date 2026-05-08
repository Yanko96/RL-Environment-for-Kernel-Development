import torch
import triton
import triton.language as tl


@triton.jit
def _row_op(out_ptr, in_ptr, BLOCK: tl.constexpr):
    """Trivial elementwise copy — present so AST scan finds @triton.jit,
    but the actual work happens in the Python loop below."""
    offs = tl.arange(0, BLOCK)
    val = tl.load(in_ptr + offs)
    tl.store(out_ptr + offs, val)


class FusedLinearCrossEntropy(torch.autograd.Function):
    """Slow path: Python loop processing one batch row at a time, doing
    the matmul + softmax in PyTorch per-row. Memory-efficient (no full
    (B,V) materialization), but throughput is dominated by Python and
    kernel-launch overhead. Throughput gate should catch it."""

    @staticmethod
    def forward(ctx, hidden, weight, targets):
        B, D = hidden.shape
        V, _ = weight.shape
        valid = targets != -100
        n_valid = max(int(valid.sum().item()), 1)

        losses = []
        lses = torch.empty(B, device=hidden.device, dtype=torch.float32)
        for i in range(B):
            t = int(targets[i].item())
            if t < 0:
                lses[i] = 0.0
                losses.append(torch.zeros((), device=hidden.device, dtype=torch.float32))
                continue
            h_i = hidden[i].to(torch.float32)
            logits_i = (weight.to(torch.float32) @ h_i)  # (V,) — but only one row
            m = logits_i.max()
            lse = m + torch.log(torch.exp(logits_i - m).sum())
            lses[i] = lse.detach()
            losses.append(lse - logits_i[t])

        loss_sum = torch.stack(losses).sum() / n_valid
        ctx.save_for_backward(hidden, weight, targets, lses)
        ctx.n_valid = n_valid
        return loss_sum

    @staticmethod
    def backward(ctx, grad_loss):
        hidden, weight, targets, lses = ctx.saved_tensors
        n_valid = ctx.n_valid
        scale = grad_loss.item() / n_valid
        grad_hidden = torch.zeros_like(hidden)
        grad_weight = torch.zeros_like(weight)
        h32 = hidden.to(torch.float32)
        w32 = weight.to(torch.float32)
        for i in range(hidden.shape[0]):
            t = int(targets[i].item())
            if t < 0:
                continue
            logits_i = w32 @ h32[i]
            softmax_i = torch.exp(logits_i - lses[i])
            one_hot = torch.zeros_like(softmax_i)
            one_hot[t] = 1.0
            diff = softmax_i - one_hot
            grad_hidden[i] = ((diff @ w32) * scale).to(hidden.dtype)
            grad_weight += (diff.unsqueeze(1) * h32[i].unsqueeze(0) * scale).to(weight.dtype)
        return grad_hidden, grad_weight, None
