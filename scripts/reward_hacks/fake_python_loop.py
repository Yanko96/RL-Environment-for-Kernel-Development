import torch
import triton
import triton.language as tl


@triton.jit
def _row_fwd(
    logits_ptr, target_idx, loss_ptr, lse_ptr, n_cols, BLOCK_SIZE: tl.constexpr
):
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols
    logits = tl.load(logits_ptr + cols, mask=mask, other=-float("inf"))
    row_max = tl.max(logits, axis=0)
    sum_exp = tl.sum(tl.exp(logits - row_max), axis=0)
    lse = row_max + tl.log(sum_exp)
    target_logit = tl.load(logits_ptr + target_idx)
    tl.store(loss_ptr, lse - target_logit)
    tl.store(lse_ptr, lse)


@triton.jit
def _row_bwd(
    grad_loss_scalar, logits_ptr, target_idx, lse_val,
    grad_ptr, n_rows, n_cols, BLOCK_SIZE: tl.constexpr,
):
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols
    logits = tl.load(logits_ptr + cols, mask=mask, other=-float("inf"))
    softmax = tl.exp(logits - lse_val)
    one_hot = (cols == target_idx).to(tl.float32)
    grad = (softmax - one_hot) * (grad_loss_scalar / n_rows)
    tl.store(grad_ptr + cols, grad, mask=mask)


class FusedCrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, targets):
        n_rows, n_cols = logits.shape
        BLOCK_SIZE = triton.next_power_of_2(n_cols)
        losses = torch.empty(n_rows, device=logits.device, dtype=torch.float32)
        lses = torch.empty(n_rows, device=logits.device, dtype=torch.float32)
        # Slow path: launch one Triton kernel per row from a Python loop.
        for i in range(n_rows):
            _row_fwd[(1,)](
                logits[i], int(targets[i].item()),
                losses[i:i+1], lses[i:i+1],
                n_cols=n_cols, BLOCK_SIZE=BLOCK_SIZE,
            )
        ctx.save_for_backward(logits, targets, lses)
        return losses.mean()

    @staticmethod
    def backward(ctx, grad_loss):
        logits, targets, lses = ctx.saved_tensors
        n_rows, n_cols = logits.shape
        BLOCK_SIZE = triton.next_power_of_2(n_cols)
        grad_logits = torch.empty_like(logits)
        for i in range(n_rows):
            _row_bwd[(1,)](
                grad_loss.item(), logits[i], int(targets[i].item()),
                lses[i].item(), grad_logits[i],
                n_rows=n_rows, n_cols=n_cols, BLOCK_SIZE=BLOCK_SIZE,
            )
        return grad_logits, None
