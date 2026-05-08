import torch
import triton
import triton.language as tl


@triton.jit
def _fwd_kernel(
    logits_ptr,
    targets_ptr,
    loss_ptr,
    lse_ptr,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    row_logits_ptr = logits_ptr + row * n_cols
    target = tl.load(targets_ptr + row).to(tl.int32)

    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols

    logits = tl.load(row_logits_ptr + cols, mask=mask, other=-float("inf"))
    row_max = tl.max(logits, axis=0)
    sum_exp = tl.sum(tl.exp(logits - row_max), axis=0)
    lse = row_max + tl.log(sum_exp)

    target_logit = tl.load(row_logits_ptr + target)
    loss = lse - target_logit

    tl.store(loss_ptr + row, loss)
    tl.store(lse_ptr + row, lse)


@triton.jit
def _bwd_kernel(
    grad_loss_scalar,
    logits_ptr,
    targets_ptr,
    lse_ptr,
    grad_logits_ptr,
    n_rows,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    row_logits_ptr = logits_ptr + row * n_cols
    row_grad_ptr = grad_logits_ptr + row * n_cols
    target = tl.load(targets_ptr + row).to(tl.int32)
    lse = tl.load(lse_ptr + row)

    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols

    # other=-inf so masked positions become exp(-inf) = 0, contributing nothing.
    logits = tl.load(row_logits_ptr + cols, mask=mask, other=-float("inf"))
    softmax = tl.exp(logits - lse)
    one_hot = (cols == target).to(tl.float32)
    grad = (softmax - one_hot) * (grad_loss_scalar / n_rows)

    tl.store(row_grad_ptr + cols, grad, mask=mask)


class FusedCrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        n_rows, n_cols = logits.shape
        BLOCK_SIZE = triton.next_power_of_2(n_cols)
        num_warps = 8 if BLOCK_SIZE >= 2048 else 4

        loss_per_row = torch.empty(n_rows, device=logits.device, dtype=torch.float32)
        lse_per_row = torch.empty(n_rows, device=logits.device, dtype=torch.float32)

        _fwd_kernel[(n_rows,)](
            logits, targets,
            loss_per_row, lse_per_row,
            n_cols=n_cols,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
        )

        ctx.save_for_backward(logits, targets, lse_per_row)
        return loss_per_row.mean()

    @staticmethod
    def backward(ctx, grad_loss: torch.Tensor):
        logits, targets, lse_per_row = ctx.saved_tensors
        n_rows, n_cols = logits.shape
        BLOCK_SIZE = triton.next_power_of_2(n_cols)
        num_warps = 8 if BLOCK_SIZE >= 2048 else 4

        grad_logits = torch.empty_like(logits)

        _bwd_kernel[(n_rows,)](
            grad_loss.item(),
            logits, targets, lse_per_row,
            grad_logits,
            n_rows=n_rows,
            n_cols=n_cols,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
        )
        return grad_logits, None


def fused_cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return FusedCrossEntropy.apply(logits, targets)
