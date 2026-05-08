"""Reference implementation of fused linear cross-entropy loss.

Computes cross-entropy on `logits = hidden @ weight.T` mean-reduced over
the batch, **without ever materializing the (B, V) logits or softmax
tensors**. This is the "Cut Cross Entropy" pattern (Wijmans et al., 2024)
used for memory-efficient LLM-head loss in modern training pipelines.

Supports:
- fp16 and bf16 input (with fp32 accumulators)
- Variable-length sequences via `ignore_index = -100` (rows with negative
  targets contribute 0 to loss and gradients; mean is over valid rows only,
  matching `torch.nn.functional.cross_entropy(reduction='mean')` semantics)

Forward returns a scalar fp32 loss. Backward returns gradients for both
`hidden` (B, D) and `weight` (V, D), matching the input dtype.
"""

import torch
import triton
import triton.language as tl


IGNORE_INDEX = -100


@triton.jit
def _fwd_kernel(
    hidden_ptr, weight_ptr, targets_ptr,
    losses_ptr, lse_ptr,
    B, D, V,
    BLOCK_D: tl.constexpr, BLOCK_V: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= B:
        return

    target_raw = tl.load(targets_ptr + row).to(tl.int32)
    is_valid = target_raw >= 0
    target = tl.where(is_valid, target_raw, 0)

    d_offs = tl.arange(0, BLOCK_D)
    d_mask = d_offs < D

    hidden_row = tl.load(hidden_ptr + row * D + d_offs, mask=d_mask, other=0.0).to(tl.float32)

    target_w = tl.load(weight_ptr + target * D + d_offs, mask=d_mask, other=0.0).to(tl.float32)
    target_logit = tl.sum(hidden_row * target_w)

    m = -float("inf")
    s = 0.0
    for v_start in range(0, V, BLOCK_V):
        v_offs = v_start + tl.arange(0, BLOCK_V)
        v_mask = v_offs < V

        w_tile = tl.load(
            weight_ptr + v_offs[:, None] * D + d_offs[None, :],
            mask=v_mask[:, None] & d_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        logits_tile = tl.sum(w_tile * hidden_row[None, :], axis=1)
        logits_tile = tl.where(v_mask, logits_tile, -float("inf"))

        m_new = tl.maximum(m, tl.max(logits_tile, axis=0))
        s = s * tl.exp(m - m_new) + tl.sum(tl.exp(logits_tile - m_new))
        m = m_new

    lse = m + tl.log(s)
    loss = lse - target_logit

    final_loss = tl.where(is_valid, loss, 0.0)
    final_lse = tl.where(is_valid, lse, 0.0)
    tl.store(losses_ptr + row, final_loss)
    tl.store(lse_ptr + row, final_lse)


@triton.jit
def _bwd_hidden_kernel(
    gl_over_eff,
    hidden_ptr, weight_ptr, targets_ptr, lse_ptr,
    grad_hidden_ptr,
    B, D, V,
    BLOCK_D: tl.constexpr, BLOCK_V: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= B:
        return

    target_raw = tl.load(targets_ptr + row).to(tl.int32)
    is_valid = target_raw >= 0
    target = tl.where(is_valid, target_raw, 0)
    lse = tl.load(lse_ptr + row)

    d_offs = tl.arange(0, BLOCK_D)
    d_mask = d_offs < D

    hidden_row = tl.load(hidden_ptr + row * D + d_offs, mask=d_mask, other=0.0).to(tl.float32)
    grad_h = tl.zeros((BLOCK_D,), dtype=tl.float32)

    for v_start in range(0, V, BLOCK_V):
        v_offs = v_start + tl.arange(0, BLOCK_V)
        v_mask = v_offs < V

        w_tile = tl.load(
            weight_ptr + v_offs[:, None] * D + d_offs[None, :],
            mask=v_mask[:, None] & d_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        logits_tile = tl.sum(w_tile * hidden_row[None, :], axis=1)
        logits_tile = tl.where(v_mask, logits_tile, -float("inf"))

        softmax_tile = tl.exp(logits_tile - lse)
        grad_h += tl.sum(softmax_tile[:, None] * w_tile, axis=0)

    target_w = tl.load(weight_ptr + target * D + d_offs, mask=d_mask, other=0.0).to(tl.float32)
    grad_h = (grad_h - target_w) * gl_over_eff

    # Zero out for ignored rows.
    valid_f = is_valid.to(tl.float32)
    grad_h = grad_h * valid_f

    tl.store(grad_hidden_ptr + row * D + d_offs, grad_h, mask=d_mask)


@triton.jit
def _bwd_weight_kernel(
    gl_over_eff,
    hidden_ptr, weight_ptr, targets_ptr, lse_ptr,
    grad_weight_ptr,
    B, D, V,
    BLOCK_D: tl.constexpr, BLOCK_V: tl.constexpr,
):
    v_block_idx = tl.program_id(0)
    v_offs = v_block_idx * BLOCK_V + tl.arange(0, BLOCK_V)
    v_mask = v_offs < V

    d_offs = tl.arange(0, BLOCK_D)
    d_mask = d_offs < D

    w_tile = tl.load(
        weight_ptr + v_offs[:, None] * D + d_offs[None, :],
        mask=v_mask[:, None] & d_mask[None, :],
        other=0.0,
    ).to(tl.float32)

    grad_w_tile = tl.zeros((BLOCK_V, BLOCK_D), dtype=tl.float32)

    for b in range(B):
        target_raw = tl.load(targets_ptr + b).to(tl.int32)
        is_valid = (target_raw >= 0).to(tl.float32)
        target = tl.where(target_raw >= 0, target_raw, 0)

        hidden_b = tl.load(hidden_ptr + b * D + d_offs, mask=d_mask, other=0.0).to(tl.float32)
        lse_b = tl.load(lse_ptr + b)

        logits_tile = tl.sum(w_tile * hidden_b[None, :], axis=1)
        logits_tile = tl.where(v_mask, logits_tile, -float("inf"))
        softmax_tile = tl.exp(logits_tile - lse_b)

        one_hot = (v_offs == target).to(tl.float32)
        diff = softmax_tile - one_hot

        # Mask contribution from ignored rows.
        grad_w_tile += is_valid * diff[:, None] * hidden_b[None, :]

    grad_w_tile = grad_w_tile * gl_over_eff

    tl.store(
        grad_weight_ptr + v_offs[:, None] * D + d_offs[None, :],
        grad_w_tile,
        mask=v_mask[:, None] & d_mask[None, :],
    )


def _pick_block_v(D: int) -> int:
    block_d = triton.next_power_of_2(D)
    target_tile = 16 * 1024
    block_v = max(8, min(64, target_tile // block_d))
    p = 1
    while p * 2 <= block_v:
        p *= 2
    return p


class FusedLinearCrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        hidden: torch.Tensor,
        weight: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        assert hidden.dtype == weight.dtype
        assert hidden.dtype in (torch.float16, torch.bfloat16)
        B, D = hidden.shape
        V, D2 = weight.shape
        assert D == D2

        BLOCK_D = triton.next_power_of_2(D)
        BLOCK_V = _pick_block_v(D)

        valid_mask = targets >= 0
        effective_batch = max(int(valid_mask.sum().item()), 1)

        losses = torch.empty(B, device=hidden.device, dtype=torch.float32)
        lse = torch.empty(B, device=hidden.device, dtype=torch.float32)

        _fwd_kernel[(B,)](
            hidden, weight, targets,
            losses, lse,
            B=B, D=D, V=V,
            BLOCK_D=BLOCK_D, BLOCK_V=BLOCK_V,
            num_warps=4,
        )

        ctx.save_for_backward(hidden, weight, targets, lse)
        ctx.BLOCK_D = BLOCK_D
        ctx.BLOCK_V = BLOCK_V
        ctx.effective_batch = effective_batch
        return losses.sum() / effective_batch

    @staticmethod
    def backward(ctx, grad_loss: torch.Tensor):
        hidden, weight, targets, lse = ctx.saved_tensors
        B, D = hidden.shape
        V, _ = weight.shape
        BLOCK_D = ctx.BLOCK_D
        BLOCK_V = ctx.BLOCK_V

        grad_hidden = torch.empty_like(hidden)
        grad_weight = torch.empty_like(weight)

        gl_over_eff = grad_loss.item() / ctx.effective_batch

        _bwd_hidden_kernel[(B,)](
            gl_over_eff,
            hidden, weight, targets, lse,
            grad_hidden,
            B=B, D=D, V=V,
            BLOCK_D=BLOCK_D, BLOCK_V=BLOCK_V,
            num_warps=4,
        )

        n_v_blocks = triton.cdiv(V, BLOCK_V)
        _bwd_weight_kernel[(n_v_blocks,)](
            gl_over_eff,
            hidden, weight, targets, lse,
            grad_weight,
            B=B, D=D, V=V,
            BLOCK_D=BLOCK_D, BLOCK_V=BLOCK_V,
            num_warps=4,
        )

        return grad_hidden, grad_weight, None


def fused_linear_cross_entropy(
    hidden: torch.Tensor, weight: torch.Tensor, targets: torch.Tensor
) -> torch.Tensor:
    return FusedLinearCrossEntropy.apply(hidden, weight, targets)
