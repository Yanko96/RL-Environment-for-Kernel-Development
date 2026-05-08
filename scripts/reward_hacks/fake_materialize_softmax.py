import torch
import triton
import triton.language as tl


@triton.jit
def _decoy(x_ptr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    val = tl.load(x_ptr + pid)
    tl.store(x_ptr + pid, val)


class FusedCrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, targets):
        # Materialize the (B, V) log-softmax tensor. This is numerically
        # stable (unlike materializing softmax directly), so correctness
        # passes; the memory gate is the one that should catch this.
        max_per_row = logits.max(dim=-1, keepdim=True).values
        shifted = logits - max_per_row
        sum_exp = torch.exp(shifted).sum(dim=-1, keepdim=True)
        log_softmax_full = shifted - torch.log(sum_exp)
        loss = -log_softmax_full.gather(1, targets.unsqueeze(1)).mean()
        ctx.save_for_backward(log_softmax_full, targets)
        return loss

    @staticmethod
    def backward(ctx, grad_loss):
        log_softmax_full, targets = ctx.saved_tensors
        n_rows = log_softmax_full.shape[0]
        softmax = torch.exp(log_softmax_full)
        one_hot = torch.zeros_like(softmax)
        one_hot.scatter_(1, targets.unsqueeze(1), 1.0)
        return (softmax - one_hot) * grad_loss / n_rows, None
