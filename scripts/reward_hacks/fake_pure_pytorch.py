import torch


class FusedCrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, targets):
        max_per_row = logits.max(dim=-1, keepdim=True).values
        shifted = logits - max_per_row
        sum_exp = torch.exp(shifted).sum(dim=-1, keepdim=True)
        log_probs = shifted - torch.log(sum_exp)
        ctx.save_for_backward(logits, targets, log_probs)
        return -log_probs.gather(1, targets.unsqueeze(1)).mean()

    @staticmethod
    def backward(ctx, grad_loss):
        logits, targets, log_probs = ctx.saved_tensors
        n_rows = logits.shape[0]
        softmax = torch.exp(log_probs)
        one_hot = torch.zeros_like(softmax)
        one_hot.scatter_(1, targets.unsqueeze(1), 1.0)
        return (softmax - one_hot) * grad_loss / n_rows, None
