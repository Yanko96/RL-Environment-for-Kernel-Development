import sys
from textwrap import dedent

from pm_env.get_data_dir import get_env_data_dir
from pm_env.judges.executable_judge import ExecutableJudge
from pm_env.schemas.evaluation_run_config import EvaluationRunConfig
from pm_env.task import Step, Task


def get_tasks(config: EvaluationRunConfig) -> list[Task]:
    solution_path = f"{get_env_data_dir()}/linear_ce.py"

    instructions = dedent(f"""
        Implement a memory-efficient fused linear cross-entropy loss as a
        Triton GPU kernel. This is the LM-head loss used in modern LLM
        training: cross-entropy on `logits = hidden @ weight.T` where
        `hidden` is the last layer's activations and `weight` is the output
        projection matrix. The challenge is to compute the loss and its
        gradients **without ever materializing the (batch, vocab) logits
        or softmax tensors** — at vocab=131072 these matrices dominate
        peak activation memory in real LLM training.

        BACKGROUND

        Mathematical specification (matching `F.cross_entropy(logits,
        targets, reduction='mean', ignore_index=-100)`):

            logits[b, v] = sum_d hidden[b, d] * weight[v, d]
            valid[b]     = (targets[b] != -100)
            n_valid      = sum_b valid[b]
            loss         = sum_b ( valid[b] * (-log softmax(logits[b])[targets[b]]) ) / n_valid

        Gradients (mean over n_valid, not B; ignored rows contribute 0):

            grad_hidden[b, :] = valid[b] * ( sum_v softmax(b,v) * weight[v,:] - weight[targets[b],:] ) / n_valid
            grad_weight[v, :] = sum_b valid[b] * ( softmax(b,v) - 1[v==targets[b]] ) * hidden[b,:] / n_valid

        DELIVERABLE

        Create the file `{solution_path}` defining:

            class FusedLinearCrossEntropy(torch.autograd.Function):
                @staticmethod
                def forward(
                    ctx,
                    hidden: torch.Tensor,    # (B, D), fp16 OR bf16, requires_grad=True
                    weight: torch.Tensor,    # (V, D), fp16 OR bf16, requires_grad=True
                    targets: torch.Tensor,   # (B,),   int64
                ) -> torch.Tensor:           # scalar fp32 loss
                    ...
                @staticmethod
                def backward(ctx, grad_loss: torch.Tensor):
                    return grad_hidden, grad_weight, None  # grad_hidden/weight match input dtype

        The judge invokes your code as:

            loss = FusedLinearCrossEntropy.apply(hidden, weight, targets)
            loss.backward()
            grad_hidden, grad_weight = hidden.grad, weight.grad

        I/O contract:
            hidden, weight: same dtype, either float16 or bf16, contiguous, on CUDA
            targets: int64, on CUDA. A target of -100 means "ignore this row"
                     (no contribution to loss or any gradient; mean is over
                     non-ignored rows only).
            loss: scalar fp32 tensor on CUDA
            grad_hidden: same shape and dtype as hidden
            grad_weight: same shape and dtype as weight

        RULES

            - At least one function MUST be decorated with @triton.jit.
            - The submitted file MUST NOT contain references to
              torch.nn.functional.cross_entropy, torch.nn.functional.log_softmax,
              or torch.nn.functional.nll_loss. Dynamic lookups via getattr
              or string-based attribute access are also disallowed.
            - You MUST NOT allocate the full (B, V) logits matrix nor a full
              (B, V) softmax matrix. Logits should be computed in vocab tiles
              on the fly. Saving small per-row state (log-sum-exp, shape (B,))
              from forward to backward is fine.
            - Inside kernels, compute reductions and exp() in float32.
              fp16/bf16 inputs must be upcast on load; bf16 has wider range
              than fp16 but lower precision, so accumulators must still be fp32.
            - `ignore_index = -100`: rows with that target contribute zero
              to loss and zero to all gradients. Mean reduces over non-ignored
              rows only.

        CONSTRAINTS

            - B ∈ [1, 256]
            - D ∈ [128, 1024]   (hidden dim; aligns with small-LLM practice)
            - V ∈ [4096, 131072]   (up to Llama-3 vocab scale)
            - hidden.dtype == weight.dtype, one of {{torch.float16, torch.bfloat16}}
            - All inputs on CUDA

        SCORING (multi-dimensional, multiplicative)

            Final score = correctness × memory × throughput, each in [0, 1].

            1. Correctness — compared to a fp64 PyTorch ground truth across
               5 shapes × 2 dtypes × 2 target patterns (all-valid and
               some-ignored). Tolerance: 5e-3 for fp16, 5e-2 for bf16. Worse
               than tolerance on any test → score 0. Otherwise scales smoothly
               from worst-observed diff.
            2. Memory — peak GPU memory during forward+backward, measured
               as `excess = peak - (hidden + weight + grad_hidden + grad_weight)`.
               Compared to logits-matrix size `B * V * dtype_size`:
                   excess < 0.1 × logits_matrix → full credit
                   excess >= 1.0 × logits_matrix → zero credit (you allocated
                   one logits matrix worth of memory beyond the bare minimum)
            3. Throughput — measured against a clean Triton reference
               implementation (not hand-tuned ATen). The reference uses
               element-wise multiply + sum for its inner matmul rather
               than tensor cores, so a competent `tl.dot`-based agent
               implementation should comfortably exceed it.
               Score is log-scale around the reference's speed:
                   ratio  ≤ 0.1x ref  → 0      (clearly slower than naive ref)
                   ratio  =  1.0x ref → 0.5    (matches the naive reference)
                   ratio  ≥ 10x ref   → 1.0    (full credit)
               Concretely:  score = clamp(0.5 + log10(ratio) / 2, 0, 1).
               This rewards going beyond the naive reference rather than
               just matching it.

            Any dimension at 0 collapses the total to 0. Partial credit on
            each dimension multiplies, so a near-perfect submission must
            do well on all three.

        You may write helper functions and multiple kernels in the same file.
        You may test in a separate file before submitting; the file the
        judge scans is `{solution_path}`.

        Triton, torch, and numpy are installed in your environment. You can
        inspect APIs at runtime, e.g.
        `python -c "import triton.language as tl; help(tl.exp)"`,
        or read source from the installed packages directly.
    """).strip()

    judge = ExecutableJudge([
        sys.executable,
        "-m",
        "pm_env.score_linear_ce",
        solution_path,
        "/tmp/linear_ce_score.json",
    ])

    return [
        Task(
            id="fused-linear-cross-entropy",
            tools=["bash"],
            required_hardware="h100",
            steps=[
                Step(instructions=instructions, judge=judge),
            ],
        ),
    ]
