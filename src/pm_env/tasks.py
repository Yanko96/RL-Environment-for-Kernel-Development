import sys
from textwrap import dedent

from pm_env.get_data_dir import get_env_data_dir
from pm_env.judges.executable_judge import ExecutableJudge
from pm_env.schemas.evaluation_run_config import EvaluationRunConfig
from pm_env.task import Step, Task


def get_tasks(config: EvaluationRunConfig) -> list[Task]:
    solution_path = f"{get_env_data_dir()}/fused_loss.py"

    instructions = dedent(f"""
        Implement a fused cross-entropy loss (forward + backward) as a Triton GPU kernel.

        BACKGROUND

        Standard cross-entropy with mean reduction over a batch:

            loss = mean over i of (-log(softmax(logits[i])[targets[i]]))

        The naive implementation materializes a (batch, vocab) softmax tensor
        in GPU memory. A fused implementation computes per-row log-sum-exp
        directly from logits, and in backward writes the gradient
        `(softmax - one_hot) * grad_loss / batch` without ever allocating the
        full softmax tensor.

        DELIVERABLE

        Create the file `{solution_path}` defining:

            class FusedCrossEntropy(torch.autograd.Function):
                @staticmethod
                def forward(ctx, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
                    ...
                @staticmethod
                def backward(ctx, grad_loss: torch.Tensor):
                    ...

        The judge invokes your code as:

            loss = FusedCrossEntropy.apply(logits, targets)
            loss.backward()
            grad_logits = logits.grad

        I/O contract:
            logits        (batch, vocab), float32, contiguous, on CUDA, requires_grad=True
            targets       (batch,), int64, on CUDA
            return value  scalar float32 tensor on CUDA, equal to the mean per-row loss

        RULES

            - At least one function in the file MUST be decorated with @triton.jit.
            - The submitted file MUST NOT contain references to
              torch.nn.functional.cross_entropy, torch.nn.functional.log_softmax,
              or torch.nn.functional.nll_loss. Dynamic lookups via getattr or
              string-based attribute access are also disallowed.
            - Loss is mean-reduced over the batch dimension. Sum and none
              reductions are out of scope.
            - Do not allocate a full (batch, vocab) softmax tensor in either
              forward or backward. Saving small per-row state (e.g. log-sum-exp,
              shape (batch,)) from forward to backward is fine; allocating a
              second (batch, vocab) tensor besides the gradient output is not.

        CONSTRAINTS

            - vocab is in [1024, 8192].
            - batch is in [1, 256].
            - logits dtype is float32. You do not need to handle float16 or bfloat16.
            - All inputs are on CUDA.

        SCORING

            1. Correctness — forward and backward must match PyTorch's
               F.cross_entropy within 1e-4 absolute tolerance on multiple
               shapes. Failure here scores 0.
            2. Memory — peak GPU memory during forward+backward must be close
               to the theoretical minimum (logits + grad_logits). Materializing
               a softmax tensor scores 0.
            3. Throughput — the speed bar is a clean Triton reference, not
               hand-tuned ATen. Slow implementations (e.g. Python loops over
               rows calling Triton per row) lose points; the kernel should
               handle the batch in a single launch.

        You may write helper functions and multiple kernels in the same file.
        You may test your implementation against PyTorch in a separate file
        before submitting; the file scanned by the judge is `{solution_path}`.

        Triton, torch, and numpy are installed in your environment. You can
        inspect APIs at runtime, e.g.
        `python -c "import triton.language as tl; help(tl.exp)"`,
        or read source from the installed packages directly.
    """).strip()

    judge = ExecutableJudge([
        sys.executable,
        "-m",
        "pm_env.score_fused_loss",
        solution_path,
        "/tmp/fused_loss_score.json",
    ])

    return [
        Task(
            id="fused-cross-entropy",
            tools=["bash"],
            required_hardware="h100",
            steps=[
                Step(instructions=instructions, judge=judge),
            ],
        ),
    ]
