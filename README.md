# Fused Linear Cross-Entropy, RL environment

Take-home submission for Preference Model. The task asks an agentic LLM
to implement a memory-efficient fused linear cross-entropy loss
(forward + backward) as a Triton GPU kernel, the LM-head loss used in
modern LLM training. The agent must compute the loss and its gradients
**without ever materializing the (B, V) logits or softmax tensors**, a
pattern sometimes called "Cut Cross Entropy" (Wijmans et al., 2024).

## What this task tests

Three independent silent-failure surfaces, judged on continuous
multi-dimensional axes:

1. **Algorithmic correctness**, particularly the per-row gradient
   chain through the implicit `hidden @ weight.T` matmul, with
   `ignore_index = -100` masking and fp16/bf16 inputs that require
   fp32 accumulators.
2. **Memory discipline**. Peak GPU memory must be a small constant
   above `hidden + weight + grad_hidden + grad_weight`. Materializing
   either logits (B, V) or softmax (B, V) immediately fails the memory
   gate at production vocab sizes.
3. **Throughput**. The agent's kernel is benchmarked against a clean
   Triton reference. The reference is intentionally a naive
   element-wise + sum implementation, not hand-tuned ATen, so a
   competent `tl.dot`-based agent implementation should comfortably
   exceed it. Score is log-scale around the reference, so beating it
   1.5x looks meaningfully different from beating it 10x.

The composition is multiplicative: `total = correctness × memory × throughput`,
each in `[0, 1]`. Any dimension at 0 collapses the total. Partial
credit on each dimension multiplies, so a "near-perfect" submission
must do well on all three.

## Why Cut Cross Entropy specifically

Frontier-LLM training pipelines spend a non-trivial fraction of
activation memory on the (batch, vocab) logits tensor at the final
LM-head. At vocab=128k and large batch this is multi-GB, often
dominating peak memory and limiting context length. Memory-efficient
fused loss kernels (Liger-Kernel, Apple's `ml-cross-entropy`) are an
active area in production training. The implementation requires:

- An online softmax over vocab tiles (Flash-Attention-style running
  max + log-sum-exp).
- Two backward gradients (`grad_hidden` and `grad_weight`) computed
  without materializing softmax. `grad_weight` in particular requires
  a tile-batch contraction that is easy to get wrong on first attempt.
- Mixed-precision discipline: agent input is fp16 or bf16 with fp32
  accumulators throughout exp/log/reduce.

Compared to plain fused softmax cross-entropy, this task has
substantially less training-data exposure (≈2024 paper, one
production library) and more independent failure modes, making it
appropriately challenging for current frontier models.

## I/O contract

Agent submits a file `linear_ce.py` defining:

```python
class FusedLinearCrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        hidden: torch.Tensor,    # (B, D), fp16 or bf16, on CUDA, requires_grad=True
        weight: torch.Tensor,    # (V, D), fp16 or bf16, on CUDA, requires_grad=True
        targets: torch.Tensor,   # (B,),   int64.   -100 means "ignore this row"
    ) -> torch.Tensor:           # scalar fp32 loss

    @staticmethod
    def backward(ctx, grad_loss: torch.Tensor):
        return grad_hidden, grad_weight, None
```

Constraints in scoring:
- `B ∈ [1, 256]`, `D ∈ [128, 1024]`, `V ∈ [4096, 131072]`
- `hidden.dtype == weight.dtype` ∈ {fp16, bf16}
- gradients match input dtype
- Loss is mean-reduced over **non-ignored** rows.

## Scoring details

When the rollout ends (agent submits its file), the judge
(`pm_env/score_linear_ce.py`) runs once and returns three
real-valued dimension scores in `[0, 1]` plus their product as the
total. Scoring is not per-turn; the agent is free to iterate
internally, only the final submitted file is judged.

### Correctness
Compared to a fp64 PyTorch ground truth across **5 shapes × 2 dtypes
× 2 target patterns** (all-valid, some-ignored). Tolerance is the
fp16/bf16 quantization noise floor scaled up:

- fp16: 5e-3 (≈5x quantization noise of fp16 grads)
- bf16: 5e-2 (≈5x quantization noise of bf16 grads)

`worst_norm = max(observed_diff / fail_tol)`. Scores `1.0` if best,
`0.0` if worst exceeds `fail_tol` on any test:
```
score = clamp((1 - worst_norm) / 0.9, 0, 1)
```

### Memory
On 2 large-vocab shapes (B=256 V=64k and B=128 V=128k, both fp16):
```
raw_excess         = peak_bytes - (hidden + weight + grad_hidden + grad_weight)
materialized_excess = max(0, raw_excess - 20 MB)   # subtracts CUDA/Triton workspace floor
ratio              = materialized_excess / (B * V * 2)   # vs logits-matrix size
```
Score:
- `ratio < 0.1` → 1.0  (clearly no logits materialization)
- `ratio ≥ 1.0` → 0.0  (allocated one logits-matrix-worth on top)
- linearly between

### Throughput
Average ratio of agent time to reference time across the 5
correctness shapes. Reference is the same naive implementation in
`pm_env/reference_linear_ce.py`: element-wise + sum for the inner
matmul, no `tl.dot`.

```
score = clamp(0.5 + log10(avg_ratio) / 2, 0, 1)
# 0.1x ref → 0     (10x slower than naive)
# 1.0x ref → 0.5   (matches naive)
# 10x  ref → 1.0   (full credit)
```

The reference scoring 0.5 against itself is by design: the reference
is a baseline, not the upper bound. Agents have ~10x to climb before
saturating, which on Ada/Hopper is exactly where a good `tl.dot`
implementation lands.

## Reward-hacking analysis

We constructed five concrete attack vectors and verified each is
caught by the intended gate. See `scripts/reward_hacks/` for the full
"adversarial agent" submissions and `scripts/test_reward_hacks.py`
for the runner.

| Attack | Caught by |
|---|---|
| Pure PyTorch (no `@triton.jit` anywhere) | AST scan: no `@triton.jit`/`autotune`/`heuristics` found |
| Decoy `@triton.jit` + direct `F.cross_entropy(...)` | AST scan: `.cross_entropy` attribute access |
| `getattr(F, "cross_" + "entropy")` to bypass the static name scan | AST scan: dynamic-lookup builtin (`getattr`/`eval`/`exec`/`__import__`/`globals`/`vars`) banned |
| Numerically correct (fp32 internals) but materializes (B, V) `log_softmax` | Memory gate: peak excess > 1.0× logits matrix |
| Real Triton kernel called once per row from a Python `for` loop (correct but slow) | Throughput gate: avg ratio drops far below ref |

The forbidden-name string scan only triggers on string literals passed
as arguments to a Call (the only place a string can flow into a
forbidden API via `getattr`/`eval`/etc.), so docstrings and comments
that mention "cross_entropy" do not false-positive.

A vector we **deliberately allow**: agents may save a small per-row
fp32 buffer (log-sum-exp, shape `(B,)`) from forward to backward.
This is the standard fused-loss optimization and stays well below the
memory gate's noise floor.

### A real case caught by the AST gate

`out/transcript_claude-haiku-4-5-20251001_20260507_150251.json` is a
phase-1 rollout (see "Dev iteration log" below). haiku produced a
numerically correct kernel for the predecessor task, but pasted
`F.cross_entropy` into the file's `if __name__ == '__main__':` block
for self-testing and ran its own forbidden-name scan that excluded
that block. The judge's AST scan covers the whole file and flagged
`.cross_entropy`, scoring 0. This is exactly the "code works but the
agent didn't read the spec" failure the gate is designed for; the
gate caught it on a real rollout, not just on synthetic adversarial
submissions.

## Dev iteration log

The deliverable converged through three phases. Each phase corresponds
to one commit on this branch (`git log --oneline`); checking out an
earlier commit shows the code state that produced its transcripts.

### Phase 1: Plain fused cross-entropy

The original task asked for a fused log-softmax + NLL kernel directly
on a `(B, V)` logits tensor passed in by the caller, with no LM-head
linear projection and no `ignore_index`. Two transcripts:

- `out/transcript_claude-haiku-4-5-20251001_20260507_150251.json`
  (35 turns, **total 0.00**). Numerically correct, but tripped the
  AST gate. Detail above.
- `out/transcript_claude-opus-4-7_20260507_151515.json`
  (5 turns, **total 1.00**, avg 1.29× ref). opus solved the task in
  five tool calls. This was the trigger to harden the task: a top
  frontier model at full credit in five turns is not a useful RL
  signal.

### Phase 2: Cut Cross Entropy with linear-ramp throughput

We added the LM-head linear projection (`hidden @ weight.T`),
`ignore_index = -100` masking, fp16 and bf16 inputs (with fp32
accumulators), and split scoring into three independent multiplicative
dimensions. The throughput dimension was a linear ramp,
`min(avg_ratio_to_reference, 1.0)`, with full credit at any ratio
≥ 1.0× ref. Two opus rollouts:

- `out/transcript_claude-opus-4-7_20260507_175707.json`
  (70 turns, **total 1.00**, avg **1.40× ref**).
- `out/transcript_claude-opus-4-7_20260507_185457.json`
  (68 turns, **total 1.00**, avg **9.10× ref**, single shape peaked
  at 35× ref).

Same model, same task, same total score (1.00), but actual speedup
varied 7×. The linear cap collapsed the entire optimization signal
once an agent passed the reference. We changed the formula.

### Phase 3: Log-scale throughput

We changed the throughput score to
`clamp(0.5 + log10(avg_ratio) / 2, 0, 1)`: 0.5 at 1× ref, 1.0 at 10×
ref, 0 at 0.1× ref. The prompt's SCORING section was updated to
describe the new formula explicitly, including the 10× full-credit
target. Four rollouts:

- `out/transcript_claude-sonnet-4-6_20260508_002643.json`
  (73 turns, **total 0.18**, avg 1.48× ref, dim scores
  c=0.31 / m=1.0 / t=0.59).
- `out/transcript_claude-sonnet-4-6_20260508_010445.json`
  (128 turns, **total 0.68**, avg 2.29× ref, c=1.00 / m=1.0 / t=0.68).
- `out/transcript_claude-opus-4-7_20260508_015711.json`
  (81 turns, **total 0.64**, avg 1.93× ref, c=1.00 / m=1.0 / t=0.64).
- `out/transcript_claude-opus-4-7_20260508_021852.json`
  (83 turns, **total 0.38**, avg 2.40× ref, c=0.54 / m=1.0 / t=0.69).

The interesting finding: **none of the four came close to 10× ref,
despite the prompt explicitly stating that target.** Opus topped out
at 2.40× ref and sonnet at 2.29×, well below the 9.10× one opus
rollout had reached under the old (saturating) formula. Same model,
same task; visibly less optimized kernels when the optimization
target was further away.

The most plausible reading is that agents satisfice to the *visible*
optimization target. Under the old prompt ("match the reference at
1.0× → full credit") an opus that explored `tl.dot` and tile sizes
discovered, almost incidentally, a 9× kernel; the high score was a
side-effect of routine optimization. Under the new prompt ("10× → full
credit") the same model treats the threshold as a budget and stops
once the kernel "compiles, is correct, and beats the reference",
typically at ~2×. Pushing to 10× would require sustained tile-size /
warp-count tuning that the agents do not undertake when there is no
in-rollout feedback signal pulling them past "good enough".

This is itself a useful signal for RL training: agents are
threshold-aware, not capability-driven. Well-calibrated formulas
(where the anchor lines up with what the agent should reasonably
reach) likely matter more than aggressive thresholds.

In the same phase we also moved the memory gate from a plain ratio
threshold to `(excess - 20 MB) / logits_matrix_size` after observing
a constant ~17 MB of CUDA/Triton kernel-launch workspace that scales
with neither shape nor implementation quality. Agents that genuinely
materialize a (B, V) tensor still trip the gate at any
production-relevant shape; agents with merely-noisy small-B
allocations no longer false-positive.

### What the phase-3 rollouts struggle with, concretely

- **bf16 grad_hidden in small-batch corners.** Across the failing
  runs, worst-case diff is on `B=1, D=128, V=4096, bf16` where
  grad_hidden lands just inside fp16 noise but outside our 5e-2 bf16
  tolerance. Fix: keep the `softmax @ weight` accumulator in fp32
  throughout, rather than casting back to bf16 between vocab tiles.
- **Optimization stopping criterion.** See above: agents see "≥10× ref
  → full credit" but stop at ~2× because the kernel feels "fast
  enough". A real software-engineering anti-pattern that the
  log-scale throughput score makes legible.
- **`ignore_index` consistency across three kernels.** A correct
  implementation has three Triton kernels (forward, backward into
  `hidden`, backward into `weight`) and each one must independently
  zero out contributions from rows where `target == -100`, and the
  mean must be taken over `n_valid` rather than over `B`. It is easy
  to filter ignored rows in two of the three kernels and forget the
  third; runs that did this passed every `all_valid` test (because
  there were no ignored rows for the missing filter to matter on)
  but failed on the `some_ignored` test patterns.

### Summary of all 8 transcripts

| Transcript | Phase | Model | Turns | Total | Speed vs ref | c / m / t |
|---|:---:|---|---:|---:|---:|---|
| `haiku ..._150251.json` | 1 | haiku-4-5 | 35 | 0.00 | n/a | AST gate fail |
| `opus ..._151515.json` | 1 | opus-4-7 | 5 | 1.00 | 1.29× | (legacy formula) |
| `opus ..._175707.json` | 2 | opus-4-7 | 70 | 1.00 | 1.40× | 1.00 / 1.00 / 1.00 |
| `opus ..._185457.json` | 2 | opus-4-7 | 68 | 1.00 | 9.10× | 1.00 / 1.00 / 1.00 |
| `sonnet ..._002643.json` | 3 | sonnet-4-6 | 73 | 0.18 | 1.48× | 0.31 / 1.00 / 0.59 |
| `sonnet ..._010445.json` | 3 | sonnet-4-6 | 128 | 0.68 | 2.29× | 1.00 / 1.00 / 0.68 |
| `opus ..._015711.json` | 3 | opus-4-7 | 81 | 0.64 | 1.93× | 1.00 / 1.00 / 0.64 |
| `opus ..._021852.json` | 3 | opus-4-7 | 83 | 0.38 | 2.40× | 0.54 / 1.00 / 0.69 |

## Containerized verification

PM's evaluation infrastructure runs `pm_env run --config <config>`
on a host with podman or docker installed. The framework itself
calls `<runtime> build --tag pm_env --file Containerfile .` and then
spawns the agent inside that locally-built image
(`run_helpers.py:96` and `:147`). Reviewers do not need to pull a
prebuilt image; the Containerfile in this repo is the source of
truth for the agent runtime.

We did not have direct access to a docker-enabled NVIDIA host during
development (our dev clouds were themselves sandbox containers that
block docker-in-docker), so the full single-shot
`pm_env run --runtime docker` orchestration was not exercised
end-to-end. Each sub-component was verified independently:

| Sub-component | How verified |
|---|---|
| `Containerfile` builds cleanly (cu128 torch + Triton + gcc) | GitHub Actions on every push to main (`.github/workflows/build.yml`) |
| Built image: `import torch / triton / numpy`, `gcc` present | CI smoke test inside the just-built image |
| Built image: CUDA passthrough works, the reference implementation runs against fp64 ground truth | Manual `docker run` of the CI-published image on a RunPod RTX 5090 |
| Framework defaults to containerized mode and assembles the right build command | Local `pm_env run --config run_config.json` shows `Building container image with command: 'podman build --tag pm_env .'` before erroring out at "podman not installed" on an AMD-only host |
| Framework + our `tasks.py` task registry + `_update_run_config.py` overrides + MCP server + agent loop wire up correctly | Local `pm_env run --config run_config.json --no-containerized` on an AMD-only host: agent receives the full task instructions, MCP binds to 39100 as configured, agent makes its first bash tool calls, `import torch; import triton` succeed in the agent's subprocess (torch 2.11.0+cu128, Triton 3.6.0; CUDA unavailable as expected on AMD) |
| Agent + judge + transcript flow on a real CUDA GPU | 8 transcripts in `out/`, all produced via `--no-containerized` on vast.ai RTX 5090 |

The piece we could not exercise directly is `<runtime> build` followed
by the agent-spawn `<runtime> run` chained inside one `pm_env run`
invocation on a docker-enabled NVIDIA host. The build half is
covered by CI; the run half is covered by manual `docker run` of an
image built from the same Containerfile. The composition should
work, but flagging.

One concrete fix this verification surfaced: **PM's reference
Containerfile installs no C compiler**, but Triton compiles a small
CUDA driver utility module at first kernel launch and fails with
`RuntimeError: Failed to find C compiler` without one. We added
`gcc gcc-c++ python3-devel` to the `# INSTALL EXTRA SYSTEM
DEPENDENCIES HERE` slot. Without this fix, every agent submission
would fail at the first `triton.jit` invocation regardless of its
algorithmic quality.

The CI workflow also pushes the built image to this repository's
GHCR namespace (`ghcr.io/<owner>/<repo>:latest`, auto-derived from
`${{ github.repository }}`) purely as a CI artifact so build
breakage is caught before submission. PM's evaluation does not pull
from GHCR.

## Files

```
src/pm_env/
  tasks.py                  # Agent prompt (BACKGROUND/DELIVERABLE/RULES/SCORING)
  reference_linear_ce.py    # Cut CE reference (3 Triton kernels + autograd Function)
  score_linear_ce.py        # Multi-dim judge: AST + correctness + memory + throughput

scripts/                    # Dev tooling, runnable from the host with the venv
  verify_reference.py       # reference vs fp64 PyTorch ground truth
  bench_reference.py        # raw timing/memory of the reference
  test_scoring.py           # end-to-end smoke: reference fed back through scoring
  test_reward_hacks.py      # 5 fakes verified caught by their intended gate
  reward_hacks/             # 5 concrete adversarial submissions
  run_rollout.sh            # convenience wrapper for our dev rollouts
  analyze_transcript.py     # transcript inspection
  verify_triton.py          # tiny Triton-on-this-GPU sanity check

out/                        # 8 rollout transcripts referenced in dev iteration log
.github/workflows/build.yml # CI: builds Containerfile, smoke test, pushes to GHCR

Containerfile               # +gcc/g++/python3-devel for Triton runtime compile
env_requirements.txt        # pinned to torch==2.11.0+cu128 (driver 555+)
pyproject.toml              # judge-side deps; pytorch-cu128 source via tool.uv.sources
```

## Hardware requirements

The task requires a single NVIDIA CUDA GPU. The framework's
`required_hardware` field is set to `"h100"` to match PM's evaluation
infra, though we did not have direct access to an H100. Any modern
data-center or Ada-or-newer consumer GPU with ≥24 GB VRAM and a
CUDA-12.8-compatible driver (≥555) should be sufficient for the
published shapes (largest test is `B=128, V=131072, fp16`).
Development was done on a vast.ai instance (RTX 5090, 32 GB VRAM);
the image built from `Containerfile` was sanity-checked on a RunPod
RTX 5090 (`docker run` directly: imports work, GPU passthrough
works, the reference implementation completes against fp64 ground
truth). See "Containerized verification" above for the distinction
between this image-content check and a full `pm_env run` end-to-end
test. CPU-only and AMD/ROCm hosts are not supported: Triton emits
CUDA PTX, and the judge's correctness/memory/throughput measurements
all run on-device.

## Reproducing

### Containerized (PM's evaluation path)
On a host with podman or docker, an NVIDIA GPU, and a
CUDA-12.8-compatible driver:
```bash
uv sync                                # installs the framework's pm_env CLI
export ANTHROPIC_API_KEY=sk-ant-...
MODEL=claude-opus-4-7                  # any Claude id; opus produced the best transcripts

# 1. Framework writes a default run_config.json with the chosen model:
uv run pm_env create-run-config --model "$MODEL" --model-api-key "$ANTHROPIC_API_KEY" > /dev/null

# 2. Our script overrides task_id, MCP port, and transcript path
#    (reads the model from run_config.json itself):
uv run python scripts/_update_run_config.py

# 3. Framework builds the image from Containerfile, tags it `pm_env`,
#    spawns a container, drives the agent inside it, and runs the judge
#    when the agent submits:
uv run pm_env run --config run_config.json --runtime docker
# (use --runtime podman if your host has podman; podman is the framework default)
```
The transcript lands in `out/transcript_<run_id>.json` on the host
(the `out/` directory is mounted into the container by the
framework). The judge writes its score into the same transcript at
the end.

### Non-containerized (dev wrapper)
Useful when the host is itself a sandbox container that disallows
docker-in-docker (vast.ai, RunPod, AutoDL). This is how all eight
transcripts under `out/` were produced:
```bash
uv sync
export ANTHROPIC_API_KEY=sk-ant-...
bash scripts/run_rollout.sh claude-opus-4-7
```
`scripts/run_rollout.sh` is just a wrapper around the same official
commands plus `--no-containerized` and our `_update_run_config.py`
overrides. It is not a substitute for the containerized path PM
uses.

### Judge standalone (debugging an agent file)
```bash
python -m pm_env.score_linear_ce <agent_file.py> <output_score.json>
```
Bypasses the rollout entirely and just scores a `linear_ce.py`
defining `FusedLinearCrossEntropy`. Used by `scripts/test_scoring.py`
and `scripts/test_reward_hacks.py`.

## AI usage statement

This submission was developed with assistance from Claude Code,
primarily for code-level support: debugging numerical and runtime
issues, recalling Triton/PyTorch API details, writing test and
benchmark boilerplate, and a few unfamiliar standard-library APIs
(e.g. Python's `ast` module). All task-design decisions, including
the choice of Cut Cross Entropy as the target, the multi-dimensional
scoring composition, the reward-hacking attack vectors, and the
calibration adjustments, were author-driven. Claude served as a
sounding board and code assistant rather than a designer.
