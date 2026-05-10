# Fused Linear Cross-Entropy: an RL evaluation environment

An RL-style evaluation environment for measuring agentic LLMs on a
non-trivial GPU kernel task. The agent is asked to implement a
memory-efficient fused linear cross-entropy loss (forward + backward)
as a Triton GPU kernel — the LM-head loss used in modern LLM training —
computing the loss and its gradients **without ever materializing the
(B, V) logits or softmax tensors**, a pattern sometimes called "Cut
Cross Entropy" (Wijmans et al., 2024).

The repo is a study of how to design such an environment so it produces
a useful learning signal: the reference kernel, a multi-dimensional
judge (correctness / memory / throughput), an adversarial-submission
test suite verifying each scoring gate, and an iteration log showing
how the scoring formula evolved across three versions of the task.

## What this environment measures

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

`out/transcript_claude-haiku-4-5-20251001_20260507_150251.json` is an
iteration-1 rollout (see "Iteration log" below). haiku produced a
numerically correct kernel for the predecessor task, but pasted
`F.cross_entropy` into the file's `if __name__ == '__main__':` block
for self-testing and ran its own forbidden-name scan that excluded
that block. The judge's AST scan covers the whole file and flagged
`.cross_entropy`, scoring 0. This is exactly the "code works but the
agent didn't read the spec" failure the gate is designed for; the
gate caught it on a real rollout, not just on synthetic adversarial
submissions.

## Iteration log

The task converged through three iterations. Each iteration corresponds
to one commit on `main` (`git log --oneline`); checking out an
earlier commit shows the code state that produced its transcripts.

### Iteration 1: Plain fused cross-entropy

The first version asked for a fused log-softmax + NLL kernel directly
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

### Iteration 2: Cut Cross Entropy with linear-ramp throughput

I added the LM-head linear projection (`hidden @ weight.T`),
`ignore_index = -100` masking, fp16 and bf16 inputs (with fp32
accumulators), and split scoring into three independent multiplicative
dimensions. The throughput dimension was a linear ramp,
`min(avg_ratio_to_reference, 1.0)`, with full credit at any ratio
≥ 1.0× ref. Three opus rollouts:

- `out/transcript_claude-opus-4-7_20260507_165308.json`
  (67 turns, **total 0.32**, avg 1.76× ref, c=0.32 / m=1.00 / t=1.00).
  Correctness gate fired: worst-case grad_hidden diff on
  `B=1, D=128, V=4096, bf16` was 3.6e-2, just inside the 5e-2 bf16
  tolerance. Same small-batch bf16 corner case that bit iteration-3
  rollouts (see below).
- `out/transcript_claude-opus-4-7_20260507_175707.json`
  (70 turns, **total 1.00**, avg **1.40× ref**).
- `out/transcript_claude-opus-4-7_20260507_185457.json`
  (68 turns, **total 1.00**, avg **9.10× ref**, single shape peaked
  at 35× ref).

The two `total=1.00` rollouts illustrate the calibration problem:
same model, same task, same total score, but actual speedup varied
6.5×. The linear cap collapsed the entire optimization signal once
an agent passed the reference, so I changed the formula. (The `0.32`
rollout shows the correctness gate is independent of the throughput
formula change; that one would have scored ~the same under either
formula since correctness, not throughput, was the bottleneck.)

### Iteration 3: Log-scale throughput

I changed the throughput score to
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

In the same iteration I also moved the memory gate from a plain ratio
threshold to `(excess - 20 MB) / logits_matrix_size` after observing
a constant ~17 MB of CUDA/Triton kernel-launch workspace that scales
with neither shape nor implementation quality. Agents that genuinely
materialize a (B, V) tensor still trip the gate at any
production-relevant shape; agents with merely-noisy small-B
allocations no longer false-positive.

### What rollouts struggle with, concretely

- **bf16 grad_hidden in small-batch corners.** The `B=1, D=128,
  V=4096, bf16` configuration shows up across both iteration-2 (opus
  165308) and iteration-3 failing runs. grad_hidden values land just
  inside fp16 noise but outside our 5e-2 bf16 tolerance. Fix: keep
  the `softmax @ weight` accumulator in fp32 throughout, rather than
  casting back to bf16 between vocab tiles.
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

### Summary of all 9 transcripts

Iteration-2 totals are computed under the linear-ramp throughput
formula (any ratio ≥ 1.0 saturates to 1.00); iteration-3 totals are
under the log-scale formula. So the apparent total drop between
iterations is mostly the formula change, not actual performance
regression: the iteration-2 9.10× opus is faster than every
iteration-3 rollout but receives 1.00 because the cap saturates,
while every iteration-3 rollout receives 0.59-0.69 throughput because
log-scale spreads the band.

| Transcript | Iter | Model | Turns | Total | Speed vs ref | c / m / t |
|---|:---:|---|---:|---:|---:|---|
| `haiku ..._150251.json` | i1 | haiku-4-5 | 35 | 0.00 | n/a | AST gate fail |
| `opus ..._151515.json` | i1 | opus-4-7 | 5 | 1.00 | 1.29× | (legacy formula) |
| `opus ..._165308.json` | i2 | opus-4-7 | 67 | 0.32 | 1.76× | 0.32 / 1.00 / 1.00 |
| `opus ..._175707.json` | i2 | opus-4-7 | 70 | 1.00 | 1.40× | 1.00 / 1.00 / 1.00 |
| `opus ..._185457.json` | i2 | opus-4-7 | 68 | 1.00 | 9.10× | 1.00 / 1.00 / 1.00 |
| `sonnet ..._002643.json` | i3 | sonnet-4-6 | 73 | 0.18 | 1.48× | 0.31 / 1.00 / 0.59 |
| `sonnet ..._010445.json` | i3 | sonnet-4-6 | 128 | 0.68 | 2.29× | 1.00 / 1.00 / 0.68 |
| `opus ..._015711.json` | i3 | opus-4-7 | 81 | 0.64 | 1.93× | 1.00 / 1.00 / 0.64 |
| `opus ..._021852.json` | i3 | opus-4-7 | 83 | 0.38 | 2.40× | 0.54 / 1.00 / 0.69 |

### A note on what these transcripts are not

None of the nine are production-grade. Even the strongest rollout
(opus 185457, 9.10× ref) took 68 turns to land there, and a
competent kernel author would write something faster in fewer
iterations. We read this as evidence that the task is doing what an
RL task is supposed to do. The distance between "frontier model
out-of-the-box behavior" and "engineer-quality kernel" is exactly
the gradient an RL training loop would close; if every transcript
ended with a clean tl.dot kernel in 10 turns, the task would be too
easy to teach anything.

## Containerized verification

The framework's intended execution path is `pm_env run --config <config>`,
which builds the image from `Containerfile` and spawns the agent inside
it. I did not have access to a docker-enabled NVIDIA host during
development (the dev clouds I used block docker-in-docker), so the full
orchestration was not exercised end-to-end. I verified the components
independently:

- **Containerfile builds cleanly**: CI on every push to main
  (`.github/workflows/build.yml`), including smoke tests for
  torch/triton/numpy imports and gcc presence.
- **Built image runs on GPU**: manual `docker run` of the
  CI-published image on a RunPod RTX 5090; CUDA passthrough works,
  reference implementation passes against fp64 ground truth.
- **Framework + task wiring**: local `pm_env run --no-containerized`
  confirms the agent receives correct instructions, MCP binds, and
  tool calls execute. All 9 transcripts in `out/` were produced via
  `--no-containerized` on a vast.ai RTX 5090.

The untested piece is the composition: `<runtime> build` followed
by `<runtime> run` chained in one `pm_env run` invocation. Each
half is independently verified; the composition should work, but
flagging.

One concrete fix this verification surfaced: the upstream reference
Containerfile installs no C compiler, but Triton needs one at first
kernel launch. I added `gcc gcc-c++ python3-devel` to the
Containerfile.

## Known limitations and trade-offs

1. **Memory dimension functions as a hard gate, not a smooth signal
   among competent agents.** All iteration-3 rollouts scored 1.00 on
   memory; the prompt framing ("never materialize logits") is
   explicit enough that frontier models do not trip it. The memory
   gate's value is catching attack vectors
   (`scripts/reward_hacks/fake_materialize_softmax.py` confirms this)
   rather than differentiating competent submissions. A future
   redesign could push toward a continuous peak-memory ratio more
   aggressively, at the cost of false-positives on Triton workspace
   noise.

2. **Multiplicative scoring composition loses partial-credit signal
   at extreme failures.** `total = correctness × memory × throughput`
   means any dimension at 0 zeroes the total. This is intentional:
   it prevents reward hacking by acing one dim (e.g., a memory-clean
   but incorrect submission, or a fast but incorrect submission,
   both score 0). The trade-off is that for very-failed submissions
   we cannot distinguish "almost passed correctness" from
   "completely wrong": a kernel that misses correctness by 5× the
   tolerance and one that misses by 100× both end up at 0. Additive
   composition (or min-min) would preserve that signal but make it
   easier to hack the easier dimensions.

3. **Throughput "10× full-credit" anchor was calibrated on RTX 5090.**
   The score formula is ratio-based against the same in-repo Triton
   reference run on the same hardware, so relative ranking is robust
   to GPU choice. But the absolute "10× ref" anchor's calibration
   semantics ("≈ a competent tl.dot kernel") were chosen for the GPU
   we tested on; on H100 the same agent code may produce different
   absolute speedups and the anchor may need re-tuning.

4. **One-step task, no multi-turn shaping.** The judge runs once at
   the end; there is no interactive feedback or partial-credit
   milestones during the rollout. Two consequences: (a) at inference
   time, the agent receives no in-rollout signal pulling it past
   "good enough" (a contributor to the "agents satisfice at 2× ref"
   finding documented in the dev iteration log); (b) for downstream
   RL training, the terminal reward means every tool call across a
   70-130 turn trajectory shares the same gradient credit, which is
   high-variance and sample-inefficient compared to denser-reward
   designs.

5. **No turn budget on rollout length.** The framework does not
   enforce a maximum rollout length and our task prompt does not
   request one. Across our nine transcripts, length varied from 5
   turns (iteration-1 opus) to 128 turns (iteration-3 sonnet 010445), a 25×
   spread. At inference this only translates to API cost variance,
   but for RL training it adds variance to per-trajectory compute
   and to the gradient signal (per-tool-call credit assignment is
   muddier when trajectory length itself is high-variance). A future
   task spec could surface a soft budget in the prompt ("you have
   ~N tool calls") or a hard kill at K turns; both trade agent
   flexibility against training-time predictability.

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

out/                        # 9 rollout transcripts referenced in dev iteration log
.github/workflows/build.yml # CI: builds Containerfile, smoke test, pushes to GHCR

Containerfile               # +gcc/g++/python3-devel for Triton runtime compile
env_requirements.txt        # pinned to torch==2.11.0+cu128 (driver 555+)
pyproject.toml              # judge-side deps; pytorch-cu128 source via tool.uv.sources
```

## Hardware requirements

The task requires a single NVIDIA CUDA GPU. The framework's
`required_hardware` field is set to `"h100"`, though I did not have
direct access to an H100 during development. Any modern
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

### Containerized path
On a host with podman or docker, an NVIDIA GPU, and a
CUDA-12.8-compatible driver:
```bash
uv sync                                # installs the framework's pm_env CLI
export ANTHROPIC_API_KEY=sk-ant-...
MODEL=claude-opus-4-7                  # any Claude id; opus produced the best transcripts

# 1. Framework writes a default run_config.json with the chosen model:
uv run pm_env create-run-config --model "$MODEL" --model-api-key "$ANTHROPIC_API_KEY" > /dev/null

# 2. The helper script overrides task_id, MCP port, and transcript path
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
docker-in-docker (vast.ai, RunPod, AutoDL). This is how all nine
transcripts under `out/` were produced:
```bash
uv sync
export ANTHROPIC_API_KEY=sk-ant-...
bash scripts/run_rollout.sh claude-opus-4-7
```
`scripts/run_rollout.sh` is a wrapper around the same official
commands plus `--no-containerized` and the `_update_run_config.py`
overrides. It is not a substitute for the containerized path.

### Judge standalone (debugging an agent file)
```bash
python -m pm_env.score_linear_ce <agent_file.py> <output_score.json>
```
Bypasses the rollout entirely and just scores a `linear_ce.py`
defining `FusedLinearCrossEntropy`. Used by `scripts/test_scoring.py`
and `scripts/test_reward_hacks.py`.
