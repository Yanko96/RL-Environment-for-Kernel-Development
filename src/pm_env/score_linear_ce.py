"""Scoring for the fused linear cross-entropy task.

Multi-dimensional score: correctness x memory x throughput, each
continuous in [0, 1]. The product is the final score, so partial
credit is awarded for "almost there" submissions but any dimension
zeroed out (e.g., a wrong gradient or a materialized logits tensor)
collapses the total to 0.
"""

import ast
import importlib.util
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import triton

from pm_env.reference_linear_ce import (
    FusedLinearCrossEntropy as RefFusedLinearCE,
    IGNORE_INDEX,
)


FORBIDDEN_NAMES = ("cross_entropy", "log_softmax", "nll_loss")
BANNED_BUILTINS = (
    "getattr", "setattr", "eval", "exec", "compile",
    "__import__", "globals", "vars",
)

# Test shapes — note V=131072 forces real LM-head scale memory pressure.
TEST_SHAPES = [
    # (B, D, V)
    (8, 256, 4096),
    (32, 512, 8192),
    (16, 512, 32768),
    (4, 1024, 131072),   # large vocab — materializing logits = 1MB even here
    (1, 128, 4096),
]

# Memory-specific shapes where B*V*2 (the materialized-logits size) is
# large enough to clearly dominate the ~17 MB constant noise floor we
# observe from CUDA / Triton kernel-launch workspace. We need
# materialized-logits >= ~32 MB to differentiate cleanly.
MEMORY_TEST_SHAPES = [
    # (B, D, V) — logits-matrix size in MB at fp16
    (256, 512, 65536),   # 32 MB logits
    (128, 512, 131072),  # 32 MB logits
]

# Empirically observed constant memory overhead per call (Triton kernel
# launch + CUDA stream workspace). Subtracted from observed excess
# before comparing to logits-matrix size.
MEMORY_NOISE_FLOOR_BYTES = 20 * 1024 * 1024  # 20 MB, slightly above the
# 17 MB we observe, to absorb run-to-run jitter.

DTYPES = [torch.float16, torch.bfloat16]

TARGETS_PATTERNS = ["all_valid", "some_ignored"]


def _make_targets(B: int, V: int, pattern: str, device: torch.device) -> torch.Tensor:
    targets = torch.randint(0, V, (B,), device=device, dtype=torch.int64)
    if pattern == "some_ignored" and B >= 2:
        n_ignored = max(1, B // 3)
        targets[:n_ignored] = IGNORE_INDEX
    return targets


def _has_triton_jit(tree: ast.Module) -> bool:
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for dec in node.decorator_list:
            target = dec.func if isinstance(dec, ast.Call) else dec
            if isinstance(target, ast.Attribute) and target.attr in ("jit", "autotune", "heuristics"):
                if isinstance(target.value, ast.Name) and target.value.id == "triton":
                    return True
            if isinstance(target, ast.Name) and target.id in ("jit", "autotune", "heuristics"):
                return True
    return False


def _find_forbidden(source: str, tree: ast.Module) -> list[str]:
    found: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr in FORBIDDEN_NAMES:
            found.append(f"attribute access .{node.attr}")
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name in FORBIDDEN_NAMES:
                    found.append(f"import: from {node.module} import {alias.name}")
        # Strings used as arguments to a Call are the only place a string
        # literal can flow into a forbidden API (e.g. `getattr(F, "..."`).
        # Free strings (docstrings, comments-as-strings) cannot bypass.
        if isinstance(node, ast.Call):
            string_args = []
            for arg in node.args:
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    string_args.append(arg.value)
            for kw in node.keywords:
                if isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
                    string_args.append(kw.value.value)
            for s in string_args:
                for name in FORBIDDEN_NAMES:
                    if name in s:
                        found.append(f"string arg to Call contains {name!r}: {s!r}")
            if isinstance(node.func, ast.Name) and node.func.id in BANNED_BUILTINS:
                found.append(f"dynamic-lookup builtin call: {node.func.id}(...)")
    return found


def _load_solution(path: str):
    spec = importlib.util.spec_from_file_location("agent_linear_ce", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot create import spec for {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["agent_linear_ce"] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------- correctness ----------

def _correctness_score(agent_class) -> tuple[float, dict]:
    """Returns (score in [0,1], details). Score 0 if any test exceeds the
    'fail' tolerance; otherwise the score scales smoothly with the worst
    observed diff."""
    diffs = []
    worst_norm = 0.0  # max(observed_diff / fail_tol) across all tests
    for B, D, V in TEST_SHAPES:
        for dtype in DTYPES:
            for tgt_pat in TARGETS_PATTERNS:
                if B == 1 and tgt_pat == "some_ignored":
                    continue  # need at least 2 rows for ignored variation
                torch.manual_seed(B * 31 + D * 17 + V + len(tgt_pat))
                hidden_data = (torch.randn(B, D, device="cuda") * 0.5).to(dtype)
                weight_data = (torch.randn(V, D, device="cuda") * 0.5).to(dtype)
                targets = _make_targets(B, V, tgt_pat, hidden_data.device)

                # fp64 ground truth
                h_ref = hidden_data.to(torch.float64).clone().requires_grad_(True)
                w_ref = weight_data.to(torch.float64).clone().requires_grad_(True)
                logits_ref = h_ref @ w_ref.T
                loss_ref = F.cross_entropy(
                    logits_ref, targets, reduction="mean", ignore_index=IGNORE_INDEX
                )
                loss_ref.backward()

                h_ag = hidden_data.detach().clone().requires_grad_(True)
                w_ag = weight_data.detach().clone().requires_grad_(True)
                loss_ag = agent_class.apply(h_ag, w_ag, targets)
                loss_ag.backward()

                fwd_d = (loss_ref.float() - loss_ag.float()).abs().item()
                gh_d = (h_ref.grad.float() - h_ag.grad.float()).abs().max().item()
                gw_d = (w_ref.grad.float() - w_ag.grad.float()).abs().max().item()

                fail_tol = 5e-2 if dtype == torch.bfloat16 else 5e-3
                worst = max(fwd_d, gh_d, gw_d)
                worst_norm = max(worst_norm, worst / fail_tol)

                diffs.append({
                    "B": B, "D": D, "V": V,
                    "dtype": str(dtype).replace("torch.", ""),
                    "targets": tgt_pat,
                    "fwd": fwd_d, "grad_hidden": gh_d, "grad_weight": gw_d,
                    "fail_tol": fail_tol,
                })

                del h_ref, w_ref, h_ag, w_ag, loss_ref, loss_ag
                torch.cuda.empty_cache()

    if worst_norm >= 1.0:
        return 0.0, {"diffs": diffs, "worst_norm": worst_norm, "score": 0.0}

    # Smooth score: 1.0 if worst < 0.1 of fail tolerance; linearly down to 0 at fail_tol.
    score = max(0.0, min(1.0, (1.0 - worst_norm) / 0.9))
    return score, {"diffs": diffs, "worst_norm": worst_norm, "score": score}


# ---------- memory ----------

def _memory_score(agent_class) -> tuple[float, dict]:
    """Strict memory check designed to detect logits or softmax
    materialization. Logits matrix is (B, V) * dtype_size bytes.
    The check: peak memory excess (above hidden + weight + grads
    baseline) must be << logits_matrix size.

    A warmup pass is run before measurement to amortize Triton kernel
    compilation and one-time CUDA context overhead, which would
    otherwise add ~10-20 MB of noise per shape.
    """
    results = []
    worst_excess_ratio = 0.0

    for B, D, V in MEMORY_TEST_SHAPES:
        torch.manual_seed(0)
        dtype_size = 2  # fp16
        hidden_bytes = B * D * dtype_size
        weight_bytes = V * D * dtype_size
        baseline = 2 * (hidden_bytes + weight_bytes)
        logits_bytes = B * V * dtype_size

        torch.cuda.empty_cache()
        hidden = torch.randn(B, D, device="cuda", dtype=torch.float16, requires_grad=True)
        weight = torch.randn(V, D, device="cuda", dtype=torch.float16, requires_grad=True)
        targets = _make_targets(B, V, "all_valid", hidden.device)

        # Warmup: compile kernels, load CUDA caches.
        loss = agent_class.apply(hidden, weight, targets)
        loss.backward()
        hidden.grad = None
        weight.grad = None
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        # Measurement run.
        loss = agent_class.apply(hidden, weight, targets)
        loss.backward()
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated()

        raw_excess = max(0, int(peak) - baseline)
        # Subtract the constant kernel-launch / CUDA workspace floor.
        materialized_excess = max(0, raw_excess - MEMORY_NOISE_FLOOR_BYTES)
        excess_vs_logits = materialized_excess / max(logits_bytes, 1)
        worst_excess_ratio = max(worst_excess_ratio, excess_vs_logits)

        results.append({
            "B": B, "D": D, "V": V,
            "peak_bytes": int(peak),
            "baseline_bytes": baseline,
            "logits_matrix_bytes": logits_bytes,
            "raw_excess_bytes": raw_excess,
            "noise_floor_bytes": MEMORY_NOISE_FLOOR_BYTES,
            "materialized_excess_bytes": materialized_excess,
            "excess_vs_logits": excess_vs_logits,
        })

        del hidden, weight, targets, loss
        torch.cuda.empty_cache()

    # Score: excess < 0.1x logits matrix → full credit; excess > 1.0x → zero.
    # (At excess == logits_matrix_bytes, agent has effectively materialized
    # one logits tensor.)
    if worst_excess_ratio >= 1.0:
        return 0.0, {"results": results, "worst_excess_vs_logits": worst_excess_ratio, "score": 0.0}
    score = max(0.0, min(1.0, (1.0 - worst_excess_ratio) / 0.9))
    return score, {"results": results, "worst_excess_vs_logits": worst_excess_ratio, "score": score}


# ---------- throughput ----------

def _throughput_score(agent_class) -> tuple[float, dict]:
    results = []
    for B, D, V in TEST_SHAPES:
        torch.manual_seed(0)
        hidden = torch.randn(B, D, device="cuda", dtype=torch.float16, requires_grad=True)
        weight = torch.randn(V, D, device="cuda", dtype=torch.float16, requires_grad=True)
        targets = _make_targets(B, V, "all_valid", hidden.device)

        def run_agent():
            hidden.grad = None
            weight.grad = None
            loss = agent_class.apply(hidden, weight, targets)
            loss.backward()

        def run_ref():
            hidden.grad = None
            weight.grad = None
            loss = RefFusedLinearCE.apply(hidden, weight, targets)
            loss.backward()

        t_agent = triton.testing.do_bench(run_agent)
        t_ref = triton.testing.do_bench(run_ref)
        ratio = t_ref / t_agent
        results.append({
            "B": B, "D": D, "V": V,
            "t_agent_ms": t_agent, "t_ref_ms": t_ref, "speed_ratio": ratio,
        })

        del hidden, weight, targets
        torch.cuda.empty_cache()

    avg_ratio = sum(r["speed_ratio"] for r in results) / len(results)
    score = _throughput_score_from_ratio(avg_ratio)
    return score, {"results": results, "avg_speed_ratio": avg_ratio, "score": score}


def _throughput_score_from_ratio(avg_ratio: float) -> float:
    """Log-scale throughput score so different orders-of-magnitude
    speedups receive different credit.

    - avg_ratio == 1.0  → 0.5  (matches the naive Triton reference)
    - avg_ratio == 10x  → 1.0  (well-optimized tl.dot kernel)
    - avg_ratio < 0.1   → 0.0  (clearly slower than the naive reference)

    Formula: 0.5 + log10(ratio) / 2, clipped to [0, 1]. Reference
    matches anchor at 0.5 by design — agents that match the naive ref
    receive partial credit but not full credit.
    """
    if avg_ratio <= 0:
        return 0.0
    return max(0.0, min(1.0, 0.5 + math.log10(avg_ratio) / 2.0))


def _write(path: str, score: float, metadata: dict) -> None:
    Path(path).write_text(json.dumps({"score": score, "metadata": metadata}, indent=2))


def main() -> None:
    solution_path = sys.argv[1]
    output_path = sys.argv[2]

    metadata: dict = {}

    p = Path(solution_path)
    if not p.exists() or p.stat().st_size == 0:
        return _write(output_path, 0.0, {**metadata, "error": "solution file missing or empty"})

    source = p.read_text()
    metadata["source_lines"] = len(source.splitlines())

    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return _write(output_path, 0.0, {**metadata, "error": f"syntax error: {e}"})

    if not _has_triton_jit(tree):
        return _write(output_path, 0.0, {**metadata, "error": "no @triton.jit (or autotune/heuristics) kernel found"})

    forbidden = _find_forbidden(source, tree)
    if forbidden:
        return _write(output_path, 0.0, {**metadata, "error": "forbidden identifiers used", "forbidden_found": forbidden})

    try:
        mod = _load_solution(solution_path)
    except Exception as e:
        return _write(output_path, 0.0, {**metadata, "error": f"import failed: {type(e).__name__}: {e}"})

    if not hasattr(mod, "FusedLinearCrossEntropy"):
        return _write(output_path, 0.0, {**metadata, "error": "module does not define FusedLinearCrossEntropy"})

    agent_class = mod.FusedLinearCrossEntropy

    try:
        c_score, c_meta = _correctness_score(agent_class)
    except Exception as e:
        return _write(output_path, 0.0, {**metadata, "error": f"correctness check raised: {type(e).__name__}: {e}"})
    metadata["correctness"] = c_meta

    try:
        m_score, m_meta = _memory_score(agent_class)
    except Exception as e:
        return _write(output_path, 0.0, {**metadata, "error": f"memory check raised: {type(e).__name__}: {e}", "correctness_score": c_score})
    metadata["memory"] = m_meta

    try:
        t_score, t_meta = _throughput_score(agent_class)
    except Exception as e:
        return _write(output_path, 0.0, {**metadata, "error": f"throughput check raised: {type(e).__name__}: {e}", "correctness_score": c_score, "memory_score": m_score})
    metadata["throughput"] = t_meta

    metadata["dimension_scores"] = {
        "correctness": c_score,
        "memory": m_score,
        "throughput": t_score,
    }
    score = c_score * m_score * t_score
    return _write(output_path, score, metadata)


if __name__ == "__main__":
    main()
