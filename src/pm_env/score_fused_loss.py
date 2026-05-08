import ast
import importlib.util
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import triton

from pm_env.reference_fused_loss import FusedCrossEntropy as RefFusedCE


FORBIDDEN_NAMES = ("cross_entropy", "log_softmax", "nll_loss")

# Dynamic-lookup builtins. These let the agent assemble forbidden names
# at runtime (e.g. `getattr(F, "cross_" + "entropy")`) and bypass
# static identifier and string-literal scans. The submitted file has no
# legitimate need for any of them.
BANNED_BUILTINS = (
    "getattr", "setattr", "eval", "exec", "compile",
    "__import__", "globals", "vars",
)

TEST_SHAPES = [
    (8, 1024),
    (32, 4096),
    (128, 8192),
    (1, 1024),
    (32, 1031),
]

DISTRIBUTIONS = [
    ("randn", lambda B, V: torch.randn(B, V, device="cuda", dtype=torch.float32)),
    ("randn_x100", lambda B, V: torch.randn(B, V, device="cuda", dtype=torch.float32) * 100),
]

CORRECTNESS_TOL = 1e-4
MEMORY_RATIO_THRESHOLD = 1.5  # peak memory may be at most 1.5x of (logits + grad_logits)


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
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            for name in FORBIDDEN_NAMES:
                if name in node.value:
                    found.append(f"string literal contains {name!r}: {node.value!r}")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in BANNED_BUILTINS:
            found.append(f"dynamic-lookup builtin call: {node.func.id}(...)")
    return found


def _load_solution(path: str):
    spec = importlib.util.spec_from_file_location("agent_fused_loss", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot create import spec for {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["agent_fused_loss"] = mod
    spec.loader.exec_module(mod)
    return mod


def _check_correctness(agent_class) -> tuple[bool, dict]:
    diffs = []
    for B, V in TEST_SHAPES:
        for dist_name, mk_logits in DISTRIBUTIONS:
            torch.manual_seed(B * 31 + V + len(dist_name))
            base = mk_logits(B, V)
            targets = torch.randint(0, V, (B,), device="cuda", dtype=torch.int64)

            logits_ref = base.clone().requires_grad_(True)
            loss_ref = F.cross_entropy(logits_ref, targets, reduction="mean")
            loss_ref.backward()

            logits_agent = base.clone().requires_grad_(True)
            loss_agent = agent_class.apply(logits_agent, targets)
            loss_agent.backward()

            fwd_diff = (loss_ref.detach() - loss_agent.detach()).abs().item()
            bwd_diff = (logits_ref.grad - logits_agent.grad).abs().max().item()
            entry = {
                "B": B, "V": V, "dist": dist_name,
                "fwd_diff": fwd_diff, "bwd_diff": bwd_diff,
            }
            diffs.append(entry)
            if fwd_diff > CORRECTNESS_TOL or bwd_diff > CORRECTNESS_TOL:
                return False, {"diffs": diffs, "failed_at": entry, "tol": CORRECTNESS_TOL}
    return True, {"diffs": diffs, "tol": CORRECTNESS_TOL}


def _check_memory(agent_class) -> tuple[bool, dict]:
    results = []
    for B, V in TEST_SHAPES:
        torch.manual_seed(0)
        logits_bytes = B * V * 4
        baseline = 2 * logits_bytes  # theoretical: logits + grad_logits

        torch.cuda.empty_cache()
        logits = torch.randn(B, V, device="cuda", dtype=torch.float32, requires_grad=True)
        targets = torch.randint(0, V, (B,), device="cuda", dtype=torch.int64)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

        loss = agent_class.apply(logits, targets)
        loss.backward()
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated()

        ratio = peak / baseline
        entry = {
            "B": B, "V": V,
            "peak_bytes": int(peak),
            "baseline_bytes": int(baseline),
            "ratio": ratio,
        }
        results.append(entry)

        del logits, targets, loss
        torch.cuda.empty_cache()

        if ratio > MEMORY_RATIO_THRESHOLD:
            return False, {
                "results": results,
                "failed_at": entry,
                "threshold": MEMORY_RATIO_THRESHOLD,
            }
    return True, {"results": results, "threshold": MEMORY_RATIO_THRESHOLD}


def _bench_throughput(agent_class) -> dict:
    results = []
    for B, V in TEST_SHAPES:
        torch.manual_seed(0)
        logits = torch.randn(B, V, device="cuda", dtype=torch.float32, requires_grad=True)
        targets = torch.randint(0, V, (B,), device="cuda", dtype=torch.int64)

        def run_agent():
            logits.grad = None
            loss = agent_class.apply(logits, targets)
            loss.backward()

        def run_ref():
            logits.grad = None
            loss = RefFusedCE.apply(logits, targets)
            loss.backward()

        t_agent = triton.testing.do_bench(run_agent)
        t_ref = triton.testing.do_bench(run_ref)
        results.append({
            "B": B, "V": V,
            "t_agent_ms": t_agent,
            "t_ref_ms": t_ref,
            "speed_ratio": t_ref / t_agent,
        })

        del logits, targets
        torch.cuda.empty_cache()

    avg_ratio = sum(r["speed_ratio"] for r in results) / len(results)
    return {"results": results, "avg_speed_ratio": avg_ratio}


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

    if not hasattr(mod, "FusedCrossEntropy"):
        return _write(output_path, 0.0, {**metadata, "error": "module does not define FusedCrossEntropy"})

    agent_class = mod.FusedCrossEntropy

    try:
        ok, corr_meta = _check_correctness(agent_class)
    except Exception as e:
        return _write(output_path, 0.0, {**metadata, "error": f"correctness check raised: {type(e).__name__}: {e}"})
    metadata["correctness"] = corr_meta
    if not ok:
        return _write(output_path, 0.0, {**metadata, "error": "correctness failed"})

    try:
        ok, mem_meta = _check_memory(agent_class)
    except Exception as e:
        return _write(output_path, 0.0, {**metadata, "error": f"memory check raised: {type(e).__name__}: {e}"})
    metadata["memory"] = mem_meta
    if not ok:
        return _write(output_path, 0.0, {**metadata, "error": "memory exceeded threshold; likely materializing softmax"})

    try:
        bench_meta = _bench_throughput(agent_class)
    except Exception as e:
        return _write(output_path, 0.0, {**metadata, "error": f"benchmark raised: {type(e).__name__}: {e}"})
    metadata["throughput"] = bench_meta

    score = max(0.0, min(1.0, bench_meta["avg_speed_ratio"]))
    return _write(output_path, score, metadata)


if __name__ == "__main__":
    main()
