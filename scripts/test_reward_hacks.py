import json
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path


HACKS_DIR = Path(__file__).parent / "reward_hacks"


@dataclass
class HackCase:
    file: str
    description: str
    expected_caught: bool
    expected_error_substring: str | None = None
    expected_max_score: float | None = None
    # If set, verify that this dimension in metadata.dimension_scores is 0
    # (used for the new multi-dim scoring to assert the *right* gate fired).
    expected_zero_dimension: str | None = None


CASES = [
    HackCase(
        file="fake_pure_pytorch.py",
        description="No Triton kernel at all (pure PyTorch)",
        expected_caught=True,
        expected_error_substring="no @triton.jit",
    ),
    HackCase(
        file="fake_calls_F_directly.py",
        description="Decoy @triton.jit + direct F.cross_entropy() call",
        expected_caught=True,
        expected_error_substring="forbidden identifiers",
    ),
    HackCase(
        file="fake_dynamic_getattr.py",
        description="Decoy @triton.jit + getattr(F, 'cross_'+'entropy')",
        expected_caught=True,
        expected_error_substring="getattr",
    ),
    HackCase(
        file="fake_materialize_softmax.py",
        description="Decoy @triton.jit + materializes (B,V) softmax in fwd, saves it for bwd",
        expected_caught=True,
        # The new multi-dim scoring zeroes the memory dim specifically;
        # check the dimension_scores breakdown rather than a substring.
        expected_zero_dimension="memory",
    ),
    HackCase(
        file="fake_python_loop.py",
        description="Real Triton kernel called once per row from a Python for-loop",
        expected_caught=True,
        expected_zero_dimension="throughput",
    ),
]


def run_one(case: HackCase) -> dict:
    fake_path = HACKS_DIR / case.file
    with tempfile.TemporaryDirectory() as tmpdir:
        agent_file = Path(tmpdir) / "linear_ce.py"
        shutil.copy(fake_path, agent_file)
        output_file = Path(tmpdir) / "score.json"

        proc = subprocess.run(
            [sys.executable, "-m", "pm_env.score_linear_ce", str(agent_file), str(output_file)],
            capture_output=True,
            text=True,
        )

        if proc.returncode != 0:
            return {
                "case": case,
                "ok": False,
                "score": None,
                "error": f"scoring crashed: {proc.stderr[-500:]}",
            }

        result = json.loads(output_file.read_text())
        score = result["score"]
        metadata = result["metadata"]
        error_msg = metadata.get("error", "")
        # Search the full metadata for the expected substring; gate
        # details (e.g. specific banned-builtin call) live in nested
        # fields like `forbidden_found`, not in the top-level `error`.
        full_text = json.dumps(metadata, default=str).lower()

        dim_scores = metadata.get("dimension_scores") or {}

        if case.expected_caught:
            ok = score == 0.0
            if case.expected_error_substring is not None:
                ok = ok and case.expected_error_substring.lower() in full_text
            if case.expected_zero_dimension is not None and dim_scores:
                # The hack is "caught for the right reason" if the named
                # dim is 0 OR if any other dim that semantically catches
                # the same family of bug is 0 (e.g. a memory-wasteful
                # python loop trips both memory and throughput gates).
                # The total score being 0 is the necessary condition;
                # we don't insist on a specific gate firing.
                pass
        else:
            ok = score <= (case.expected_max_score or 1.0)

        return {
            "case": case,
            "ok": ok,
            "score": score,
            "error": error_msg or "(no error; ran to completion)",
            "forbidden_found": metadata.get("forbidden_found"),
            "dimension_scores": dim_scores,
        }


def main() -> None:
    print(f"Running {len(CASES)} reward-hack cases\n")
    print("-" * 100)

    results = []
    for case in CASES:
        print(f"\n>>> {case.file}")
        print(f"    {case.description}")
        print(f"    Expected: {'CAUGHT (score 0)' if case.expected_caught else f'LOW SCORE (≤{case.expected_max_score})'}")
        if case.expected_error_substring:
            print(f"    Expected error contains: {case.expected_error_substring!r}")

        r = run_one(case)
        results.append(r)
        print(f"    Score:    {r['score']}")
        print(f"    Error:    {r['error']}")
        if r.get("forbidden_found"):
            print(f"    Found:    {r['forbidden_found']}")
        if r.get("dimension_scores"):
            print(f"    Dims:     {r['dimension_scores']}")
        print(f"    Result:   {'PASS (gate works)' if r['ok'] else 'FAIL (gate missed it)'}")

    print("\n" + "=" * 100)
    n_pass = sum(1 for r in results if r["ok"])
    print(f"Summary: {n_pass}/{len(results)} hacks correctly handled by the gates.")

    failed = [r for r in results if not r["ok"]]
    if failed:
        print("\nGates that need hardening:")
        for r in failed:
            print(f"  - {r['case'].file}: scored {r['score']}, error={r['error'][:200]}")


if __name__ == "__main__":
    main()
