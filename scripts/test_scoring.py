import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


REFERENCE = Path(__file__).parent.parent / "src" / "pm_env" / "reference_linear_ce.py"


def main() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        agent_file = Path(tmpdir) / "linear_ce.py"
        # The agent file must define `FusedLinearCrossEntropy`. The reference
        # file already does — just import-and-re-export to mimic what an
        # agent submission looks like.
        contents = REFERENCE.read_text()
        agent_file.write_text(contents)

        output_file = Path(tmpdir) / "score.json"

        proc = subprocess.run(
            [sys.executable, "-m", "pm_env.score_linear_ce", str(agent_file), str(output_file)],
            capture_output=True,
            text=True,
        )

        if proc.returncode != 0:
            print("STDOUT:", proc.stdout)
            print("STDERR:", proc.stderr)
            raise SystemExit(f"scoring crashed with code {proc.returncode}")

        result = json.loads(output_file.read_text())

    print(f"score: {result['score']:.4f}")
    print()
    if "dimension_scores" in result["metadata"]:
        print("dimension scores:")
        for k, v in result["metadata"]["dimension_scores"].items():
            print(f"  {k:12s} {v:.4f}")
    print()
    print("metadata:")
    print(json.dumps(result["metadata"], indent=2))

    # The reference solution is intentionally a baseline (not a tensor-core
    # implementation), so its throughput score is now 0.5 by design (matches
    # itself = midpoint on the log-scale formula). Total = correctness x
    # memory x throughput should be near 0.5 for the reference, not 1.0.
    # We assert only that correctness and memory are passing.
    dim_scores = result["metadata"].get("dimension_scores")
    if dim_scores:
        if isinstance(dim_scores, str):
            import ast as _ast
            dim_scores = _ast.literal_eval(dim_scores)
        if dim_scores.get("correctness", 0) < 0.9:
            raise SystemExit(f"reference correctness too low: {dim_scores}")
        if dim_scores.get("memory", 0) < 0.9:
            raise SystemExit(f"reference memory too low: {dim_scores}")
    elif result["score"] < 0.3:
        raise SystemExit(
            f"reference scored {result['score']:.4f}, "
            "expected total ~0.5 (correctness*memory*0.5_throughput) on the new formula."
        )


if __name__ == "__main__":
    main()
