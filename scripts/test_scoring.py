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

    if result["score"] < 0.5:
        raise SystemExit(
            f"reference scored {result['score']:.4f} (expected ~1.0). "
            "Either the reference is wrong or scoring is too strict."
        )


if __name__ == "__main__":
    main()
