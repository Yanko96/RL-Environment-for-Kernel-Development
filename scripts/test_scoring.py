import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


REFERENCE = Path(__file__).parent.parent / "src" / "pm_env" / "reference_fused_loss.py"


def main() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        agent_file = Path(tmpdir) / "fused_loss.py"
        shutil.copy(REFERENCE, agent_file)
        output_file = Path(tmpdir) / "score.json"

        proc = subprocess.run(
            [sys.executable, "-m", "pm_env.score_fused_loss", str(agent_file), str(output_file)],
            capture_output=True,
            text=True,
        )

        if proc.returncode != 0:
            print("STDOUT:", proc.stdout)
            print("STDERR:", proc.stderr)
            raise SystemExit(f"scoring script crashed with code {proc.returncode}")

        result = json.loads(output_file.read_text())

    print(f"score: {result['score']:.4f}")
    print()
    print("metadata:")
    print(json.dumps(result["metadata"], indent=2))

    if result["score"] < 0.5:
        raise SystemExit(
            f"reference solution scored {result['score']:.4f} (expected ~1.0). "
            "Either the reference is wrong or the scoring script is too strict."
        )


if __name__ == "__main__":
    main()
