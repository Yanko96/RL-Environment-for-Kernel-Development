import datetime
import json
import sys
from pathlib import Path


def main() -> None:
    model = sys.argv[1]
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{model}_{ts}"

    Path("out").mkdir(exist_ok=True)

    config_path = Path("run_config.json")
    config = json.loads(config_path.read_text())

    config["task_id"] = "fused-linear-cross-entropy"
    config["run_id"] = run_id
    config["transcript_file"] = f"out/transcript_{run_id}.json"
    # Default MCP port 8080 collides with vast.ai's Jupyter / Tensorboard
    # / etc. Pick a high port that nothing common binds to.
    config["mcp_server_config"] = {"host": "127.0.0.1", "port": 39100}

    config_path.write_text(json.dumps(config, indent=2))

    print(f"Configured run:")
    print(f"  task_id:    {config['task_id']}")
    print(f"  run_id:     {config['run_id']}")
    print(f"  transcript: {config['transcript_file']}")


if __name__ == "__main__":
    main()
