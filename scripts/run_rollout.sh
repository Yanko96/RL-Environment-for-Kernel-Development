#!/usr/bin/env bash
# Convenience wrapper used during development to drive a single rollout
# from a host with the project's venv set up. Not the official
# entrypoint: pm_env's CLI is `pm_env run --config <run_config.json>`,
# documented in pyproject.toml under [project.scripts]. This script
# only chains:
#   1. pm_env create-run-config        (framework default config)
#   2. scripts/_update_run_config.py   (our task_id / MCP port overrides)
#   3. pm_env run --no-containerized   (skip docker, see below)
#
# We force --no-containerized because the development environments we
# used (vast.ai, RunPod) are themselves sandbox containers that
# disallow docker-in-docker. PM's evaluation infrastructure runs the
# standard containerized path (the framework's default and what the
# Containerfile + GitHub Actions image is built for). This script is
# provided so reviewers can reproduce one of our dev rollouts on a
# bare host, not as a substitute for the production execution path.
#
# Usage:
#   export ANTHROPIC_API_KEY=sk-ant-...
#   bash scripts/run_rollout.sh                       # haiku (cheap)
#   bash scripts/run_rollout.sh claude-opus-4-7       # opus

set -euo pipefail

MODEL="${1:-claude-haiku-4-5-20251001}"

if [ -z "${ANTHROPIC_API_KEY:-}" ]; then
    echo "ERROR: ANTHROPIC_API_KEY is not set."
    echo "Run: export ANTHROPIC_API_KEY=sk-ant-..."
    exit 1
fi

uv run pm_env create-run-config \
    --model "$MODEL" \
    --model-api-key "$ANTHROPIC_API_KEY" \
    > /dev/null

uv run python scripts/_update_run_config.py

echo
echo ">>> Starting rollout. Streaming events to stdout."
echo ">>> Transcript will be saved to the path printed above."
echo

uv run pm_env run --config run_config.json --no-containerized
