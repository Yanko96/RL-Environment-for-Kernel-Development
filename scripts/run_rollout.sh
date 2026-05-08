#!/usr/bin/env bash
# Run a single agent rollout against the fused-cross-entropy task.
# Default model is haiku (cheap). Pass model id as $1 to override.
#
# Usage:
#   export ANTHROPIC_API_KEY=sk-ant-...
#   bash scripts/run_rollout.sh                       # haiku
#   bash scripts/run_rollout.sh claude-opus-4-7       # opus
#
# If routing through an SSH-reverse-tunnel proxy (because the cloud GPU
# can't reach api.anthropic.com directly), set HTTPS_PROXY before
# invoking. NO_PROXY is forced below to exclude local MCP traffic,
# which would otherwise be misrouted through the proxy.

set -euo pipefail

# Local MCP server traffic must bypass any HTTP(S)_PROXY the user has set.
export NO_PROXY="${NO_PROXY:-localhost,127.0.0.1,0.0.0.0}"

MODEL="${1:-claude-haiku-4-5-20251001}"

if [ -z "${ANTHROPIC_API_KEY:-}" ]; then
    echo "ERROR: ANTHROPIC_API_KEY is not set."
    echo "Run: export ANTHROPIC_API_KEY=sk-ant-..."
    exit 1
fi

# Generate a fresh run_config.json. This sets a fresh run_id (UUID) and
# defaults that we then override.
uv run --no-sync pm_env create-run-config \
    --model "$MODEL" \
    --model-api-key "$ANTHROPIC_API_KEY" \
    > /dev/null

# Override task_id, run_id (timestamp-based for sortability), transcript path.
uv run --no-sync python scripts/_update_run_config.py "$MODEL"

echo
echo ">>> Starting rollout. Streaming events to stdout."
echo ">>> Transcript will be saved to the path printed above."
echo

uv run --no-sync pm_env run --config run_config.json --no-containerized
