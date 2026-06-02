#!/bin/bash
# Quick-start script for SpatialGT Agent UI (bare-metal, no Docker)
#
# Usage:
#   bash run.sh                           # default: port 8501, GPU 0
#   bash run.sh --port 8502               # custom port
#   bash run.sh --spatialgt /path/to/env  # custom SpatialGT env root
#   bash run.sh --spatialgt=/path/to/env --port 8502
#   CUDA_VISIBLE_DEVICES=1 bash run.sh    # use GPU 1
#
# This script ALWAYS launches Streamlit through a SpatialGT environment
# that has both the model stack (torch + tensorboard + SpatialGT modules + ...)
# AND the UI stack (streamlit + plotly + ...). Mixing with another conda env
# (e.g. spatialgt_ui) will cause silent ModuleNotFoundError on imports.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
PACKAGED_ENV_ROOT="$REPO_ROOT/environment/spatialgt"
LEGACY_ENV_ROOT="$REPO_ROOT/../environment/spatialgt"
if [ -x "$PACKAGED_ENV_ROOT/bin/python" ]; then
    DEFAULT_ENV_ROOT="$PACKAGED_ENV_ROOT"
elif [ -x "$LEGACY_ENV_ROOT/bin/python" ]; then
    DEFAULT_ENV_ROOT="$LEGACY_ENV_ROOT"
else
    DEFAULT_ENV_ROOT="$PACKAGED_ENV_ROOT"
fi
ENV_ROOT="$DEFAULT_ENV_ROOT"
PORT=8501

usage() {
    sed -n '1,13p' "$0"
    cat <<'EOF'

Options:
  --spatialgt PATH        SpatialGT environment root, e.g. /home/user/environment/spatialgt
  --spatialgt=PATH        Same as above
  --port PORT             Streamlit port, default 8501
  -h, --help              Show this help
EOF
}

# Load .env if present (OPENAI_API_KEY, etc.)
if [ -f "$SCRIPT_DIR/.env" ]; then
    set -a
    source "$SCRIPT_DIR/.env"
    set +a
fi

# Default GPU 0 unless caller overrides
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# Parse args before activating the environment so --spatialgt can control it.
while [[ $# -gt 0 ]]; do
    case "$1" in
        --spatialgt)
            if [[ -z "${2:-}" ]]; then
                echo "[run.sh] ERROR: --spatialgt requires an environment path." >&2
                exit 2
            fi
            ENV_ROOT="$2"; shift 2 ;;
        --spatialgt=*)
            ENV_ROOT="${1#--spatialgt=}"; shift ;;
        --spatialgt/*|--spatialgt.*)
            # Backward-friendly form: bash run.sh --spatialgt/path/to/env
            ENV_ROOT="${1#--spatialgt}"; shift ;;
        --port)
            if [[ -z "${2:-}" ]]; then
                echo "[run.sh] ERROR: --port requires a port number." >&2
                exit 2
            fi
            PORT="$2"; shift 2 ;;
        --help|-h)
            usage; exit 0 ;;
        *)
            # Treat first bare numeric arg as port for backwards compatibility.
            if [[ "$1" =~ ^[0-9]+$ ]]; then
                PORT="$1"
                shift
            else
                echo "[run.sh] ERROR: unknown argument: $1" >&2
                usage >&2
                exit 2
            fi ;;
    esac
done

ENV_ROOT="$(cd "$ENV_ROOT" 2>/dev/null && pwd || echo "$ENV_ROOT")"
ENV_BIN="$ENV_ROOT/bin"
ENV_PARENT="$(dirname "$ENV_ROOT")"

if [ ! -x "$ENV_BIN/python" ]; then
    echo "[run.sh] ERROR: SpatialGT environment python not found: $ENV_BIN/python" >&2
    echo "[run.sh] Usage: bash run.sh --spatialgt /path/to/environment/spatialgt" >&2
    exit 2
fi

# Built-in SpatialGT environment activation. This replaces the need to source
# the selected SpatialGT environment manually.
unset VIRTUAL_ENV
export VIRTUAL_ENV="$ENV_ROOT"
export PATH="$ENV_BIN:$ENV_PARENT/local/bin:$PATH"
if [ -d "$ENV_PARENT/local/lib" ]; then
    export LD_LIBRARY_PATH="$ENV_PARENT/local/lib:${LD_LIBRARY_PATH:-}"
fi
hash -r 2>/dev/null || true

# Sanity-check: streamlit / python come from the SpatialGT venv
RESOLVED_PYTHON="$(command -v python || true)"
RESOLVED_STREAMLIT="$(command -v streamlit || true)"
EXPECTED_BIN="$ENV_ROOT/bin"
REAL_RESOLVED_PYTHON="$(readlink -f "$RESOLVED_PYTHON" 2>/dev/null || echo "$RESOLVED_PYTHON")"
REAL_RESOLVED_STREAMLIT="$(readlink -f "$RESOLVED_STREAMLIT" 2>/dev/null || echo "$RESOLVED_STREAMLIT")"
REAL_EXPECTED_BIN="$(readlink -f "$EXPECTED_BIN" 2>/dev/null || echo "$EXPECTED_BIN")"

if [[ "$REAL_RESOLVED_PYTHON" != "$REAL_EXPECTED_BIN/"* ]]; then
    echo "[run.sh] ERROR: \`python\` resolves to '$RESOLVED_PYTHON', not '$EXPECTED_BIN/python'." >&2
    echo "[run.sh] Try: bash run.sh --spatialgt $ENV_ROOT" >&2
    exit 2
fi
if [[ "$REAL_RESOLVED_STREAMLIT" != "$REAL_EXPECTED_BIN/"* ]]; then
    echo "[run.sh] ERROR: \`streamlit\` resolves to '$RESOLVED_STREAMLIT', not '$EXPECTED_BIN/streamlit'." >&2
    echo "[run.sh] Install with:  $EXPECTED_BIN/python -m pip install streamlit==1.40.1" >&2
    exit 2
fi

# Verify critical imports do not fail (catches things like the
# tensorboard regression that the user just hit).
python - <<'PY'
import importlib, sys
missing = []
for m in ("torch", "tensorboard", "streamlit", "plotly",
         "anndata", "scanpy", "openai", "dotenv"):
    try:
        importlib.import_module(m)
    except Exception as e:
        missing.append(f"{m}: {e}")
if missing:
    sys.stderr.write("[run.sh] ERROR: the following imports failed in the active env:\n")
    for x in missing:
        sys.stderr.write("  - " + x + "\n")
    sys.stderr.write(
        "[run.sh] Install missing pieces, e.g.:\n"
        "  python -m pip install streamlit==1.40.1 plotly streamlit-plotly-events "
        "openai python-dotenv tensorboard\n")
    sys.exit(3)
PY

GPU_NAME="$(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")' 2>/dev/null || echo "?")"

echo "========================================="
echo "  SpatialGT Agent UI"
echo "  URL    : http://0.0.0.0:$PORT"
echo "  Python : $RESOLVED_PYTHON"
echo "  Env    : $ENV_ROOT"
echo "  GPU    : cuda:0 (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES) -> $GPU_NAME"
echo "========================================="

cd "$SCRIPT_DIR"
exec streamlit run app.py \
    --server.address 0.0.0.0 \
    --server.port "$PORT" \
    --server.maxUploadSize 2000
