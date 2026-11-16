#!/bin/bash
# Run the pipeline once into its own run directory. Used by the determinism tests.
# usage: tests/run_on_node.sh <tag> [run.py args...]     e.g. tests/run_on_node.sh gpu_a --device cuda
# Needs MTAH_ROOT (holds data/ and receives runs/package/<tag>/) and MTAH_VENV (the uv environment;
# created from uv.lock if missing). Put site settings in tests/cluster.env (not in git).
# The CSV is named after OUTNAME (default "submission"), so one run directory can hold several outputs.
D="$(cd "$(dirname "$0")" && pwd)"
[ -f "$D/cluster.env" ] && . "$D/cluster.env"
ROOT=${MTAH_ROOT:?set MTAH_ROOT}
VENV=${MTAH_VENV:?set MTAH_VENV}
TAG=$1; shift
export PYTHONUNBUFFERED=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
if [ ! -x $VENV/bin/python ]; then
  (cd "$D/.." && UV_PROJECT_ENVIRONMENT=$VENV uv sync --frozen) || exit 1
fi
mkdir -p $ROOT/runs/package/$TAG
cd "$D/.."
$VENV/bin/python run.py --data $ROOT/data --work $ROOT/runs/package/$TAG/work --out $ROOT/runs/package/$TAG/${OUTNAME:-submission}.csv "$@"
