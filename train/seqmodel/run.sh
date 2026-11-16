#!/bin/bash
# usage: run.sh <script.py> [args...]   (runs inside train/seqmodel with the training env)
# MTAH_ROOT must hold external/<dataset>/common/*.npz and official-repo/ (the organisers' repository).
cd "$(dirname "$0")"
export PYTHONUNBUFFERED=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
exec uv run -q --with torch==2.15.0 --with numpy==2.5.4 --with scipy --with pandas python "$@"
