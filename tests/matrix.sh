#!/bin/bash
# Run every ball source of one config into one run directory, reusing its cached ball track.
# usage: tests/matrix.sh <tag> [device] [config file] [prefix]
#   outputs runs/package/<tag>/<prefix>_{video,provided,avg}.csv      (prefix defaults to v2)
T=$1; DEV=${2:-cuda}
D="$(cd "$(dirname "$0")" && pwd)"
CFG=${3:-$D/../config.json}; PRE=${4:-v2}
for src in video provided avg; do
  OUTNAME=${PRE}_$src $D/run_on_node.sh $T --device $DEV --ball-source $src --config $CFG | grep -E '"csv"|"ball_infer"|"seq"'
done
echo MATRIX_DONE $T
