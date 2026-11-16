#!/bin/bash
# Train seeds of one config on every external match (no hold-out).
# usage: train_seeds.sh <name> "<train args>" "<seeds>" <fps>      env TRAINSETS overrides the datasets
R=${MTAH_ROOT:-.}/runs/seqmodel
W="$(cd "$(dirname "$0")" && pwd)"
NAME="$1"; ARGS="$2"; SEEDS="$3"; export SEQ_FPS="${4:-12.5}"; TRAINSETS="${TRAINSETS:-dfl metrica sc}"
mkdir -p $R/$NAME
for s in $SEEDS; do
  $W/run.sh train.py --out $R/$NAME/s$s --train $TRAINSETS --w-sc 0.5 --domtok --video-mix --spell-w 0.5 --seed $s --eval-every 100000 $ARGS > $R/$NAME/s$s.log 2>&1 &
done
wait
echo SEEDS_DONE $NAME $SEEDS
