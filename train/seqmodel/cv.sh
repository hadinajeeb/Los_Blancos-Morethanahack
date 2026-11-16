#!/bin/bash
# 4-fold cross-validation over the 7 DFL matches for one variant.
# usage: cv.sh <variant-name> "<extra train args>" <fps> "<fold ids>" [parallel]
R=${MTAH_ROOT:-.}/runs/seqmodel/cv
W=$(cd "$(dirname "$0")" && pwd)
V="$1"; EXTRA="$2"; export SEQ_FPS="$3"; FOLDS="${4:-0 1 2 3}"; PAR="${5:-2}"; TRAINSETS="${TRAINSETS:-dfl metrica sc}"
mkdir -p $R
F[0]="DFL-MAT-J03WMX,DFL-MAT-J03WN1"; F[1]="DFL-MAT-J03WOH,DFL-MAT-J03WOY"; F[2]="DFL-MAT-J03WPY,DFL-MAT-J03WQQ"; F[3]="DFL-MAT-J03WR9"
n=0
for f in $FOLDS; do
  ( $W/run.sh train.py --out $R/${V}_f$f --train $TRAINSETS --w-sc 0.5 --domtok --video-mix --spell-w 0.5 --val-names ${F[$f]} --eval-every 100000 $EXTRA > $R/${V}_f$f.log 2>&1 \
    && $W/run.sh cv_eval.py --ckpt $R/${V}_f$f/model.pt --names ${F[$f]} --out $R/${V}_f$f.npz >> $R/${V}_f$f.log 2>&1 ) &
  n=$((n+1)); if [ $((n % PAR)) -eq 0 ]; then wait; fi
done
wait
echo CV_DONE $V
