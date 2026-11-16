# Retraining

This directory holds the scripts that produced the published weights. Training on a GPU is not
bit-exact (cuDNN, data-loader order), so a rerun gives close but not identical weights. The
published weights are the reference for the submitted file.

All scripts read `MTAH_ROOT`, a directory with this layout:

    $MTAH_ROOT/data/tactical/...            the provided bundle (ball detector only)
    $MTAH_ROOT/official-repo/               clone of metrica-sports/more-than-a-hack-2027 (scorer, used in validation)
    $MTAH_ROOT/external/<dataset>/{raw,common}/
    $MTAH_ROOT/derived/, $MTAH_ROOT/runs/   outputs

## 1. External matches (no account needed)

    export MTAH_ROOT=<dir>
    bash external/download_dfl.sh              # DFL/Sportec, figshare 10.6084/m9.figshare.28196177, CC BY 4.0, 2.6 GB, MD5 checked
    bash external/run_dfl.sh                   # convert 7 matches -> external/dfl/common/*.npz
    bash external/download_metrica.sh          # github.com/metrica-sports/sample-data, 234 MB
    uv run --with numpy --with pandas python external/convert_metrica.py $MTAH_ROOT/external/metrica/raw/sample-data $MTAH_ROOT/external/metrica/common
    bash external/download_skillcorner.sh      # github.com/SkillCorner/opendata @ 4340d27, MIT, 20 matches, 1.8 GB, SHA-256 checked
    uv run --with numpy python external/convert_v3.py $MTAH_ROOT/external/skillcorner/raw/opendata $MTAH_ROOT/external/skillcorner/raw/lfs $MTAH_ROOT/external/skillcorner/common

Each converted match is one `.npz`: `home`, `away`, `ball` (normalised 0–1, NaN = missing),
`label` (0 H, 1 A, 2 D, 3 X), `half`, `home_dir`, `pitch`, `fps`.

## 2. Sequence model (five seeds, about 12 minutes each on one GPU)

    bash seqmodel/train_seeds.sh final_v2 "--degv 2" "2 3 5 6 7" 12.5

This runs, per seed:

    seqmodel/run.sh train.py --out $MTAH_ROOT/runs/seqmodel/final_v2/s<seed> --train dfl metrica sc \
        --w-sc 0.5 --domtok --video-mix --spell-w 0.5 --degv 2 --seed <seed> --eval-every 100000

The sequence model was trained with torch 2.15.0 (CUDA 13.2); `seqmodel/run.sh` pins it.
The checkpoints `final_v2/s{2,3,5,6,7}/model.pt` are `weights/seq_v2_s{2,3,5,6,7}.pt`. The seed
numbers are not consecutive because seeds 0, 1, and 4 were stopped to free shared GPUs; no seed was
dropped for its result. The ensemble averages the five in the order listed in `config.json`.

Cross-validation over the seven DFL matches (four folds, the estimate quoted in TECHNICAL.md):

    bash seqmodel/cv.sh d2 "--degv 2" 12.5 "0 1 2 3" 2
    seqmodel/run.sh cv_report.py --glob "$MTAH_ROOT/runs/seqmodel/cv/d2_f*.npz"

The first candidate (`config_v1.json`) used three seeds of the same architecture trained with
degradation version 1: `--spell-w 0.5 --seed {0,1,2}` without `--degv 2`.

NOTE: `train/seqmodel/` is a snapshot of the research code. It contains options that the final
model does not use (`--bnd`, `--kink`, 25 fps). They were tried and not kept.

## 3. Ball detector (about 30 minutes on one GPU)

Labels come only from algorithms: the provided ball projected through the homography and snapped
to the nearest white blob (round 0), then relabelled once by the round-0 network (round 1).

    cd balldet && export MTAH_SCRATCH=<fast local disk>
    python blob_refine.py $MTAH_ROOT/runs/balldet/blobs.npz                      # round-0 labels: provided ball + white-blob snap
    python make_crops.py $MTAH_SCRATCH/crops 24000 3000                          # 24,000 training frames, 3,000 validation frames (47 GB)
    python train.py --crops $MTAH_SCRATCH/crops --out $MTAH_ROOT/runs/balldet/r0.pt --epochs 8
    python relabel.py --crops $MTAH_SCRATCH/crops --ckpt $MTAH_ROOT/runs/balldet/r0.pt --out $MTAH_ROOT/runs/balldet/labels_r1.npz
    python train.py --crops $MTAH_SCRATCH/crops --out $MTAH_ROOT/runs/balldet/r1.pt \
        --labels $MTAH_ROOT/runs/balldet/labels_r1.npz --init $MTAH_ROOT/runs/balldet/r0.pt --epochs 8 --lr 1e-3

`r1.pt` is `weights/ball_r1.pt`. The two `train.py` lines are copied from the job registry. The
ball detector was trained with torch 2.11.0 (CUDA 12.8), PyAV, and OpenCV; add
`opencv-python-headless` to the environment for these scripts. Validation frames are held-out time
blocks, scored against the pseudo-labels, never against possession labels.

VERIFY: the `blob_refine.py`, `make_crops.py`, and `relabel.py` lines follow the scripts' usage
strings and the reported counts; they were not rerun from this directory.
