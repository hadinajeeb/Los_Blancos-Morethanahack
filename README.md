# <team_name>: frame-level ball possession

Entry for the More than a Hack 2027 Initial Challenge. This repository turns the provided data
bundle into the submission file: one label (`H`, `A`, `D`, or `X`) for each of the 199,319 frames
of the tactical video.

- Method and limits: [TECHNICAL.md](TECHNICAL.md).
- Status: draft, not published. Last run end to end on 2026-11-15.
- Default output: the video-ball file of the five-seed ensemble, float32 ball detector (table under Run).

## What it does

| Stage | Input | Output | Code |
|---|---|---|---|
| 1 `prep` | ATD text, ATD XML, homography CSV | player, ball, and homography arrays | `mtah/prep.py` |
| 2 `ball_infer` | tactical video, ball-detector weights | top four heatmap peaks per frame | `mtah/ball_infer.py` |
| 3 `ball_link` | peaks, homography | one ball track on the pitch | `mtah/ball_link.py` |
| 4 `seq` | players, ball track, five sequence-model checkpoints | P(H), P(A), P(D) per frame | `mtah/seq.py` |
| 5 `decode` | posterior | labels and the CSV | `mtah/decode.py` |

## Install

Requirements: Linux, Python 3.12, [uv](https://docs.astral.sh/uv/), and an NVIDIA GPU with 12 GB
or more for the default configuration (about 40 minutes end to end on one GPU).

    git clone <this repository> && cd <this repository>
    uv sync --frozen                      # installs the pinned versions in uv.lock

Dependencies are pinned in `pyproject.toml` and `uv.lock`: `torch==2.11.0+cu128` (from the PyTorch
package index), `numpy==2.5.4`, `av==19.0.2`.

### Weights

The weights are not in git. Download the files of release `v2` into `weights/` and check them:

    cd weights && sha256sum -c MANIFEST.sha256

| File | Size | What |
|---|---|---|
| `ball_r1.pt` | 13 MB | ball heatmap network, trained on the match video from algorithmic pseudo-labels |
| `seq_v2_s2.pt`, `_s3`, `_s5`, `_s6`, `_s7` | 14 MB each | sequence model, five seeds, trained on other matches only |
| `seq_v1_s0.pt`, `_s1`, `_s2` | 14 MB each | earlier three-seed ensemble; only `config_v1.json` uses it |

OPEN: the release assets are not uploaded. The files are staged, with this manifest, on the
team's storage.

## Run

Put the data bundle anywhere, with the layout of the organisers' `data/README.md`
(`<data>/tactical/BAR-ATH_tactical.mp4` and the other tactical files). Then:

    uv run python run.py --data <data> --out <team_name>.csv

Then validate with the organisers' script:

    python <organisers' repository>/evaluation/validate_submission.py <team_name>.csv

`run.py` prints the SHA-256 of the peaks, the ball track, the posterior, and the CSV, and writes
them to `<work>/report_<config>_<ball source>.json`. Expected SHA-256 of the CSV:

| Command | Ball source | CSV SHA-256 |
|---|---|---|
| default (the submission) | video ball first | `1d10a7a198d9bc76753269c8d53b8417510c0324fafe4848556dafb64c76ca2a` |
| `--ball-source avg` | mean of the two posteriors | `615f8fbc1734e85600a2d2d4100ff8022aa1c4997e4b5e66e484fc3bb614c086` |
| `--ball-source provided` | provided ball only; needs no video and no GPU | `804f740bb06383baf6903d4b0b99cf9fc47c2a5ecf6cf734416ea18ac17a9c42` |
| `--config config_bf16.json` | video ball first, bfloat16 detector | `229b7434709a0ea36b28634c2b018b84e4f59da3bc67c06fdf0b1076a01dc0f9` |
| `--config config_bf16.json --ball-source avg` | mean, bfloat16 detector | `2c14c0776fb80460e7752cdc5a9f0c910a16422087e183166fb9bd81f6e3be14` |

Other options: `--device cpu` (sequence model and float32 ball detector only), `--work <dir>`
(cache directory, default `./work`). Each stage skips work that is already cached. Delete the
directory to recompute.

### The two ball-detector modes

- **Default (`config.json`, float32).** This produces the submitted file. It gave the same file on
  GPU and CPU in every test we ran, and across PyTorch 2.11 and 2.15.
- **bfloat16 (`config_bf16.json`).** About three times faster. It needs a CUDA GPU, and its result
  depends on the PyTorch build. Its file differs from the default file on 84 of 199,319 labels.

TECHNICAL.md, section 7, lists every comparison.

### Expected runtime

Measured with one NVIDIA RTX PRO 6000 Blackwell (96 GB) and an AMD EPYC 9655:

| Stage | GPU | CPU only |
|---|---|---|
| `prep` | 7 s | 7 s |
| `ball_infer`, bfloat16, 8 processes | 12 min | not possible |
| `ball_infer`, float32, 8 processes | 33 min to 36 min | about 300 core-hours (estimated from 8 of 80 chunks) |
| `ball_link` | 2 s | 2 s |
| `seq`, five checkpoints, per ball source | 14 s to 33 s | about 4 min |
| `decode` | under 1 s | under 1 s |

## Retraining

Training is documented in [train/README.md](train/README.md). GPU training is not bit-exact, so a
retrained model gives slightly different weights and can change some labels. The published weights
are the reference for the submitted file.

## Configuration

| File | Use |
|---|---|
| `config.json` | the submission: five-seed ensemble, video ball first, float32 ball detector, kick-off formation rule |
| `config_bf16.json` | the same with a bfloat16 ball detector (faster; less portable) |
| `config_v1.json` | the earlier three-seed ensemble with kick-off positions from `games.py` |

`ball_source` is `video`, `provided`, or `avg`. To use another ensemble, change `seq_ckpts`.

## Data and licences

- No match data is in this repository, and none must be added. `.gitignore` excludes `data/`,
  work directories, arrays, and CSVs.
- `mtah/third_party/games.py` is the organisers' file (MIT, licence text beside it). It supplies
  the graded frame ranges and the kick-off anchors.
- Training data from other matches: DFL/Sportec open dataset (CC BY 4.0), Metrica Sports
  sample-data, SkillCorner opendata (MIT). Details in TECHNICAL.md, section 3.
