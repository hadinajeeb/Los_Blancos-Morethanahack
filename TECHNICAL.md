# Frame-level ball possession from noisy tracking: technical document

| | |
|---|---|
| Team | `<team_name>` |
| Challenge | More than a Hack 2027, Initial Challenge |
| Applies to | this repository at the submitted commit, weights release `v2` |
| Status | DRAFT. Last verified end to end on 2026-11-15 |

## 1. Summary

We label each frame with a neural sequence model that reads player positions and the ball
position, then decode its per-frame probabilities into possessions with a Viterbi pass. The model
was trained only on other matches. We replace the provided ball track with our own, from a ball
detector that we trained on the tactical video without human labels.

| Step | What it does |
|---|---|
| Ball detector | Finds the ball in the tactical video. Trained on this match's video with labels made by an algorithm. |
| Sequence model | Outputs P(home), P(away), P(dead ball) per frame. Five seeds, trained on 30 matches from open datasets. |
| Decoder | Turns probabilities into runs of `H`, `A`, and `D` with a fixed cost for each change of label. |
| Kick-off rule | Marks frames as `D` while the teams stand in kick-off formation, and before a half's first kick-off. |

In cross-validation over seven matches from another dataset, degraded to resemble the provided
tracking, the model scores spellF1 0.71. The organisers' nearest-player baseline logic scores 0.31
on the same kind of input. Section 5 states how far these numbers can be trusted. We expect the
score on the provided match to be lower.

## 2. Approach, and why

### 2.1 What the problem needs

We measured how spellF1 reacts to each kind of error by perturbing a ground-truth file against
itself. Three results shaped the design:

- A short possession counts as much as a long one. A method that smooths away possessions under
  2 s loses many spells.
- Label flicker is the most expensive error. Twenty wrong runs of five frames, spread over five
  minutes, cut a perfect score to 0.39.
- The exact frame where a dead ball starts or ends costs almost nothing. Whether a stoppage
  separates two possessions matters.

So the method must find every change of team, including short ones, and must not flicker. A
per-frame classifier followed by a majority vote cannot do both. A sequence model with a decoder
can.

### 2.2 Why a learned model

We built a rule-based labeller first: touch detection from ball kinematics, last-touch possession,
and dead-ball logic from ball gaps and player speed. We tuned it on other matches. On
degraded matches from another dataset it scores 0.46, against 0.70 for the learned model. Its errors were team
attribution during contested play and missed short possessions. Those are pattern-recognition
problems, and labelled data from other matches exists, so we trained a model.

### 2.3 The sequence model

- **Input per frame.** One token per tracked player: team, position, velocity, and offset to the
  ball. Thirty frame-level values: ball position, speed and acceleration, distance of the nearest
  players of each team, team centroids, and the time since and until the ball was seen.
- **No identities.** Player tokens pass through a shared network and are pooled per team. The
  provided tracking has no player identities and its tracks break often, so the model never
  depends on slot order.
- **Temporal part.** Fourteen dilated 1-D convolutions (about ±40 s of context at 12.5 fps), then
  a bidirectional GRU. 3.6 million parameters.
- **View.** Metres, with the pitch rotated so that the home team always attacks the same way.
  At test time we average over team swap and left-right flip.
- **Loss.** Cross-entropy plus a smoothing term. Frames of short possessions get a higher weight,
  because the metric counts each possession once.
- **Ensemble.** Five seeds, averaged.

### 2.4 Making other matches look like this one

The open datasets have clean tracking. The provided tracking does not. Before training, each
external match passes through a random degradation that we set from **input** statistics of the
provided tracking (ball-gap lengths, players tracked per frame, camera coverage). It applies:

- long ball dropouts, placed around stoppages and also in live play, off the pitch, and in crowds;
- ball smoothing, position noise, large error bursts, and false ball tracks;
- a panning camera window that hides players, broken tracks, phantom tracks, and goalkeepers held
  in place while off screen.

No statistic of the public ground truth entered the degradation. We compared eight input
statistics between the provided tracking and the degraded matches and tightened the ranges until
they agreed (for example players tracked per team, and ball gaps per 100 minutes).

Half of the training versions imitate the provided ball. The other half imitate our video ball,
which is unsmoothed, noisier per frame, and visible through most stoppages.

One correction was needed before any of this. In the DFL data the ball is frozen on every
dead-ball frame, and in the Metrica samples it is missing on every dead-ball frame. Either would
let a model read the label from the ball. We re-simulate the ball on all dead-ball frames of the
external matches, so the shortcut does not exist in training or in validation.

### 2.5 Our own ball track

The provided ball is missing on 41% of graded frames, and where present it is smoothed over about
20 frames. A smoothed ball hides the direction changes that mark a touch. So we detect the ball in
the tactical video:

1. Project the provided ball through the homography to pixels. Snap it to the nearest small white
   blob. This gives a first set of labels with no person involved.
2. Train a small heatmap network (3.3 million parameters, no pretrained weights) on stacks of three
   frames.
3. Let that network relabel the training windows once, and retrain.
4. Run it on every graded frame. Keep the four strongest peaks per frame. The network runs in float32;
   section 7 states what that means for reproduction.
5. Link the peaks into one track with a Viterbi pass that rejects jumps over 2.5 m per frame.
   Project the track to the pitch through the inverse homography.

The video track gives a ball position on about 80% of graded frames. Where it has none, we use the
provided ball. For a ball in the air the projected pitch position is wrong by metres; we did not
estimate height.

### 2.6 Decoding and the kick-off rule

- The decoder takes the mean probability of each pair of frames and finds the best path over
  `H`, `A`, and `D`, with a cost of 4 (in log-probability units) for every change of label. On
  held-out external matches the score changed by less than 0.01 over the whole range of costs we
  tried.
- **Kick-off rule.** Teams stand entirely in their own halves only while a kick-off is pending.
  The rule finds stretches of 0.5 s or more with at least eight tracked players per team, all on
  their own side. Those frames are `D`, and so are graded frames before a half's first such
  stretch ends. The rule uses no frame number and no metadata of the match. It exists because no
  training match contains a graded stretch before a kick-off.
- Frames outside the graded halves are `X`. The graded ranges come from the organisers' `games.py`.

## 3. Training data

No label, human or otherwise, from the provided match was used to train, tune, or select the
sequence model. All labelled data is from other matches.

| Dataset | Matches | Source | Licence | Who made the labels |
|---|---|---|---|---|
| DFL / Sportec open dataset (Bassek et al. 2025) | 7, 25 Hz, about 1.0 million frames | figshare, doi 10.6084/m9.figshare.28196177 | CC BY 4.0 | The data provider. Each frame carries `BallStatus` and `BallPossession`. We used them unchanged. |
| Metrica Sports sample-data | 3, 25 Hz, about 0.43 million frames | github.com/metrica-sports/sample-data | no formal licence; attribution requested | We derived them by rule from Metrica's event list and from the stretches where the ball is missing. |
| SkillCorner opendata, 2024/25 A-League | 20, 10 Hz | github.com/SkillCorner/opendata | MIT | The data provider's automatic products: phases of play give the team, and frames outside any phase are dead ball. |

- We labelled nothing by hand on any match.
- The Metrica owner labels are our reading of the events. Nothing verifies them.
- The SkillCorner labels follow a different convention (longer possessions). The model takes the
  label source as an input, and at test time we ask for the DFL convention (section 4). SkillCorner
  frames with no tracked player carry no loss; that covers every SkillCorner frame with no ball.
- The ball detector was trained on the provided match's video. Its labels came from the provided
  tracking and from the detector's own first round, with no human decision. RULES.md allows
  pseudo-labels produced by an algorithm.

## 4. Possession rules implemented

The model learned its rules from labels, so this section describes the convention it was asked to
follow.

- **Possession follows the last team to touch the ball.** A single tackle or block by the other
  team changes possession, even if the first team has the ball back within a second. This is the
  DFL convention. We chose it because the public five minutes of ground truth show changes at
  single touches. CHALLENGE.md names that window as the reference for the definition.
- **A loose or contested ball stays with the last team.**
- **Dead ball.** The ball is dead from the referee's stoppage or from the ball leaving the pitch
  until shortly after the restart kick. A ball that is still visible and tracked can be dead.
- **Restarts.** The first possession after a stoppage belongs to the team that restarts. The model
  infers this from positions; we do not read the set-piece tags.
- **Missing ball.** The model sees how long the ball has been missing and where it was last and
  next seen. It can keep a possession alive through a gap, or call the gap dead, from the players'
  movement.

RISK: CHALLENGE.md describes possession as *controlled* possession. If the full ground truth
waits for control more often than the public window shows, our file has too many short
possessions.

## 5. Validation

All choices were made on other matches. The public five minutes were used as a correctness check
only: frame alignment, file format, and the meaning of the labels.

**Cross-validation.** Four folds over the seven DFL matches. Each fold model trains on the other
DFL matches plus Metrica and SkillCorner, and is scored on its held-out matches after degradation.
Scores are the official spellF1 at the 25 fps label rate, mean ± standard deviation over matches.

| Test input | Sequence model | Rule model | Nearest-player baseline logic |
|---|---|---|---|
| Degraded like the provided ball | 0.705 ± 0.028 | — | — |
| Degraded like our video ball | 0.712 ± 0.042 | — | — |
| The rule workstream's own degradation | 0.697 ± 0.026 | 0.455 | 0.306 |

**Ball source.** A separate test on the same seven held-out matches compared the two ball inputs
with matched noise: video ball first, 0.741 ± 0.029; provided ball, 0.699 ± 0.031. The video ball
was better on all seven matches. The mean of the two posteriors scored 0.740 ± 0.032. We submit
the video-ball file.

Limits of these numbers:

- Selection and reporting share matches in part. On the two matches that no design choice ever
  saw, the three rows of the first table are 0.704, 0.729, and 0.694.
- The rule model's 0.455 is a mean over seven DFL matches that it was tuned on.
- The simulated noise is easier than the real tracking. The nearest-player baseline logic scores
  0.31 on our degraded matches and 0.24 on the provided match (the organisers' published figure).
- The model is less certain on the provided match than on held-out halves. With the video ball,
  its mean posterior entropy is 0.172, at the top of the held-out range (0.11–0.17). A line fitted
  across held-out halves puts the expected score on the provided match near 0.68, ± 0.07. That is
  an extrapolation, not a measurement.
- Possessions under 1 s are found 13% of the time on degraded held-out matches, and possessions
  of 1 s to 2 s 47% of the time.

## 6. Assets: used, and tried and dropped

| Asset or idea | Status | Reason |
|---|---|---|
| Tactical tracking (players) | used | main input |
| Tactical tracking (ball) | used as a fallback | where the video ball has no position |
| Tactical video | used | ball detector |
| Tactical homography | used | pseudo-labels for the detector; pitch position of detections |
| Smart Tagging | dropped | It fed the rule model (phase team as a prior, set pieces as restarts). The learned model did not need it, and no external dataset has it, so it could not be validated there. |
| Broadcast tracking | dropped | No mapping to the tactical video is provided. We found one by cross-correlation. The broadcast ball then added a position on only about 5% of graded frames. |
| Broadcast camera cut-aways as a stoppage cue | dropped | The cue lags the stoppage by 2–4 s, needs the cross-video offsets, and cannot be validated on other matches. |
| Audio: referee whistle | dropped | A tone detector finds whistles. It cannot be validated on other matches, and a ball out of play gets no whistle. |
| Rule-based labeller | dropped | 0.46 against 0.70 for the learned model on degraded held-out matches. |
| Unsupervised HMM fitted on the match | dropped | Worse than the rule-based labeller on every external set. |
| Fusing rule evidence into the learned posterior | dropped | Best of 24 variants gained 0.005 (standard deviation 0.008) on held-out matches. |
| Boundary-detection head, ball velocity-change inputs, 25 fps | dropped | Three attempts to find more short possessions. None gained more than 0.003 in cross-validation; 25 fps was worse. |
| Kick-off positions from `games.py` as a dead-ball rule | replaced | The kick-off formation rule does the same from the tracking alone. |
| Self-training the sequence model on the provided match | dropped | Gained 0.008 in a test on other matches, which is inside the noise. |
| Averaging the video-ball and provided-ball posteriors | alternate | Ties with the video ball in the external test (0.740 against 0.741). Available as `--ball-source avg`. |
| Provided ball only | alternate | 0.699 against 0.741 in the external test. Available as `--ball-source provided`; needs no video. |

## 7. Reproduction

    uv sync --frozen
    (download the six weight files into weights/, then: cd weights && sha256sum -c MANIFEST.sha256)
    uv run python run.py --data <data bundle> --out <team_name>.csv

README.md has the options and runtimes. `train/README.md` has the training commands.

### 7.1 What reproduces exactly

Training on a GPU is not bit-exact, so the published weights are the reference. Given the weights,
we measured the following on NVIDIA RTX PRO 6000 Blackwell GPUs and AMD EPYC 9655 CPUs.

**Default configuration** (`config.json`: ball detector in float32). This writes the submitted
file. It differs from the bfloat16 file on 84 of 199,319 labels, in 11 places.

| Comparison | Result |
|---|---|
| Three from-scratch GPU runs, two machines (PyTorch 2.15.0) | every stage bit-identical; CSV byte-identical |
| From-scratch GPU run under PyTorch 2.11.0+cu128 against those runs | peaks differ by at most 1.1e-6 in confidence; CSV byte-identical |
| Ball detector on CPU for 8 of the 80 video chunks (16,000 frames), rest of the pipeline on CPU | peaks differ by at most 1.6e-6 in confidence and 1.2e-4 px; ball track equal within 1 mm; CSV byte-identical to the GPU file |
| Random perturbation of every peak, at the size of the CPU–GPU difference and at 10 times that size | 0 labels changed in 9 trials |
| Random perturbation at 100 times that size | 10, 10, and 0 labels changed in 3 trials |

We did not run the float32 ball detector on a CPU for the whole video (about 300 core-hours), and
we did not test another GPU model.

**bfloat16 configuration** (`config_bf16.json`: ball detector in bfloat16, PyTorch 2.11.0+cu128). This is not the submitted file:

| Comparison | Result |
|---|---|
| Three from-scratch runs in the pinned environment, two machines, one from a fresh `uv sync --frozen` | every stage bit-identical; CSV byte-identical |
| Against the file we first produced with the research code | every stage identical: the 80 chunks of ball peaks and the ball track bit for bit, and the CSV byte for byte |
| Sequence model and decode on CPU instead of GPU, same ball track | CSV byte-identical for all three ball sources |
| Same code under PyTorch 2.15.0+cu132 | ball track differs by more than 1 mm on 453 frames; 2 of 199,319 labels differ |

The bfloat16 ball detector cannot run on a CPU, and we did not test it on another GPU model. Its
output depends on the PyTorch build. A few labels are likely to differ on other hardware, because
2 differed under another PyTorch build on the same hardware.

### 7.2 What makes the file stable

- Video is decoded on the CPU by FFmpeg through PyAV, and the frame count is checked.
- Linking, the kick-off rule, and the Viterbi decode are numpy on the CPU. Ties resolve to the
  lowest index.
- The decode is insensitive to small numeric differences: independent noise of 1e-4 on every
  posterior value changed no label in 5 trials.
- Checkpoints are averaged in a fixed order. TF32 is off and cuDNN runs in deterministic mode.

## 8. Limitations, and what we would do next

- **Short possessions.** Most possessions under 1 s are missed, and half of those between 1 s and
  2 s. A boundary head, extra ball-kinematics inputs, and 25 fps did not help. Under this noise the
  information is mostly not in the tracking; a ball track with height would be the next step.
- **Convention risk.** See section 4. A ruling from the organisers on "controlled" would settle it.
- **Dead ball with the ball in view.** The model must infer a whistle stoppage from player
  movement. The referee's whistle in the audio marks these moments, and we did not use it.
- **Airborne ball.** Its pitch position is wrong, so touches during aerial play are less visible.
- **Label supply.** Only seven matches carry labels in the convention we target. More would help
  most.
- **Noise model.** Our degradation is built from input statistics. Real tracking errors cluster in
  duels, and ours do not.

## 9. Development tooling

We wrote and ran the code with an LLM coding assistant. No LLM is part of the pipeline, and no
label in the submission was produced or edited by a person or by an LLM.
