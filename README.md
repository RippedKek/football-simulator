# Part 1 — Data pipeline (Stages 0, 5, 6)

Owner of everything between the raw Metrica files and the tensors the models
train on. Nothing here uses a neural network; it is parsing, measurement and
feature engineering.

## Install

To use this folder, copy its contents into the project root, merging with what
is already there:

```
pip install numpy pandas pyarrow scipy
```

## What is in here

| file | what it does |
|---|---|
| `src/stage0_gsr/load_metrica.py` | reads the raw Metrica files (CSV pairs for games 1-2, FIFA-EPTS text + XML for game 3) and writes one tidy `tracks.parquet` per match |
| `src/stage_style/descriptors.py` | measures the four style knobs (line height, press, width, tempo) from tracking, per 60-second chunk |
| `src/stage5_storage/common.py` | velocity smoothing, speed clamping, long-format rows to dense arrays |
| `src/stage5_storage/run_storage_metrica.py` | cuts the tracking into 6-second training windows, attaches each window's style, does the train/val split |
| `src/stage6_features/relational.py` | the 24-dimension per-agent relational feature vector |
| `src/stage6_features/run_features_gsr.py` | applies that encoder to every window, adds the ball row, writes the feature tensors |

## Run it

Run all three from the `src` directory.

```bash
python -m stage0_gsr.load_metrica --all
```

```bash
python -m stage5_storage.run_storage_metrica
```

```bash
python -m stage6_features.run_features_gsr --store ../output/storage/metrica --out ../output/features/metrica
```

## What it produces

```
output/metrica/game{1,2,3}/tracks.parquet      tidy tracking, one row per agent per frame
output/storage/metrica/windows_{train,val}.npz 13,725 train / 3,402 val windows
output/storage/metrica/style_chunks.json       the measured style knobs per chunk
output/features/metrica/features_{train,val}.npz  (N, 30, 23, 24) relational features
```

Part 2 trains on the last two.

## Things worth knowing (they cost debugging time)

- **Goalkeeper detection** uses the mean of the per-row `|x - 0.5|`, not
  `|mean(x) - 0.5|`. Teams swap ends at half time, so a keeper's mean x over a
  full match lands near the centre of the pitch and the naive test finds nobody.
- **Segments are halves.** No window may cross one, because the ends swap.
- **Target deltas are physics-capped** (`cap_deltas`). The ball is interpolated
  while out of play, which produces jumps of tens of metres in a single step;
  uncapped, those few frames dominate the squared loss.
- **The validation split is the last 20% of each segment**, and windows that
  straddle the boundary are dropped. Windows overlap, so without that drop the
  training frames leak into validation.



# Part 2 — Learned models (Stage 7)

Owner of both neural networks: the architectures, the training loops, the
losses. This is the pattern-recognition core of the project.

## Install

Copy the contents of this folder into the project root, merging with what is
already there.

```
pip install numpy pandas pyarrow torch
```

Needs Part 1's output (`output/storage/metrica`, `output/features/metrica`) and
the raw event CSVs under `data/metrica`.

## What is in here

### StyleNet — how the team moves

| file | what it does |
|---|---|
| `src/stage7_style/model_gsr.py` | the architecture: encoder, FiLM conditioning, three factored spatial/temporal transformer blocks, trajectory + style heads |
| `src/stage7_style/features_torch.py` | the Stage-6 relational encoder rewritten in torch so features can be rebuilt from the model's own predictions inside the training loop |
| `src/stage7_style/train_gsr.py` | training: trajectory MSE, style-consistency loss, scheduled-sampling rollout, physics priors |

FiLM is the part worth explaining: the four style knobs go through a small MLP
that outputs a scale and a shift for every channel of every agent embedding
(`h = h * (1 + gamma) + beta`). One 4-number vector therefore re-weights the
whole network instead of being four extra inputs it can ignore. The
style-consistency head, which has to recover the knobs from the generated
motion, is what forces it to actually listen.

### KickNet — what the player on the ball decides

| file | what it does |
|---|---|
| `src/stage7_style/build_kick_data.py` | turns the event CSVs into training samples: passes with their receiver, shots, and "hold" negatives from the frames just before each kick |
| `src/stage7_style/kick_model.py` | the pointer network: a shared MLP scores every teammate, a pooled head predicts hold / pass / shot |
| `src/stage7_style/train_kick.py` | training with class-weighted cross-entropy (shots are rare) plus cross-entropy over receivers |

The receiver is chosen by scoring the teammates who are actually on the pitch
rather than by picking from fixed classes, so substitutions and formation
changes do not break it. All geometry is expressed in the kicker's attacking
direction, which lets one model serve both teams and both halves.

## Run it

From the `src` directory.

```bash
python -m stage7_style.train_gsr --rollout --feat ../output/features/metrica --store ../output/storage/metrica --out ../output/training/metrica
```

```bash
python -m stage7_style.build_kick_data
```

```bash
python -m stage7_style.train_kick
```

## What it produces

```
output/training/metrica/style_model.pt   StyleNet checkpoint (config travels inside it)
output/training/metrica/train_log.json   per-epoch losses, for the training curve figure
output/kick/kick_data.npz                the KickNet dataset
output/kick/kick_model.pt                KickNet checkpoint
output/kick/kick_meta.json               includes the fitted pass-speed relation
```

Part 3 loads the two checkpoints.

## Numbers to quote

- StyleNet validation trajectory loss 1.91e-05 against 3.26e-05 on training, so
  it generalises rather than memorising.
- Free-running for 60 seconds with no reseeding, the players keep moving
  (median 0.63 m/s), the ball behaves (median 2.6 m/s), and the team spread
  grows from 17 m to 30 m. An earlier version collapsed everyone into a knot;
  that failure is gone.
- KickNet: receiver top-3 accuracy 82% against a 10-way choice, shot recall
  0.89. Hold against pass sits near chance, which is expected, because the two
  states look almost identical one frame apart. The simulator handles the
  timing with a release gate instead of asking the model to decide it.
