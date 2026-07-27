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
