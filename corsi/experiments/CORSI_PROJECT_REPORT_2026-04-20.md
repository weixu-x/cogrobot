# Corsi Project Report

Date: `2026-04-20`

This report summarizes the current Corsi codebase status, all major completed experiments, and the main code / pipeline changes that were introduced along the way. It is written as a presentation-ready project snapshot.

## 1. Executive Summary

The project now has two clearly separated experiment tracks:

- `coordinate_base`
- `visual_base`

The coordinate track is already solved under the current synthetic setup.

- Best reference run:
  - `coord_baseline_mixed_2_6`
  - full-sequence accuracy: `1.0`
  - estimated span: `6`

The visual track has progressed from a tiny preview pipeline to a stable robosuite-based baseline.

- input: `robosuite freecam` keyframe sequence
- target: block index sequence
- model family: `CNN + encoder-decoder LSTM`

The strongest visual run so far is:

- `freecam_index_v2_20ep`
- trained / resumed to `80` epochs
- best epoch: `75`
- token accuracy: `0.780`
- full-sequence accuracy: `0.69`
- estimated span: `4`

Main interpretation:

- the current visual pipeline is real and learnable
- short and medium sequences are learned well
- long sequences remain the main bottleneck
- increasing model depth did not help
- increasing hidden width only helped under some settings, and did not clearly beat the best baseline

## 2. Repository Organization

The repo was reorganized so that reusable code stays in shared modules, while experiment-facing material is grouped by track.

### Shared reusable code

- `corsi/envs/`
  - sequence generation
  - robosuite Corsi environment helpers
- `corsi/data/`
  - dataset loaders
  - collate functions
- `corsi/models/`
  - coordinate and visual models
- `corsi/training/`
  - shared training runtimes
- `corsi/analysis/`
  - evaluation metrics
- `corsi/scripts/`
  - export, merge, preview, and camera utilities

### Experiment hubs

- `corsi/experiments/coordinate_base/`
- `corsi/experiments/visual_base/`

### Artifact roots

- `corsi_artifacts/coordinate_base/`
- `corsi_artifacts/visual_base/`

This split is important because later visual variants such as delay, heatmap, and timing can be added without mixing them into the reusable baseline code.

## 3. Coordinate Track

### Goal

Create a clean minimal reference task where memory is tested without visual perception noise.

### Task definition

- input features: `(x, y, dx, dy)`
- target: full block sequence
- model: encoder-decoder LSTM

### Main code

- dataset: `corsi/data/coord_dataset.py`
- collate: `corsi/data/collate.py`
- model: `corsi/models/lstm_coord.py`
- training: `corsi/training/train_coord.py`
- metrics: `corsi/analysis/metrics.py`

### Main result

Reference run:

- report: `corsi/experiments/coordinate_base/reports/coord_baseline_mixed_2_6.md`
- artifact root: `corsi_artifacts/coordinate_base/coord_baseline_mixed_2_6`

Best result:

- best epoch: `20`
- token accuracy: `1.0`
- full-sequence accuracy: `1.0`
- estimated span: `6`

Interpretation:

- the coordinate-only pipeline is fully functional
- under the current synthetic setup, this task is too easy to remain the long-term benchmark
- it served as a successful reference before adding visual input

## 4. Visual Track

### Goal

Move from symbolic coordinate input to a more natural embodied visual memory task.

Current official visual task:

- camera: `freecam`
- input: one keyframe per presented block
- target: full block index sequence
- model: per-frame CNN + encoder-decoder LSTM

### Main shared code

- environment / rollout:
  - `corsi/envs/robosuite_corsi.py`
- visual dataset:
  - `corsi/data/robosuite_visual_dataset.py`
- collate:
  - `corsi/data/collate_visual.py`
- model:
  - `corsi/models/lstm_visual.py`
- training:
  - `corsi/training/train_visual.py`
- export:
  - `corsi/scripts/export_robosuite_visual_dataset.py`
- shard merge:
  - `corsi/scripts/merge_robosuite_visual_dataset_shards.py`

### Visual dataset evolution

#### `freecam_index_v1`

First official visual baseline split after the reorganization.

- train: `16`
- val: `4`
- purpose: prove end-to-end train / val wiring

Result:

- best full-sequence accuracy: `0.0`

Interpretation:

- pipeline worked
- dataset was too small to support useful learning

#### `freecam_index_v2`

Scaled visual dataset with manifest-driven train / val roots.

- train: `900`
- val: `100`
- sequence length range: `2-6`
- export mode:
  - sharded generation
  - keyframe-only output
  - no rollout videos by default

Per-length sample counts:

- train:
  - length 2: `181`
  - length 3: `194`
  - length 4: `171`
  - length 5: `179`
  - length 6: `175`
- val:
  - length 2: `18`
  - length 3: `27`
  - length 4: `19`
  - length 5: `18`
  - length 6: `18`

Interpretation:

- long sequences are not underrepresented in `v2`
- difficulty on lengths `5-6` cannot be explained by lack of long-sequence samples alone

## 5. Visual Training Runs

### Run A: `freecam_index_v1_baseline`

- config:
  - `corsi/experiments/visual_base/configs/freecam_index_v1_baseline.json`
- result:
  - best full-sequence accuracy: `0.0`

Role:

- first official end-to-end visual baseline

### Run B: `freecam_index_v2_baseline`

- config:
  - `corsi/experiments/visual_base/configs/freecam_index_v2_baseline.json`
- result:
  - best full-sequence accuracy: `0.22`

Role:

- first scaled `v2` run
- established that the visual pipeline learns non-trivial signal

Important bug discovered during this run:

- variable-length evaluation originally failed when validation batches had different maximum sequence lengths
- fix:
  - pad predictions / targets / masks to a global max length before concatenation in validation
- file:
  - `corsi/training/train_visual.py`

### Run C: `freecam_index_v2_20ep`

This became the main reference visual baseline.

It started as a `20 epoch` run and was then extended with the new checkpoint / resume system to `80 epochs`.

Best result:

- best epoch: `75`
- token accuracy: `0.780`
- full-sequence accuracy: `0.69`
- estimated span: `4`

Length-wise accuracy:

- length 2: `1.0`
- length 3: `0.963`
- length 4: `0.947`
- length 5: `0.333`
- length 6: `0.056`

Interpretation:

- lengths `2-4` are learned very well
- length `5` begins to work but remains much harder
- length `6` is still largely unresolved

This is currently the strongest visual run in the repo.

### Run D: Capacity comparisons without extra regularization

#### `freecam_index_v2_hidden256_80ep`

- hidden size increased from `128` to `256`
- result:
  - best full-sequence accuracy: `0.58`

#### `freecam_index_v2_layers2_80ep`

- number of LSTM layers increased from `1` to `2`
- result:
  - best full-sequence accuracy: `0.48`

Interpretation:

- simply increasing depth hurt performance
- simply increasing width did not beat the best baseline
- deeper recurrent dynamics also produced more repeated-block failures

### Run E: Fair comparison with shared training recipe

To make the structural comparison fairer, three new runs were launched under the same training recipe:

- same dataset
- same task
- same epochs
- same batch size
- same seed
- same `dropout=0.1`
- same `learning_rate=5e-4`

Runs:

- `freecam_index_v2_baseline_drop01_lr5e4_80ep`
- `freecam_index_v2_hidden256_drop01_lr5e4_80ep`
- `freecam_index_v2_layers2_drop01_lr5e4_80ep`

Results:

#### baseline + dropout/lr

- best epoch: `72`
- best full-sequence accuracy: `0.68`
- best token accuracy: `0.783`
- best val loss: `0.409`

#### hidden256 + dropout/lr

- best epoch: `70`
- best full-sequence accuracy: `0.68`
- best token accuracy: `0.826`
- best val loss: `0.575`

#### layers2 + dropout/lr

- best epoch: `78`
- best full-sequence accuracy: `0.37`
- best token accuracy: `0.634`
- best val loss: `1.085`

Interpretation:

- with a shared training recipe, `baseline` and `hidden256` tie on full-sequence accuracy
- `hidden256` reaches higher token accuracy but worse validation loss
- `layers2` remains clearly inferior

## 6. Model and Training Analysis

### 6.1 What does `hidden_dim=128` mean?

Current visual model:

- CNN frame encoder
- encoder LSTM
- decoder LSTM
- output head

`hidden_dim=128` applies to:

- encoder LSTM hidden size
- decoder LSTM hidden size
- output head input size

The CNN mainly compresses each frame into a feature vector:

- `cnn_feature_dim=128`

So there are two different `128`s in the model:

- one for frame features
- one for recurrent state size

### 6.2 Decoder error accumulation

The decoder predicts one token at a time.

During greedy decoding:

- predict next block
- feed that prediction into the next step
- continue autoregressively

This means an early mistake can corrupt later predictions.

Why this matters:

- short sequences have fewer opportunities for error propagation
- long sequences are more vulnerable to compounding decoder mistakes

This is one plausible reason why lengths `5-6` remain difficult even when lengths `2-4` are solved well.

### 6.3 Train-val gap

Yes, there is a visible train-val gap in the strong visual runs.

Example:

- `freecam_index_v2_20ep`
  - best epoch `75`
  - train loss `0.001044`
  - val loss `0.466115`
  - gap `~0.465`

The same pattern is stronger in larger models:

- `hidden256_80ep`
  - best gap `~0.789`
- `layers2_80ep`
  - best gap `~1.029`

Interpretation:

- the model fits the training set extremely strongly
- validation performance still improves, so this is not a total failure to generalize
- bigger models widen the gap under the current data and optimizer settings

This strongly suggests:

- the current bottleneck is not only capacity
- optimization and generalization matter as much as raw size

### 6.4 Training curve evidence

Representative points from the strongest baseline `freecam_index_v2_20ep`:

- epoch 1:
  - train loss `2.183`
  - val loss `2.163`
  - full-seq acc `0.00`
- epoch 10:
  - train loss `1.188`
  - val loss `1.315`
  - full-seq acc `0.17`
- epoch 20:
  - train loss `0.181`
  - val loss `0.630`
  - full-seq acc `0.51`
- epoch 40:
  - train loss `0.028`
  - val loss `0.526`
  - full-seq acc `0.58`
- epoch 75:
  - train loss `0.0010`
  - val loss `0.466`
  - full-seq acc `0.69`
- epoch 80:
  - train loss `0.00085`
  - val loss `0.469`
  - full-seq acc `0.67`

Interpretation:

- continued training remains useful up to late epochs
- gains slow down over time
- strong fitting appears long before long-sequence performance saturates

## 7. Major Code / Pipeline Changes

Below is the main list of meaningful changes made during this phase.

### Project structure and experiment organization

- reorganized experiment hubs into:
  - `coordinate_base`
  - `visual_base`
- grouped artifacts under:
  - `corsi_artifacts/coordinate_base`
  - `corsi_artifacts/visual_base`

### Robosuite camera and layout work

- corrected the Corsi board placement in robosuite
- aligned the board and frame with the intended `freecam` view
- added tools to inspect and tune camera parameters

### Visual dataset pipeline

- added robosuite visual dataset export
- added manifest-driven dataset roots
- added `freecam` export path
- added sharded dataset generation
- added shard merge utility
- added keyframe-only export mode without rollout video writing

### Visual training pipeline

- added dataset-info export into the training output directory
- fixed variable-length validation aggregation
- added full checkpoint resume support with:
  - `last_checkpoint.pt`
  - `best_model.pt`
  - RNG state save / restore
  - `--resume`
  - `--init_model`
  - `--checkpoint_dir`
  - `--save_rng_state`
  - `--strict_resume`
  - `--auto_resume`
- added worker seeding and DataLoader generator seeding

### Documentation

- added:
  - `VERSION_NOTES.md`
  - `freecam_index_v1_baseline.md`
  - `freecam_index_v2_baseline.md`
- updated:
  - `corsi/README.md`
  - `corsi/WORKLOG.md`

## 8. Current Best Answers

### What works well now?

- coordinate baseline is solved
- visual baseline is stable and meaningfully trainable
- `freecam + keyframe + index` is now a valid baseline task
- checkpoint resume works
- longer training improves visual performance significantly

### What does not work well now?

- long sequences remain much harder than short ones
- deeper LSTM (`2 layers`) is not helping
- simply making the hidden state larger does not automatically beat the best baseline

### Best current visual reference

Use:

- `freecam_index_v2_20ep`

Reason:

- strongest full-sequence accuracy seen so far
- strongest overall reference point before moving to the next variant

## 9. Recommended Next Step

The next stage should be:

1. keep the task definition fixed
2. use the strongest current visual baseline as the main reference
3. investigate why lengths `5-6` remain difficult
4. only then move to the next branch of experiments

Likely next research directions:

- analyze failure cases by sequence length and error type
- inspect whether errors are mostly order drift, block confusion, or decoder collapse
- test better generalization strategies rather than only larger models
- then move to:
  - delay / blank interval
  - heatmap output
  - timing-expanded representations

## 10. Suggested PPT Flow

This report can be turned into a talk with this structure:

1. Motivation
   - from symbolic Corsi to embodied visual memory
2. Repository and experiment organization
3. Coordinate baseline
   - solved reference task
4. Visual pipeline construction
   - robosuite camera, dataset, CNN+LSTM
5. Dataset evolution
   - `v1` to `v2`
6. Main visual results
   - strongest baseline curve
   - accuracy by length
7. Comparative modeling experiments
   - width vs depth
   - regularized fair comparison
8. Analysis
   - train-val gap
   - decoder error accumulation
   - long-sequence bottleneck
9. Next steps
   - explainable diagnosis first
   - task extensions after baseline understanding
