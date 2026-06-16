# Visual Retention Delay

This baseline keeps the existing `CNN frame encoder -> encoder LSTM -> decoder LSTM` architecture and only adds a post-sequence retention manipulation between observation encoding and autoregressive recall.

## Delay Modes

- `hold_state`: the encoder processes the observed frames once, then its final `(h, c)` is passed directly into the decoder. The delay is logged as an experimental condition, but the model state does not evolve during the delay.
- `encoder_blanks`: the encoder processes the observed frames, then continues for `delay_steps` extra recurrent steps using blank inputs. This changes encoder time unfolding while keeping the recall target unchanged.

## Why Target Length Stays Fixed

- `target_length` is always the block sequence length and stays in the original `2..6` range.
- `frame_lengths` and `encoder_time_length` refer to encoder-side time only.
- In `encoder_blanks`, encoder time increases to `original_frame_length + delay_steps`, but the decoder still recalls the same block list.

## Why `encoder_blanks` Is The Retention Manipulation

`hold_state` is useful as a bookkeeping control because it tags trials with a delay condition without introducing recurrent state drift. `encoder_blanks` is closer to a real retention manipulation because the encoder hidden state evolves over blank time before recall starts.

## New Logs And Analysis Outputs

- `epoch_logs.jsonl` now records `delay_mode`, `delay_steps`, `hidden_dim`, per-length accuracy, serial-position accuracy, and aggregate error analysis.
- `results_summary.json` stores the best epoch, best token/full-sequence accuracy, best per-length metrics, delay condition, hidden dimension, and checkpoint path.
- When `--return-hidden-traces` is enabled, validation saves sampled traces under `analysis/hidden_traces/epoch_XXX/trace_samples.pt` with:
  - observation-end encoder hidden state
  - optional delay hidden trace
  - recall-start hidden state
  - decoder hidden trace

## Example Commands

```bash
python corsi/train_visual.py --delay-mode none --delay-steps 0
python corsi/train_visual.py --delay-mode hold_state --delay-steps 2
python corsi/train_visual.py --delay-mode encoder_blanks --delay-steps 2
python corsi/train_visual.py --delay-mode encoder_blanks --delay-steps 8
```
