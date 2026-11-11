# EmoWOZ extension: workstreams 2 and 3

Native seven-class, in-domain experiments; 66,474 train / 8,509 dev / 8,634 test user targets.
Unlabelled system context is not a target. No Ekman mapping or cross-domain transfer claim.

| Model | Weighted-F1 excl. neutral | Population SD | 95% dialogue CI (seed 0) | Macro-F1 excl. neutral | Derived shift F1 |
|---|---:|---:|---|---:|---:|
| b1 | 76.50 | 0.00 | [74.56, 78.28] | 47.10 | 74.36 |
| b2_bigru | 77.05 | 0.45 | [74.85, 78.12] | 48.24 | 73.93 |
| b2_bilstm | 77.01 | 0.19 | [74.89, 78.52] | 48.73 | 74.12 |

Dev-selected recurrent architecture: **b2_bilstm**. Both architectures are reported regardless of test ranking.

## Protocol

SVM: original word / word+char feature families; C = 0.1, 0.3, 1.0, 3.0; unweighted / balanced loss. Sixteen dev configurations, no logistic-regression search in this SVM workstream.
RNN: reuse the existing word-level BiGRU/BiLSTM, learned embeddings, max 12 epochs, patience 3, train-only vocabulary and inverse-square-root class weights. No architecture or hyperparameter tuning on test.
Five seeds: 0, 1, 2, 3, 4. Checkpoints use best native non-neutral dev weighted-F1. The primary RNN uses mean best-dev across seeds, frozen before reading test targets.
All recognition metrics use the native label list. CIs use 1,000 whole-dialogue resamples for seed 0. Standard deviation describes training-seed variation; intervals do not.
CPU inference efficiency is measured using loaded checkpoints and included per model. Training device/runtime is recorded in manifests; hardware differences limit timing comparisons.

## Interpretation

The two recurrent test means are close (about 0.04 percentage points apart). The reported single-model intervals do not establish a significant difference; no paired significance claim is made. The dev-selected architecture remains BiLSTM even though BiGRU has the slightly higher test mean.
The recurrent models improve the headline score modestly over SVM in this run. Macro-F1 excluding neutral remains around 47-49%, well below non-neutral weighted-F1, indicating uneven performance across emotion classes. Per-class tables should accompany the headline score.
SVM predictions are identical across the five configured seeds; its zero seed SD does not imply zero sampling uncertainty. Its dialogue-bootstrap interval remains nonzero.
First/stable/shift groups have different label compositions. Their F1 values describe these subsets; a higher shift-subset F1 is not proof that an utterance-only model understands dialogue transitions. Derived shift F1 is reported separately.

## Reproduce

```powershell
uv sync --locked --extra rnn
uv run --extra rnn python -X utf8 scripts/train_emowoz_baselines.py --device cpu
uv run --extra rnn python -X utf8 scripts/make_emowoz_baseline_artifacts.py
```

CUDA is optional. With a supported CUDA PyTorch build use its Python interpreter and `--device cuda`; see the runtime manifest. Completed RNN seeds can resume only if source/config/data/runtime signatures match. Epoch-interrupted seeds retrain from their fixed seed.

## Files and attribution

Existing B1/B2 model authors' implementations are reused. This contribution extends native-label probability/checkpoint handling, adds the EmoWOZ protocol, trains the models and prepares audited outputs.
Original MELD/DailyDialog experiment configurations, predictions and reports are untouched. Old checkpoint label defaults remain compatible.
`reports/b1/emowoz/` and `reports/b2/emowoz/` contain dev choices, per-seed scores, confusion/stratum tables, histories and summaries. Predictions retain the original twelve columns first, followed by provenance.
Checkpoints are generated under `models/`; small B2 `.pt` checkpoints can be supplied for the demo. SVM joblib remains ignored like other B1 artifacts.
`notebooks/emowoz_b1_b2.ipynb` is the executed walkthrough. Slides, own-voice narration and personal forms remain separate. AI assistance must be disclosed according to the team guide.
