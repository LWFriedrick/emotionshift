# Native EmoWOZ extension for workstreams 2 and 3

This contribution extends the team's existing utterance-only TF-IDF/SVM,
BiGRU and BiLSTM implementations to EmoWOZ. Original model authors and the
MELD/DailyDialog experiments retain their attribution and results. The extension
is implemented by 13ray, using AI assistance to be disclosed per the team guide.

## Data and selection

Use `load("emowoz")` and `emotions("emowoz")`. Retain all seven native labels:
neutral, fearful, dissatisfied, apologetic, abusive, excited, satisfied.
Only user turns with a gold label are training/evaluation targets. System turns
remain available in the shared loader but these utterance-only baselines do not
consume them as context. Missing system labels are never treated as neutral.

Official labelled target counts: train 66,474 / dev 8,509 / test 8,634. This is
an **in-domain** extension, not a cross-dataset transfer experiment. No lossy
mapping is used. Do not compare native non-neutral F1 numerically with the
MELD/DailyDialog headline metrics as if they were the same task.

- **SVM:** sixteen dev configurations: original word or word+character
  features, C in [0.1, 0.3, 1.0, 3.0], and unweighted or balanced class weights.
  Selection uses non-neutral weighted-F1 on dev; ties follow configuration order.
- **BiGRU/BiLSTM:** preserve the original word-level architecture and fixed
  hyperparameters in `configs/b2_birnn.yaml`: train-only vocabulary, learned
  64-dimensional embeddings, 64 hidden units per direction, maximum 64 tokens,
  dropout 0.3, batch 128, maximum 12 epochs, patience 3, and inverse-square-root
  class weights. Only the label order and dev-selection metric are dataset-aware.
- **Seeds:** 0, 1, 2, 3, 4. Each RNN checkpoint uses the best dev epoch. The
  primary architecture is selected by mean best-dev score across the five seeds;
  both architectures are reported, regardless of their test ranking.
- All choices are persisted before test evaluation. Vocabulary/vectorizers and
  weights use train only. Test scores never select hyperparameters or seeds.

The three models are scored on the same official test targets, using native
weighted-F1 excluding neutral as the headline. Macro-F1 excluding neutral,
full-label macro/weighted-F1, per-class metrics, first/stable/shift groups,
confusion matrices and derived shift detection are also reported. Confidence
intervals resample whole dialogues 1,000 times for seed 0; seed variation uses
population SD, consistent with the original B1/B2 reports. These are different
uncertainty summaries and are labelled separately.

## Reproduce

CPU-only reproduction with the project's locked environment:

```powershell
uv sync --locked --extra rnn
uv run --extra rnn python -X utf8 scripts/train_emowoz_baselines.py --device cpu
uv run --extra rnn python -X utf8 scripts/make_emowoz_baseline_artifacts.py
uv run --extra rnn python -X utf8 -m pytest -q
```

Use `--models b1` or `--models b2` to run a single family. The dedicated extension
runner writes only new EmoWOZ reports and predictions; it does not overwrite
`reports/b1/results.md`, `reports/b2/results.md` or old dataset experiments.

For CUDA, create an isolated Python 3.12 environment, install the project and
notebook tools plus an official GPU-compatible torch build, then invoke its
Python directly with `--device cuda`. The measured GPU run uses PyTorch
2.6.0+cu126 and an NVIDIA RTX 4050 laptop GPU. Installation options are listed at
https://pytorch.org/get-started/previous-versions/ . `requirements_cpu.txt` and
`requirements_cuda.txt` in `reports/emowoz_baselines/` record the actual package
versions; install the project itself separately as an editable package.
The existing uv lock and default project environment are not changed to CUDA.

RNN checkpoints can resume completed seeds if config/source/data/runtime/device
signatures match. An interrupted epoch retrains that seed; no partial epoch is
reported as complete. CPU and CUDA runs can differ numerically, so do not mix
their completed checkpoints within one experiment. Runtime/device is recorded.

## Compatibility

`TfidfBaseline(..., labels=...)` now aligns probability columns to the configured
label list, with zero columns for absent training classes. New saved models
retain label order; old joblib checkpoints without label metadata default to
the original `EMOTIONS` order. SVM probabilities remain uncalibrated softmax
scores rather than confidence guarantees.

`BiRNNBaseline(..., labels=...)` uses the configured native label IDs for targets,
dev scoring, prediction and checkpoints. Original shared-label format-1
checkpoints still load; native-label format-1 checkpoints carry their own fixed
label order. `weights_only=True` loading is preserved. No shared data or shift
definition changes.

## Deliverables

- `reports/b1/emowoz/`: dev grid, selected SVM config, per-seed scores, top
  features, class/stratum/confusion tables and summary.
- `reports/b2/emowoz/`: five-seed dev comparison and architecture selection;
  each architecture has training histories, resume metadata, scores and tables.
- `predictions/b1_emowoz_test_seed<k>.csv` and
  `predictions/b2_bi{gru,lstm}_emowoz_test_seed<k>.csv`: fifteen CSVs total,
  original twelve columns first, source metadata afterwards.
- `models/b1_emowoz.joblib` (ignored like existing B1 models) and small
  `models/b2_bi{gru,lstm}_emowoz_seed<k>.pt` checkpoints for reproducibility/demo.
- `reports/emowoz_baselines/`: combined results, training manifests, verification,
  native-label figures with source CSVs, and runtime requirements.
- `notebooks/emowoz_b1_b2.ipynb`: executed data/selection/results/demo walkthrough.

The artifact generator audits source row identity, native probability order,
checkpoint round trips, dev selection and recomputed scores. Every figure is
at least 1600 pixels wide with source numbers supplied. Code cells intended for
the video are at most 15 lines. Slides, own-voice narration and personal forms
remain separate hand-in items.
