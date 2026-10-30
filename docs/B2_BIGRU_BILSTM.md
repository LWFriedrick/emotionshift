# B2: BiGRU / BiLSTM (Thanh)

This workstream compares two **utterance-only** recurrent baselines. Both read
the words of the current utterance in both directions. Neither receives other
utterances, speaker IDs, previous gold emotions, or gold shift labels at inference.
The shared loader still supplies speaker history and shift strata for evaluation.

## Run on Windows or Linux

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
python -m pip install -e ".[rnn]" pytest
python scripts/fetch_data.py
python -m pytest -q
python scripts/train_b2.py
python scripts/make_b2_artifacts.py
```

The Python download script uses the exact sources and raw formats in the shared
`fetch_data.sh`. Raw data stays under the existing ignored `data/raw/` directory.
PyTorch is an optional dependency, so the existing B1 workflow stays lightweight.
Neural tests skip when PyTorch is absent; the shared metric regression test still runs.
The full run uses CPU, four PyTorch threads, both datasets, both architectures,
and seeds 0, 1, 2, 3, 4. A GPU is optional (`--device cuda`, compatible PyTorch required).
For deterministic CUDA runs set `CUBLAS_WORKSPACE_CONFIG=:4096:8` before starting.
Use `--datasets meld` to run only MELD. Measured training seconds and CPU details
are recorded separately for every architecture/dataset in `summary.json`.
`reports/b2/analysis.md` gives the measured training times and their total.

```bash
python scripts/train_b2.py --quick
```

Quick mode uses small train/dev samples and five complete test dialogues, two
epochs, one seed and 20 bootstrap samples. All quick files are isolated under
`outputs_quick/b2/`; they must never be used as headline results.

## Architecture and training

Text is lowercased and split into Unicode words and individual punctuation.
The vocabulary is fitted on **train only**, retains tokens occurring at least
twice, and caps vocabulary size at 15,000 including PAD/UNK. Inputs are truncated
to the first 64 tokens. Empty strings become one UNK token. Batches are padded
and packed, so padding never becomes an extra recurrent observation.

Both models learn 64-dimensional embeddings from scratch, use a single
bidirectional recurrent layer with 64 units in each direction, concatenate the
two final hidden states, and apply a seven-class linear output. Dropout is 0.3
before the recurrent layer and before the classifier. For LSTM, classification
uses the hidden state, not the cell state. No pretrained vectors are downloaded.

Training uses AdamW (learning rate 0.002, weight decay 0.0001), batch size 128,
gradient norm clipping at 1.0, and a maximum of 12 epochs. Cross-entropy weights
are inverse square roots of training class counts, normalized to mean one among
present classes. Missing training classes receive zero loss weight. This gives
rare emotions more influence without full inverse-frequency weighting.

For each model and seed, the best dev headline score selects the checkpoint;
training stops after three successive epochs without a strict improvement.
The primary architecture for each dataset is the one with the highest **mean
best-dev score across the five seeds**. Ties follow config order. Both
architectures are reported as a predefined comparison. Selection is saved before
any test scoring; test results never select a checkpoint, hyperparameter or seed.
No train+dev refit is performed. The config is `configs/b2_birnn.yaml`.

## Evaluation and outputs

All metrics call `emotionshift.eval.metrics`. MELD headline is weighted F1 on
official test. DailyDialog headline is micro F1 excluding neutral on test_dedup;
official test is a secondary comparison. Every report includes macro F1,
per-class F1, confusion matrices, same-speaker strata, shift types and derived
shift detection. Derived shifts compare current and previous **predicted**
emotions. No history oracle is used.

The only shared evaluation change handles empty strata explicitly: zero F1,
`n=0`, and undefined accuracy. This preserves the intended `zero_division=0`
behavior with newer scikit-learn versions that reject empty arrays. A regression
test covers a sample with no stable turns, shifts or shift types.

Reported means and population standard deviations use seeds 0, 1, 2, 3, 4. Each 95%
percentile interval uses the shared 1,000-resample **dialogue** bootstrap on
seed 0. It describes uncertainty for that checkpoint, not across-seed uncertainty.
The separate standard deviation describes variation across seeds.

| Artifact | Location |
|---|---|
| Results table | `reports/b2/results.md` |
| Dev scores and architecture selection | `reports/b2/<dataset>/dev_comparison.csv`, `selection.json` |
| Metrics, CIs, efficiency, config | `reports/b2/<dataset>/b2_bi{gru,lstm}/summary.json` |
| Per-epoch training/dev history | same folder, `history_seed<k>.csv` |
| Strata and shift-type metrics | same folder, `stratified_<split>[_per_seed].csv` |
| Class metrics and confusion matrices | same folder, `per_class_<split>_seed<k>.csv`, `confusion_<split>_seed<k>.csv` |
| Predictions | `predictions/b2_bi{gru,lstm}_<dataset>_<split>_seed<k>.csv` |
| Demo checkpoints | `models/b2_bi{gru,lstm}_<dataset>_seed<k>.pt` |
| Figures (2160 pixels wide) and source numbers | `reports/b2/figures/*.png`, `*.csv` |
| Executed notebook | `notebooks/b2_bigru_bilstm.ipynb` |
| Config and data/source hashes | `reports/b2/run_manifest.json` |
| Result notes, references, contribution draft | `deliverables/03_thanh/` |

CSV columns exactly match the shared prediction contract. Model names are
`b2_bigru` and `b2_bilstm`; probabilities follow `EMOTIONS` order. Checkpoints
contain tensor weights, config, vocabulary, label order and dev history, and are
loaded with `torch.load(..., weights_only=True)`. Each checkpoint is below the
team's 50 MB limit. Keep the original B1 model-ignore rule; B2 `.pt` demo files
are intentionally eligible to include with this workstream.

## Demo interface

```python
from emotionshift.models.birnn import BiRNNBaseline
model = BiRNNBaseline.load("models/b2_bigru_meld_seed0.pt")
model.predict(["Oh my God!", "Yeah."])  # list[str]
model.predict_proba(["Oh my God!"])    # array, shape (1, 7)
```

Use `reports/b2/<dataset>/selection.json` to choose the dev-selected architecture
for the demo. Seed 0 is the designated demo checkpoint; it is not chosen by test
score. Empty batches return an empty list / `(0, 7)` probability array.
Softmax outputs have not been calibrated. CPU latency is measured by the shared
efficiency helper with one utterance per call and recorded alongside throughput,
parameter count, vocabulary size, checkpoint size and training time.

## Video snippets

In the notebook show the cells immediately under **Shared prediction export**
and **Demo interface**. Both are at most 15 lines. The notebook executes loading,
inspection, exported prediction validation, charts and inference. Full training
runs in the reproducible script rather than being repeated on every notebook run.
Result notes and matching PNG/CSV assets are supplied in Thanh's folder.
Thanh will prepare the slides and video separately, using the team template
and own-voice narration as required by the guide.

## Interpretation and limits

These experiments test word order without conversational context. Within an
utterance, a backward encoder is available because the whole current utterance
is observed; it never reads a later conversation turn. Short texts, class
imbalance, learned-from-scratch embeddings and token truncation can constrain
performance. A gap between stable and shift groups is descriptive: it does not
prove that adding context will improve a model. The B1 reference line comes from
the repository's existing report and uses a different model.
Compare actual results in `reports/b2/results.md`; do not assume either neural
model improves on the TF-IDF baseline.

See `deliverables/03_thanh/refs.txt` for APA references. Implementation and
preparation used AI assistance. Review and understand the code, disclose this
complete unedited chat as the guide requires, and complete the personal forms.

## Validation and GitHub hand-in

The full test suite passed all 19 checks. The smoke run completed on both datasets.
The artifact audit (`python scripts/verify_b2.py`) checks all 30 prediction CSVs
against the shared loader and recomputes their stored metrics, checks checkpoint
sizes, figure resolution and source CSVs, and validates notebook execution.
Its result is `reports/b2/verification.json`.

To contribute, fork the upstream repository, add the new/modified files from the
submission bundle to your fork, review the diff, and open a pull request for
Lixian. Use `docs/B2_PR_DESCRIPTION.md` as the PR description. Keep `data/raw/`,
`.venv/`, caches and `outputs_quick/` out of Git. The bundle includes the small
saved `.pt` files required for the demo. Slides and video are prepared separately.
