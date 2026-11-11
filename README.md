# EmotionShift — EE6405 Group Project (Table 4)

Recognizing, anticipating and explaining emotional turning points in conversation.
We test where conversational context helps or hurts at the moments a speaker's
emotion changes, on MELD and DailyDialog.

**Team members: read the hand-in standards first —
[`docs/TEAM_GUIDE.pdf`](docs/TEAM_GUIDE.pdf) (2 pages) or [`docs/TEAM_GUIDE.md`](docs/TEAM_GUIDE.md).**

## Workstreams

| # | Model | Owner | Status |
|---|---|---|---|
| 1 | Majority / emotional-inertia / Markov baseline | Ruizhi | ✅ done, merged 2 Nov: [`docs/B0_BASELINES.md`](docs/B0_BASELINES.md), [`results`](reports/b0/results.md) |
| 2 | TF-IDF + linear SVM | Lixian | ✅ done: [`docs/B1_TFIDF_SVM.md`](docs/B1_TFIDF_SVM.md) |
| 3 | BiGRU / BiLSTM | Thanh | ✅ done, merged 30 Oct: [`docs/B2_BIGRU_BILSTM.md`](docs/B2_BIGRU_BILSTM.md) |
| 4 | Transformer, utterance only (RoBERTa-base) | KL Y | ✅ done, merged 30 Oct: [`docs/B3_ROBERTA.md`](docs/B3_ROBERTA.md); trained models on Google Drive |
| 5 | Context-aware RoBERTa | Anqi | ✅ done, merged 10 Nov: [`docs/B4_CONTEXT.md`](docs/B4_CONTEXT.md), [`results`](reports/b4/results.md) |
| 6 | ShiftAware (speaker / shift-aware) | Fpengz | ✅ done, merged 1 Nov: [`results`](reports/shiftaware/results.md), [`selection`](reports/shiftaware/selection.md); notebook and α column still to add |
| — | Demo, slides and video integration | Lixian | in progress |

Shared pieces already in the repo, for everyone to reuse: data pipeline, shift
labels, metrics, efficiency helpers, prediction format, tests.

## Quick start

```bash
uv sync                          # Python 3.12, dependencies and notebook tools
bash scripts/fetch_data.sh       # downloads MELD, MELD-FR, DailyDialog JSON and EmoWOZ into data/raw/
                                 # (scripts/fetch_data.py does the same, cross-platform)
uv run pytest                    # data rules, pilot numbers, metrics, original APIs and model interfaces
                                 # B2 checks need the rnn extra (uv sync --extra rnn),
                                 # B3 checks the transformers extra (uv sync --extra transformers); skipped without them
```

```python
from emotionshift.data.loaders import load

df = load("meld")  # or "dailydialog" or "emowoz"; one row per utterance
# Official train/dev/test by default, plus test_dedup for DailyDialog.
dd = load("dailydialog", splits=("train", "dev", "test_dedup"))
```

## How to add your work (no invite needed)

1. **Fork** this repo.
2. Add your files in your fork, **in your own folder or module** (git push, or *Add file › Upload files* on GitHub).
3. **Contribute › Open pull request.** Lixian reviews and merges.

Before opening the pull request: use the shared loader (no separate data splits),
write predictions in the shared format below, include one executed notebook,
keep raw data and files over 50 MB out of Git, and make sure `pytest` passes.

## Shared data rules (aligned with the proposal)

All in `emotionshift/data/`, checked by `tests/test_data.py`.

| Rule | What the code does | Check |
|---|---|---|
| Splits | MELD official 9,989 / 1,109 / 2,610 utterances. DailyDialog official 11,118 / 1,000 / 1,000 dialogues | test |
| Labels | 7 shared labels, fixed order `neutral, joy, surprise, anger, sadness, disgust, fear`. DailyDialog `happiness`→`joy`, `no_emotion`→`neutral` | test |
| DailyDialog speakers | Two-party; speakers alternate A, B, A, … | — |
| EmoWOZ | Official split 9,233 / 1,100 / 1,100 dialogues (66,474 / 8,509 / 8,634 labelled user turns). Own 7 labels `neutral, fearful, dissatisfied, apologetic, abusive, excited, satisfied` (`labels.emotions("emowoz")`), not mapped to the shared set. Speakers `user`, `system`; system turns are kept as context with no `emotion` and no `stratum`, and are never scored | test |
| DailyDialog de-dup | `test_dedup` = test minus dialogues that appear **verbatim** in train (105 of 996 unique) and minus repeats inside test → 891 dialogues, 7,019 utterances. **Headline DailyDialog numbers use `test_dedup`**; official `test` is reported for comparison | test |
| Shift | Emotion differs from the **same speaker's** previous utterance in the dialogue; a speaker's first turn is `first` (neither shift nor stable). MELD shifts 4,001 / 427 / 1,003 = MELD-FR flips ±1 | test |
| Shift types | Priority: onset (neutral→x), offset (x→neutral), surprise (surprise↔non-neutral), within-negative, valence flip, within-positive (EmoWOZ only). MELD test: 355 / 340 / 158 / 81 / 69 | test |
| Tuning | Hyper-parameters chosen on **dev** only; test scored once per seed | — |
| Metrics | MELD headline weighted-F1; DailyDialog headline micro-F1 excluding neutral; EmoWOZ headline weighted-F1 excluding neutral (macro-F1 excluding neutral alongside); always also macro-F1, per-class F1, stratified scores, 95% CI | test vs sklearn |
| CIs | Bootstrap over **dialogues** (1,000 resamples), since utterances in one dialogue are not independent | — |
| No gold at inference | History features use predicted emotions; gold-history results are labelled "oracle" | — |

The proposal's MELD and DailyDialog pilot numbers are reproduced exactly by
`tests/test_data.py`: always-neutral 0.313 and oracle inertia 0.449 weighted-F1 on
MELD test; MELD shift rate 53.8%; Markov shift forecast AUROC 0.663 / AUPRC 0.673;
the self-transition rates and neutral-attractor range in the Interpretation
section; DailyDialog shift rate 18.0% and oracle inertia 0.397 micro-F1 excl.
neutral. The MELD-FR previous-utterance trigger baseline (test F1 0.684) is reproduced
by `uv run python pilot/pilot_stats.py`, separately from the B1 tests. Two notes: the proposal's "197 of 1,000 DailyDialog test dialogues
share their first two turns with train" comes out as 182 with exact matching (the
105 verbatim figure used for de-dup matches); and the shift-type order above is the
one that reproduces the proposal's counts.

## Shared prediction format

`predictions/<model>_<dataset>_<split>_seed<k>.csv`, one row per utterance:

| Column | Meaning |
|---|---|
| `dataset`, `split`, `dialogue_id`, `turn`, `speaker`, `text` | Identify the utterance |
| `emotion` | Gold label |
| `pred` | Predicted label (one of the 7) |
| `stratum`, `shift_type` | Gold shift labels, for stratified scoring |
| `model`, `seed` | Which run |

The comparison table, the stratified analysis and the demo read every model's
predictions this way. Models for the demo (#3–#6) also expose
`predict(list_of_utterances) -> list_of_emotions`.

## Shared evaluation helpers

```python
from emotionshift.eval import metrics as M, efficiency as E

M.scores(gold, pred)  # weighted / macro / micro-excl-neutral F1, accuracy
M.stratified(df)  # per stratum and shift type, with n
M.derived_shift_detection(df)  # T2 from predicted labels
M.bootstrap_ci(df, "weighted_f1")  # 95% CI, dialogue-level resampling
E.latency_ms(model.predict, texts)  # E9: per-utterance CPU latency (median, p95)
```

## Repository layout

```
emotionshift/data/       loaders.py (one Dialogue/Turn table), labels.py, shifts.py
emotionshift/models/     tfidf.py (#2), birnn.py (#3), roberta.py (#4), shiftaware/  · add your model here
emotionshift/eval/       metrics.py, efficiency.py, plots.py
configs/                 one YAML per experiment (b1_tfidf.yaml, b2_birnn.yaml, b3_roberta.yaml,
                         shiftaware.yaml, shiftaware_train.yaml)
scripts/                 fetch_data.sh (fetch_data.py: same, cross-platform), train_b1.py, train_b2.py,
                         make_b2_artifacts.py, verify_b2.py, train_b3.py, analyse_b3.py,
                         make_notebook_b3.py, train_shiftaware.py, evaluate_shiftaware.py,
                         summarize_shiftaware.py, diagnose_shiftaware.py, smoke_shiftaware.py
notebooks/               original exploration notebooks and executed walkthroughs
                         (b1_tfidf_svm.ipynb, b2_bigru_bilstm.ipynb, b3_roberta.ipynb)
tests/                   original tests, model tests (B1, B2, B3, ShiftAware) and merge compatibility checks
reports/                 results tables and figures per model (reports/b1/, reports/b2/, reports/b3/,
                         reports/shiftaware*/)
predictions/             shared-format prediction CSVs
models/                  small demo checkpoints only (b2_*.pt); large models live on Google Drive
docs/                    proposal snapshot, team guide, per-model write-ups
pilot/                   original proposal pilot scripts and statistics
deliverables/            temporary collection folders for the video (removed before submission)
```

## EmoWOZ shared-label view

**Contributor: Rachel ([13ray](https://github.com/13ray)).**
Details: **[EmoWOZ integration, label mapping and run instructions](docs/EMOWOZ.md)**.

`load("emowoz")` keeps EmoWOZ's native labels. `load_emowoz_shared()` returns user
turns on the shared seven labels through an explicit, lossy mapping, for
cross-dataset comparison. [`reports/emowoz/`](reports/emowoz/) has reproducible
split, native-label, shared-label and shift tables from
`uv run python scripts/emowoz_stats.py`. No trained-model results come from this view.

## Data sources

MELD (Poria et al., 2019) from github.com/declare-lab/MELD; DailyDialog (Li et al.,
2017) official split via github.com/declare-lab/conv-emotion; EmoWOZ (Feng et al.,
2022) from zenodo.org/record/6506504 (CC BY-NC 4.0). All are for
non-commercial research use; the raw data is downloaded by `scripts/fetch_data.sh`
and never committed.

## Data interface

All imports now share the same shift labels (`surprise`) and seven-class macro-F1.
All notebooks and models use the unified loader in `emotionshift/data/loaders.py`;
`datasets.py` retains compatibility imports only. `load(dataset, splits=None,
root=DATA_ROOT)` defaults to official train/dev/test, plus `test_dedup` for
DailyDialog, and reads only the requested splits. To use another data directory, pass `root` containing `meld/`,
`dailydialog/` and `emowoz/`. No data is downloaded during loading.

Every utterance has `dataset`, `split`, `dialogue_id`, `source_split`,
`source_dialogue_id`, `turn`, `speaker`, `raw_text`, `text`, `emotion`,
`prev_emotion`, `turn_gap`, `stratum`, and `shift_type`. `dialogue_id` is the
source dialogue ID, unique only within a split, so key utterances by dataset,
split, dialogue ID and turn. Rows follow source dialogue ID and turn order within
each requested split.

DailyDialog uses the conv-emotion JSON release for every model. `raw_text` is
retained unchanged; `text` uses B1's Unicode/punctuation normalization and
whitespace collapse, preserving tokenized punctuation spacing. Deduplication
uses ordered, case-sensitive raw text with whitespace collapsed, before text
cleanup. `test_dedup` removes train overlap and repeated test dialogues,
retaining the first remaining copy. Official `test` is unchanged.
MELD-FR remains available through `load_meld_fr(split, root=...)` with its
instance IDs, trigger labels, raw text, and the same text cleanup.

## Notebooks and pilot

`notebooks/01_data_exploration.ipynb`, `notebooks/02_quick_baselines.ipynb`,
`pilot/`, their reports, and the proposal snapshot in `docs/proposal.md` are retained.
Open notebooks with `uv run jupyter notebook`.
The working proposal is the shared
[EmotionShift Project Proposal](https://claude.ai/artifact/JbtymfhtEeBZyuho7A4Eka);
the local snapshot may lag behind it.

The exploration notebooks and their reports have been rerun using these rules.
The pilot script also uses the unified loader; `pilot/pilot_output.txt` is retained
as a historical record of the original proposal run.
Saved B1 prediction CSVs retain their predictions and scores; their dialogue IDs
and source metadata have been migrated to the unified schema.

## PyTorch training

See [the training guide](docs/TRAINING.md) for utterance and dialogue datasets,
causal context and forecasting, tokenizers, tensor shapes, padding and loss masks.

The modular [ShiftAware implementation](emotionshift/models/shiftaware/README.md)
supports cached embeddings or an optional RoBERTa encoder, causal speaker
attention, global and speaker memories, forecasting, and a joint masked loss.
Run `uv run --extra train python scripts/smoke_shiftaware.py` for an offline
optimization check; this does not train a benchmark model.
For an end-to-end text run, use
`uv run --extra train python scripts/train_shiftaware.py --dataset meld --output-dir models/shiftaware/meld_seed42`
after fetching MELD and the chosen pretrained encoder. Training settings are
in `configs/shiftaware_train.yaml`; the best checkpoint is selected on dev.

```bash
uv sync --extra train
uv run --extra train pytest
```

## Native EmoWOZ SVM and recurrent baselines

The workstream 2/3 extension trains TF-IDF/SVM, BiGRU and BiLSTM on the
native EmoWOZ labels using official splits and five seeds. See the
[protocol and reproduction guide](docs/EMOWOZ_B1_B2.md),
[audited results](reports/emowoz_baselines/results.md) and
[executed notebook](notebooks/emowoz_b1_b2.ipynb).
The original model authors retain credit; 13ray contributes this dataset
extension. Existing MELD/DailyDialog results are preserved.

## Development checks

Use the tools directly from the repository root:

```bash
uv run ruff check .
uv run ruff format --check .
uv run --extra train ty check
uv run vulture
uv run --extra train pytest
```

Run `uv run ruff check --fix .` and `uv run ruff format .` to apply safe fixes
and formatting. Ruff also formats code cells while retaining notebook outputs.
Tool settings live in `pyproject.toml`; there is no separate task runner.

Vulture reports findings at 80% confidence and above. Notebook-only plotting
helpers, public TypedDict keys, and framework hooks often appear as unused at
lower confidence. To review those candidates manually, run
`uv run vulture --min-confidence 60`; a low-confidence finding alone is not a
reason to delete a public interface.

Edit the notebooks directly. `docs/TEAM_GUIDE.md` is the authoritative hand-in
guide; the PDF is an exported handout. Neither needs a Python generator as a
second copy of its content. Local caches and smoke-test output are ignored by Git.

## AI use

The capstone requires a declaration of how AI tools were used, and code
"generated purely by AI tools" is penalized. The original pilot scripts, data and
evaluation modules, tests and exploration notebooks were written with AI
assistance. This merge and its compatibility changes were also AI-assisted;
record AI use in the declaration and make sure the team understands and owns
the code.
