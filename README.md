# EmotionShift — EE6405 Group Project (Table 4)

Recognizing, anticipating and explaining emotional turning points in conversation.
We test where conversational context helps or hurts at the moments a speaker's
emotion changes, on MELD and DailyDialog.

**Team members: read the hand-in standards first —
[`docs/TEAM_GUIDE.pdf`](docs/TEAM_GUIDE.pdf) (2 pages) or [`docs/TEAM_GUIDE.md`](docs/TEAM_GUIDE.md).**

## Workstreams

| # | Model | Owner | Status |
|---|---|---|---|
| 1 | Majority / emotional-inertia baseline | open | — |
| 2 | TF-IDF + linear SVM | Lixian | ✅ done: [`docs/B1_TFIDF_SVM.md`](docs/B1_TFIDF_SVM.md) |
| 3 | BiGRU / BiLSTM | Thanh | claimed |
| 4 | Transformer, utterance only (proposal: RoBERTa-base) | KL Y | claimed |
| 5 | Context-aware RoBERTa | Anqi | claimed |
| 6 | ShiftAware (speaker / shift-aware) | open | — |
| — | Demo, slides and video integration | Lixian | in progress |

Shared pieces already in the repo, for everyone to reuse: data pipeline, shift
labels, metrics, efficiency helpers, prediction format, tests.

## Quick start

```bash
uv sync                          # or: pip install numpy pandas scikit-learn scipy joblib pyyaml matplotlib pytest
bash scripts/fetch_data.sh       # downloads MELD + DailyDialog into data/raw/ (not in Git)
uv run pytest                    # 15 checks: data rules, pilot numbers, metrics, B1 interface
```

```python
from emotionshift.data.loaders import load
df = load("meld")        # or "dailydialog"; one row per utterance
# columns: dataset, split, dialogue_id, turn, speaker, text, emotion, prev_emotion, stratum, shift_type
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
| DailyDialog de-dup | `test_dedup` = test minus dialogues that appear **verbatim** in train (105 of 996 unique) and minus repeats inside test → 891 dialogues, 7,019 utterances. **Headline DailyDialog numbers use `test_dedup`**; official `test` is reported for comparison | test |
| Shift | Emotion differs from the **same speaker's** previous utterance in the dialogue; a speaker's first turn is `first` (neither shift nor stable). MELD shifts 4,001 / 427 / 1,003 = MELD-FR flips ±1 | test |
| Shift types | Priority: onset (neutral→x), offset (x→neutral), surprise (surprise↔non-neutral), within-negative, valence flip. MELD test: 355 / 340 / 158 / 81 / 69 | test |
| Tuning | Hyper-parameters chosen on **dev** only; test scored once per seed | — |
| Metrics | MELD headline weighted-F1; DailyDialog headline micro-F1 excluding neutral; always also macro-F1, per-class F1, stratified scores, 95% CI | test vs sklearn |
| CIs | Bootstrap over **dialogues** (1,000 resamples), since utterances in one dialogue are not independent | — |
| No gold at inference | History features use predicted emotions; gold-history results are labelled "oracle" | — |

The proposal's MELD and DailyDialog pilot numbers are reproduced exactly by
`tests/test_data.py`: always-neutral 0.313 and oracle inertia 0.449 weighted-F1 on
MELD test; MELD shift rate 53.8%; Markov shift forecast AUROC 0.663 / AUPRC 0.673;
the self-transition rates and neutral-attractor range in the Interpretation
section; DailyDialog shift rate 18.0% and oracle inertia 0.397 micro-F1 excl.
neutral. Not yet reproduced: the MELD-FR previous-utterance trigger baseline
(F1 0.684), which needs the MELD-FR data (SemEval-2024 Task 10) used by E7. Two notes: the proposal's "197 of 1,000 DailyDialog test dialogues
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
M.scores(gold, pred)                 # weighted / macro / micro-excl-neutral F1, accuracy
M.stratified(df)                     # per stratum and shift type, with n
M.derived_shift_detection(df)        # T2 from predicted labels
M.bootstrap_ci(df, "weighted_f1")    # 95% CI, dialogue-level resampling
E.latency_ms(model.predict, texts)   # E9: per-utterance CPU latency (median, p95)
```

## Repository layout

```
emotionshift/data/       loaders.py (one Dialogue/Turn table), labels.py, shifts.py
emotionshift/models/     tfidf.py (#2)  · add your model here
emotionshift/eval/       metrics.py, efficiency.py, plots.py
configs/                 one YAML per experiment (b1_tfidf.yaml)
scripts/                 fetch_data.sh, train_b1.py, make_notebook.py, make_team_guide_pdf.py
notebooks/               executed walkthroughs (b1_tfidf_svm.ipynb)
tests/                   test_data.py, test_tfidf.py
reports/                 results tables and figures per model (reports/b1/)
predictions/             shared-format prediction CSVs
docs/                    team guide, per-model write-ups
deliverables/            temporary collection folders for the video (removed before submission)
```

## Data sources

MELD (Poria et al., 2019) from github.com/declare-lab/MELD; DailyDialog (Li et al.,
2017) official split via github.com/declare-lab/conv-emotion. Both are for
non-commercial research use; the raw data is downloaded by `scripts/fetch_data.sh`
and never committed.
