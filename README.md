# EmotionShift — B1: TF-IDF + linear classifier

EE6405 group project (Table 4). This folder holds **B1**, the traditional-ML
baseline from the proposal (Ruizhi's table: #2 TF-IDF + SVM), plus the shared
data pipeline and evaluation code it needs. Owner: Lixian.

B1 reads **one utterance at a time**: no dialogue context, no speaker. It is the
reference point that every context-aware model (#4–#6) should beat.

**Team members: read [`docs/TEAM_GUIDE.md`](docs/TEAM_GUIDE.md) for what to hand in and in which format.**

## Quick start

```bash
uv sync                          # or: pip install numpy pandas scikit-learn scipy joblib pyyaml pytest
bash scripts/fetch_data.sh       # downloads MELD + DailyDialog into data/raw/
uv run pytest                    # 9 data/metric checks (takes ~5 s)
uv run python scripts/train_b1.py          # full run, about 20 minutes on CPU
uv run python scripts/train_b1.py --quick  # 1-minute smoke test
```

## Results

Mean ± std over seeds, [95% dialogue-bootstrap CI, seed 0], in %.
Configs chosen on dev: meld: {'classifier': 'svm', 'C': 0.1, 'class_weight': 'balanced', 'features': 'word'}; dailydialog: {'classifier': 'svm', 'C': 0.3, 'class_weight': 'balanced', 'features': 'word'}

| Dataset | Test split | n | Weighted-F1 | Macro-F1 | Micro-F1 excl. neutral | Derived shift F1 |
|---|---|---|---|---|---|---|
| meld | test | 2610 | 56.78 ± 0.00 [54.6, 58.9] | 38.39 ± 0.00 [36.1, 40.5] | 40.91 ± 0.00 [38.1, 43.7] | 68.53 |
| dailydialog | test_dedup | 7019 | 83.88 ± 0.00 [82.5, 85.1] | 46.16 ± 0.00 [38.7, 51.0] | 53.03 ± 0.00 [49.9, 55.9] | 50.57 |
| dailydialog | test | 7740 | 83.74 ± 0.00 [82.5, 84.8] | 48.09 ± 0.00 [40.9, 53.3] | 53.45 ± 0.00 [50.5, 56.1] | 50.45 |

MELD test, stratified by gold shift labels (mean over seeds, n per cell):

| Group | n | Weighted-F1 | Macro-F1 |
|---|---|---|---|
| all | 2610 | 56.8 | 38.4 |
| stratum:first | 746 | 62.2 | 40.8 |
| stratum:stable | 861 | 62.1 | 32.9 |
| stratum:shift | 1003 | 49.1 | 38.9 |
| shift_type:onset | 355 | 46.3 | 33.2 |
| shift_type:offset | 340 | 85.9 | 12.3 |
| shift_type:surprise | 158 | 51.6 | 28.7 |
| shift_type:within_negative | 81 | 32.1 | 16.5 |
| shift_type:valence_flip | 69 | 45.4 | 23.0 |

Reading the numbers:
- MELD weighted-F1 56.8 sits between the oracle-inertia floor (44.9) and the
  published context models (DialogueGCN 58.10, COSMIC 65.21), as expected for an
  utterance-only model.
- On MELD shift turns B1 scores 49.1 vs 62.1 on stable turns: the gap the project
  is about, even without any context. Offsets (emotion → neutral) are easy (85.9)
  because B1 predicts neutral often; within-negative shifts are hardest (32.1).
- Dev selection picked a linear SVM with balanced class weights and word 1–2-grams
  on both datasets; adding character n-grams did not help on dev.
- Std over seeds is 0: LinearSVC converges to the same solution whatever the
  seed, so the 3 seeds satisfy R4 formally but the bootstrap CI is the
  meaningful uncertainty here.
- Highest-weighted n-grams (MELD): surprise "wow", "whoa", "really ?";
  disgust "ugh", "ew"; sadness "sorry"; fear "scared", "oh no" -- see
  `reports/b1/meld/top_features.json`.


## Data rules (aligned with the proposal — please use the same in your model)

All of this lives in `emotionshift/data/` and is checked by `tests/test_data.py`.
Anyone can reuse it: `from emotionshift.data.loaders import load; df = load("meld")`.

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

Pilot numbers from the proposal are reproduced exactly by `tests/test_data.py`:
always-neutral 0.313 and oracle inertia 0.449 weighted-F1 on MELD test; MELD
shift rate 53.8%; DailyDialog shift rate 18.0% and oracle inertia 0.397
micro-F1 excl. neutral.

Two small notes for the group:
- The proposal says 197 of 1,000 DailyDialog test dialogues share their first two
  turns with train. Exact matching gives 182 here; the 105 verbatim figure (the one
  that matters for de-dup) matches.
- The shift-type order above is the one that reproduces the proposal's counts;
  "surprise-involved" means surprise next to another *non-neutral* emotion.

## Shared prediction format (for #1 inertia, the comparison table and the demo)

`predictions/b1_<dataset>_<split>_seed<k>.csv`, one row per utterance:

| column | meaning |
|---|---|
| `dataset`, `split`, `dialogue_id`, `turn`, `speaker`, `text` | identify the utterance |
| `emotion` | gold label |
| `pred` | predicted label (one of the 7) |
| `stratum`, `shift_type` | gold shift labels, for stratified scoring |
| `model`, `seed` | which run |

If every model writes this format, the comparison table, the stratified analysis
and the demo can read all of them the same way.

## Using the trained model

```python
from emotionshift.models.tfidf import TfidfBaseline
m = TfidfBaseline.load("models/b1_meld.joblib")
m.predict(["Oh my God!", "Yeah."])
m.predict_proba(["Oh my God!"])      # (n, 7) in label order above
m.top_features(10)                   # top n-grams per emotion (interpretation slide)
```

## Files

```
emotionshift/data/    loaders.py (one Dialogue/Turn table), labels.py, shifts.py
emotionshift/models/  tfidf.py (B1)
emotionshift/eval/    metrics.py (scores, stratified, derived shift detection, bootstrap CI)
configs/b1_tfidf.yaml dev grid, seeds, test splits
scripts/              fetch_data.sh, train_b1.py, make_notebook.py
notebooks/            b1_tfidf_svm.ipynb (executed walkthrough with outputs, for the video)
tests/                test_data.py
reports/b1/           figures/*.png, dev_grid.csv, summary.json, stratified / per-class / confusion CSVs, results.md
predictions/          shared-format prediction CSVs
```

Data sources: MELD (Poria et al., 2019) from github.com/declare-lab/MELD;
DailyDialog (Li et al., 2017) official split via github.com/declare-lab/conv-emotion.
Both are for non-commercial research use.
