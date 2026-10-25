# deliverables/ — collection folders for the final video (temporary)

Each member puts the material for the video in their own folder, through a pull
request (or sends it to Lixian if the files are too large for Git). What to put
here and in which format: [`docs/TEAM_GUIDE.pdf`](../docs/TEAM_GUIDE.pdf).

| Folder | Workstream | Owner |
|---|---|---|
| `01_inertia/` | #1 Majority / emotional-inertia baseline | open |
| `02_lixian/` | #2 TF-IDF + linear SVM | Lixian |
| `03_thanh/` | #3 BiGRU / BiLSTM | Thanh |
| `04_kly/` | #4 Transformer, utterance only | KL Y |
| `05_anqi/` | #5 Context-aware RoBERTa | Anqi |
| `06_shiftaware/` | #6 ShiftAware | open |

Typical contents: `slides.pptx`, `script.md`, `figures/` (PNG + matching CSV),
`audio/` (one file per slide, own voice), `refs.txt`, `contribution.txt`.

Code and results do **not** go here: they go in the package (`emotionshift/`),
`notebooks/`, `reports/` and `predictions/`.

This folder is removed before the final submission, so the repo ends up holding
the project's code and results only.
