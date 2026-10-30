# Add BiGRU and BiLSTM utterance baselines for Thanh's workstream

Adds word-level BiGRU and BiLSTM classifiers with train-only vocabularies,
dev-selected checkpoints and the shared `predict` / `predict_proba` demo interface.
The experiment runner uses the shared MELD and DailyDialog splits, labels and
metrics, five seeds, dialogue-bootstrap CIs, and standard prediction CSVs.
Both architectures are reported; primary architecture selection uses dev only.

Includes 20 demo checkpoints, 30 prediction CSVs, an executed notebook, machine-readable
reports, figures with source CSVs, and reproducible run instructions. MELD's
dev-selected BiGRU scores 55.08% weighted F1; DailyDialog's dev-selected BiLSTM
scores 48.28% micro F1 excluding neutral on test_dedup (means across five seeds).
The existing TF-IDF baseline remains stronger on both headline metrics.

A small shared metric fix preserves zero-F1 / n=0 behavior for empty strata with
newer scikit-learn versions. PyTorch is optional; neural tests skip when it is
absent, preserving the lightweight B1 install.

Validation: 19 tests passed; both-dataset smoke run completed; optional-dependency
skip behavior checked; all 30 CSVs match shared-loader rows and reported metrics;
all nine notebook code cells executed without errors. Checkpoints are below
50 MB. Raw data and the Python environment are excluded from the submission bundle.

See `docs/B2_BIGRU_BILSTM.md` for methods and commands and `reports/b2/analysis.md`
for result interpretation, confidence intervals and measured CPU efficiency.
