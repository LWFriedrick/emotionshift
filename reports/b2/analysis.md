# B2 measured results

Five seeds (0-4); architecture selected on mean dev score.
Reported CI is the seed-0 dialogue bootstrap, not an interval over seeds.

## meld

Dev-selected model: b2_bigru. Headline weighted_f1 on test: 55.08% +/- 0.45 percentage points across seeds. Seed-0 95% dialogue-bootstrap CI: [52.44, 56.73]%.

Existing repository B1 reference: 56.78%; B2 minus B1 = -1.70 percentage points. B1 was not rerun.

Stable turns: 62.12%; shift turns: 47.19%; stable minus shift: 14.93 percentage points. These are means across seeds of the dataset's headline metric, using gold strata for evaluation only.

## dailydialog

Dev-selected model: b2_bilstm. Headline micro_f1_no_neutral on test_dedup: 48.28% +/- 0.78 percentage points across seeds. Seed-0 95% dialogue-bootstrap CI: [46.04, 51.36]%.

Existing repository B1 reference: 53.03%; B2 minus B1 = -4.76 percentage points. B1 was not rerun.

Stable turns: 30.20%; shift turns: 70.55%; stable minus shift: -40.35 percentage points. These are means across seeds of the dataset's headline metric, using gold strata for evaluation only.

## CPU efficiency

| Dataset | Model | Parameters | Checkpoint MB | Mean fit seconds | Latency median / p95 ms |
|---|---|---|---|---|---|
| meld | b2_bigru | 223,623 | 0.95 | 35.6 | 0.98 / 2.12 |
| meld | b2_bilstm | 240,263 | 1.02 | 28.6 | 0.95 / 2.27 |
| dailydialog | b2_bigru | 831,623 | 3.57 | 362.1 | 1.10 / 2.38 |
| dailydialog | b2_bilstm | 848,263 | 3.64 | 358.5 | 0.96 / 2.29 |

All 20 fits together took 65.4 minutes on this CPU; test inference, bootstrap reporting and artifact generation take additional time.

## Interpretation limits

Both models learn embeddings from scratch and observe only the current utterance (up to 64 tokens). They use no conversation history, speakers, audio or images. A stable/shift gap is descriptive and does not establish that context will improve a model. DailyDialog micro F1 excludes neutral, so differences in emotion prevalence between strata also affect the comparison; a higher shift score is not evidence of shift forecasting. Softmax probabilities have not been calibrated.

Figures and matching CSVs are in figures/. Code snippets are named in the executed notebook. Thanh will prepare the slides and video separately.
