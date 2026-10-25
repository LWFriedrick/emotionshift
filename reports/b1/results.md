# B1 TF-IDF results

Mean ± std over 5 seeds, [95% dialogue-bootstrap CI, seed 0], in %.
Configs chosen on dev: meld: {'classifier': 'svm', 'C': 0.1, 'class_weight': 'balanced', 'features': 'word'}; dailydialog: {'classifier': 'svm', 'C': 0.3, 'class_weight': 'balanced', 'features': 'word'}

| Dataset | Test split | n | Weighted-F1 | Macro-F1 | Micro-F1 excl. neutral | Derived shift F1 |
|---|---|---|---|---|---|---|
| meld | test | 2610 | 56.78 ± 0.00 [54.6, 58.9] | 38.39 ± 0.00 [36.1, 40.5] | 40.91 ± 0.00 [38.1, 43.7] | 68.53 |
| dailydialog | test_dedup | 7019 | 83.88 ± 0.00 [82.5, 85.1] | 46.16 ± 0.00 [38.7, 51.0] | 53.03 ± 0.00 [49.9, 55.9] | 50.57 |
| dailydialog | test | 7740 | 83.74 ± 0.00 [82.5, 84.8] | 48.09 ± 0.00 [40.9, 53.3] | 53.45 ± 0.00 [50.5, 56.1] | 50.45 |

## Efficiency (E9)

| Dataset | Features | Parameters | Model file | Train time | Latency per utterance (median / p95) | Batch throughput |
|---|---|---|---|---|---|---|
| meld | 11,261 | 78,834 | 0.88 MB | 0.4 s | 0.64 / 1.04 ms | 70,777 utt/s |
| dailydialog | 98,289 | 688,030 | 7.90 MB | 7.9 s | 0.47 / 0.67 ms | 62,936 utt/s |

Measured on CPU only (2 cores, x86_64, Python 3.13.17); latency = one utterance per call, as in the demo.
