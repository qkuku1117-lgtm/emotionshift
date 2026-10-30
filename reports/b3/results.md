# B3 RoBERTa (utterance only) results

Mean ± std over 5 seeds, [95% dialogue-bootstrap CI, first seed], in %. roberta-base, lr 1e-05, batch 32; meld: 4 epochs, best seed on dev = 1; dailydialog: 3 epochs, best seed on dev = 0.

| Dataset | Test split | n | Weighted-F1 | Macro-F1 | Micro-F1 excl. neutral | Derived shift F1 |
|---|---|---|---|---|---|---|
| meld | test | 2610 | 62.85 ± 0.28 [60.0, 64.8] | 40.56 ± 0.72 [37.7, 41.1] | 49.88 ± 0.55 [46.2, 52.1] | 72.02 ± 0.71 |
| dailydialog | test_dedup | 7019 | 84.80 ± 0.03 [83.5, 86.1] | 39.69 ± 0.98 [36.2, 45.0] | 55.34 ± 0.35 [52.5, 58.2] | 52.28 ± 0.51 |
| dailydialog | test | 7740 | 84.65 ± 0.04 [83.4, 85.7] | 41.52 ± 1.01 [37.5, 47.3] | 55.30 ± 0.38 [52.5, 57.8] | 51.72 ± 0.48 |
| meld → dailydialog (E8) | test_dedup | 7019 | 80.15 ± 0.69 [79.6, 82.3] | 32.00 ± 0.63 [30.0, 34.2] | 40.92 ± 1.39 [39.5, 45.2] | 42.61 ± 0.84 |
| meld → dailydialog (E8) | test | 7740 | 79.62 ± 0.63 [79.1, 81.7] | 31.58 ± 0.54 [29.9, 33.6] | 40.06 ± 1.29 [38.4, 43.9] | 41.85 ± 0.72 |

## Efficiency (E9)

| Dataset | Parameters | Model files | Train time per seed | CPU latency per utterance (median / p95) | Batch throughput |
|---|---|---|---|---|---|
| meld | 124.7 M | 502 MB | 4.2 min | 105.8 / 138.5 ms | 488 utt/s (cuda) |
| dailydialog | 124.7 M | 502 MB | 14.0 min | 106.8 / 149.7 ms | 329 utt/s (cuda) |

Latency measured on CPU (2 cores, x86_64, Python 3.13.16), one utterance per call as in the demo; throughput on the training device.
