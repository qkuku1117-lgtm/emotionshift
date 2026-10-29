# ShiftAware results (test, mean ± std over seeds)

Checkpoints and variants are selected on dev only. Shift-turn F1 is the seed mean.

| dataset | split | model | seeds | weighted F1 | macro F1 | micro F1 (no neutral) | weighted F1 on shift turns | shift-head AUROC |
|---|---|---|---|---|---|---|---|---|
| meld | test | B1 TF-IDF | 5 | 56.78 ± 0.00 | 38.39 ± 0.00 | 40.91 ± 0.00 | 49.15 | n/a |
| meld | test | shiftaware | 5 | 62.68 ± 0.46 | 41.71 ± 1.53 | 50.43 ± 0.44 | 53.94 | 73.78 ± 1.19 |
| meld | test | shiftaware_cw-sqrt | 5 | 62.52 ± 0.51 | 46.33 ± 0.72 | 49.64 ± 0.73 | 54.95 | 74.75 ± 0.48 |
| dailydialog | test_dedup | B1 TF-IDF | 5 | 83.88 ± 0.00 | 46.16 ± 0.00 | 53.03 ± 0.00 | 70.06 | n/a |
| dailydialog | test | B1 TF-IDF | 5 | 83.74 ± 0.00 | 48.09 ± 0.00 | 53.45 ± 0.00 | 70.60 | n/a |
