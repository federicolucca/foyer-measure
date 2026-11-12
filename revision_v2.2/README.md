# revision_v2.2 — scripts behind paper v2.2 (arXiv:2610.32955 v4)

Paper v2.2 corrects v2.1 after an external review. Every number that changed in v2.2 comes from these scripts, run on
the private label files and the saved per-row result files of the earlier runs. **No data are included**: no emails,
labels, per-email predictions or result JSONs. The JSON keys and some file names are in Italian, as in the paper's runs.

| Script | What it produces |
|---|---|
| `revisione_xsci.py` | One JSON of aggregate numbers: label accounting, contacts with more than one relationship, accuracies with thread-cluster bootstrap intervals, exact McNemar against B, bodies longer than 700 characters, repeated-CV baseline, crossovers, thread-bootstrap intervals on cost differences. Reads only saved results (no model is re-run). |
| `revisione_costi.py` | The cost-chosen is-noise threshold for every system that returns P(noise) (B, Nimble v4-max, Jev, Tev1 0.8B), chosen inside thread-grouped CV, and thread-bootstrap intervals on L_S(10) differences. Needs `cost_threshold.py` from the repository root. |
| `azioni_politiche_v22.py` | The agent's actions under the three composition policies (noise wins, action wins, consistency) and the cost L_S(c) = c·m + t + 0.5·d + 0.5·f + U·n for c = 1..20, on the 1,500-character benchmark corpus. |
| `figura_costi_v22.py` | The cost-curve figure (two panels, v2 and v3) and the exact crossovers JSON. |
| `tabella_politiche_v22.py` | The policy table (`tab_policy.tex`) from the policy JSONs. |
| `tabella_test_v22.py` | The appendix table: accuracy on every field with 95% thread-cluster bootstrap intervals and exact McNemar against B, Holm-corrected over all 38 tests. |
| `b_semi_v22.py` | The trained baseline B over 5 seeds of thread-grouped StratifiedGroupKFold(5), with word 1-2-grams and with word + character 3-5-grams. |
| `rerun_v22.py` | The hosted runs of v2.2: the relationship negative control (zero-shot sender relationship from one email, 8 classes, with the paying-client definition) and one repeated Gemma run on type. It imports the pilot harness the original hosted runs used (`bench_online_pilota.py`, folder in `FOYER_PILOT_DIR`), which is not in this repository; `online_backends.py` exposes the same backends. Keys from environment variables only. |

Each script's docstring gives its command line.
