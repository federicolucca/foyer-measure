"""
curve_baseline — the TF-IDF baseline of the learning curve on the fixed test set (no GPU).

  python curve/curve_baseline.py --user 1 --consent u1=consent_u1.json --data labels.jsonl \
      --split split_paper3.json --subsets curve_subsets.json --out results

Same run as `decision_model.py --system baseline`: for each seed (7, 11) and size (50 ... all) of
curve_subsets.json, one model per field (word TF-IDF 1-2 + character TF-IDF 3-5, balanced logistic regression,
random_state = seed) fitted on the subset and evaluated on the fixed test ids of split_paper3.json; plus the
majority class of the test set. Writes <out>/curve_baseline.json (metrics only). Options: decision_model.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from decision_model import main  # noqa: E402

if __name__ == '__main__':
    main(sistema='baseline')
