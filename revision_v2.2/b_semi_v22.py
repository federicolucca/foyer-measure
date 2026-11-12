"""
b_semi_v22 — v2.2 (XSci #17): the trained baseline B over 5 seeds of thread-grouped StratifiedGroupKFold(5), every field,
v2 and v3, with the saved run's features (word 1-2-grams) and with word + character 3-5-grams; mean, sd, min, max.

  python -I -X utf8 revision_v2.2/b_semi_v22.py email_labels_v2_corpus.jsonl email_labels_v3_corpus.jsonl b_semi_v22.json
"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from revisione_xsci import carica_righe, gruppo, modello_b, testo  # noqa: E402

CAMPI = ('tipo', 'urgenza', 'serve_risposta', 'azione_48h', 'e_rumore')


def main():
    out = {}
    for nome, p in (('v2', sys.argv[1]), ('v3', sys.argv[2])):
        righe = carica_righe(p)
        X = np.array([testo(r) for r in righe], dtype=object)
        g = np.array([gruppo(r) for r in righe], dtype=object)
        out[nome] = {}
        for char in (False, True):
            for c in CAMPI:
                y = np.array([str(r['labels'][c]) for r in righe], dtype=object)
                acc, f1 = [], []
                for seme in range(5):
                    pred = np.empty(len(y), dtype=object)
                    for tr, te in StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seme).split(X, y, g):
                        pred[te] = modello_b(char).fit(X[tr], y[tr]).predict(X[te])
                    acc.append(float((pred == y).mean()))
                    f1.append(float(f1_score(y, pred, average='macro')))
                k = f"{c}{'_word_char' if char else '_word'}"
                out[nome][k] = {'acc': [round(a, 4) for a in acc], 'media': round(float(np.mean(acc)), 4), 'sd': round(float(np.std(acc, ddof=1)), 4),
                                'min': round(min(acc), 4), 'max': round(max(acc), 4),
                                'macro_f1_media': round(float(np.mean(f1)), 4), 'macro_f1_sd': round(float(np.std(f1, ddof=1)), 4)}
                print(nome, k, out[nome][k]['media'], '+-', out[nome][k]['sd'], 'F1', out[nome][k]['macro_f1_media'], flush=True)
    Path(sys.argv[3]).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()
