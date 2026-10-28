"""
cost_threshold — la soglia su P(e_rumore) scelta per costo, con le stesse regole per la baseline e per i sistemi zero-shot.

Fold esterni: StratifiedGroupKFold 5 per thread (shuffle, seme 0), stratificati su e_rumore. Nel train di ogni fold si sceglie
la soglia t che minimizza  C_FR × falso rumore + C_TI × trattata inutilmente  (a parità vince la t più alta) e la si applica
al fold di test. Falso rumore = email che chiedeva una bozza o un task (serve_risposta o azione_48h, e_rumore = no) prevista
rumore; trattata inutilmente = rumore vero prevista non rumore.
  - B (TF-IDF + regressione logistica bilanciata): le probabilità nel train vengono da una CV interna per thread (5 fold).
  - sistemi zero-shot: le loro probabilità (da un file grezzo di decision_bench.py: `noul`, o P(sì) di una `choice`).

  python cost_threshold.py labels.jsonl results_threshold.json --sistema jev=raw_jev.jsonl --sistema nimble=raw_nimble.jsonl
Nel paper (v2): con la stessa regola il modello decisionale ospitato scende sotto la baseline sul corpus non distorto,
non su quello avversario: la soglia appartiene alla regola di decisione, non a un classificatore.
"""
import argparse
import html
import json
import re
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline


def gruppo(r):
    """Thread = mittente + oggetto senza prefissi di risposta/inoltro."""
    s = r['state']
    oggetto = re.sub(r'^\s*((re|r|fwd?|i|inoltro)\s*:\s*)+', '', s.get('subject', '').lower()).strip()
    return f"{s.get('sender', '').strip().lower()}|{oggetto}"


def testo(r):
    """Il testo che vede la baseline: mittente, oggetto e corpo senza style/script, tag e URL."""
    s = r['state']
    corpo = html.unescape(re.sub(r'<(style|script)[^>]*>.*?</\1>', ' ', s.get('body', '') or '', flags=re.S | re.I))
    corpo = re.sub(r'<[^>]+>|https?://\S+', ' ', corpo)
    return f"{s.get('sender', '')} {s.get('subject', '')} {corpo}"


def folds(y, g, idx, k=5):
    return [(idx[a], idx[b]) for a, b in StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=0).split(idx, y[idx], g[idx])]


def soglia(p, serve, rumore, c_fr, c_ti):
    cand = np.unique(np.concatenate([[0.0, 1.0001], p]))
    costi = [(c_fr * int((serve & (p >= t)).sum()) + c_ti * int((rumore & (p < t)).sum()), -t) for t in cand]
    return float(cand[min(range(len(cand)), key=lambda i: costi[i])])


def p_baseline(X, y, tr, te):
    m = Pipeline([('tfidf', TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2), max_features=20000)),
                  ('lr', LogisticRegression(max_iter=2000, class_weight='balanced', random_state=0))]).fit(X[tr], y[tr])
    return m.predict_proba(X[te])[:, list(m.classes_).index('sì')]


def p_raw(path, ids):
    d = {}
    for l in Path(path).read_text(encoding='utf-8').splitlines():
        x = json.loads(l)
        if x.get('ok') and x['campo'] == 'e_rumore' and not x.get('tag'):
            a = x.get('risposta') or {}
            d[x['id']] = float(a['noul']) if 'noul' in a else float((a.get('probabilities') or {}).get('sì', float(x['label'])))
    return np.array([d.get(i, 0.0) for i in ids])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('labels', type=Path)
    ap.add_argument('out', type=Path)
    ap.add_argument('--sistema', action='append', default=[], help='nome=file grezzo di decision_bench.py')
    ap.add_argument('--costo-falso-rumore', type=float, default=10)
    ap.add_argument('--costo-trattata-inutilmente', type=float, default=1)
    a = ap.parse_args()
    righe = [r for r in (json.loads(l) for l in a.labels.read_text(encoding='utf-8').splitlines() if l.strip()) if r.get('labels')]
    ids = [r['id'] for r in righe]
    L = lambda c: np.array([bool(r['labels'][c]) for r in righe])
    sv, az, ru = L('serve_risposta'), L('azione_48h'), L('e_rumore')
    serve = ~ru & (sv | az)
    y = np.array(['sì' if v else 'no' for v in ru], dtype=object)
    g = np.array([gruppo(r) for r in righe], dtype=object)
    X = np.array([testo(r) for r in righe], dtype=object)
    cf, ct = a.costo_falso_rumore, a.costo_trattata_inutilmente
    esterni = folds(y, g, np.arange(len(y)))

    def conta(pr):
        fr, ti = int((serve & pr).sum()), int((ru & ~pr).sum())
        return {'falso_rumore': fr, 'email_con_bozza_o_task': int(serve.sum()), 'trattate_inutilmente': ti, 'costo': cf * fr + ct * ti}

    res = {'labels': a.labels.name, 'n': len(righe), 'thread': len(set(g)), 'costo': f'{cf} x falso rumore + {ct} x trattata inutilmente',
           'sistemi': {}}
    # B: probabilità fuori campione nel train con una CV interna, soglia scelta lì, applicata al test
    pred_b, arg_b = np.zeros(len(y), bool), np.zeros(len(y), bool)
    for tr, te in esterni:
        oof = np.zeros(len(y))
        for a_, b_ in folds(y, g, tr):
            oof[b_] = p_baseline(X, y, a_, b_)
        t = soglia(oof[tr], serve[tr], ru[tr], cf, ct)
        p_te = p_baseline(X, y, tr, te)
        pred_b[te], arg_b[te] = p_te >= t, p_te >= 0.5
    res['sistemi']['B: TF-IDF + LogReg'] = {'senza_soglia': conta(arg_b), 'con_soglia': conta(pred_b)}
    for s in a.sistema:
        nome, path = s.split('=', 1)
        p = p_raw(path, ids)
        pred = np.zeros(len(y), bool)
        for tr, te in esterni:
            pred[te] = p[te] >= soglia(p[tr], serve[tr], ru[tr], cf, ct)
        res['sistemi'][nome] = {'senza_soglia': conta(p >= 0.5), 'con_soglia': conta(pred)}
    a.out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
    for nome, v in res['sistemi'].items():
        s0, s1 = v['senza_soglia'], v['con_soglia']
        print(f"{nome:22s} senza: FR {s0['falso_rumore']}/{s0['email_con_bozza_o_task']} costo {s0['costo']} | con soglia: FR {s1['falso_rumore']} costo {s1['costo']}")
    print('scritto', a.out)


if __name__ == '__main__':
    main()
