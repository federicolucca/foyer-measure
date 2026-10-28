"""
metrics — dalle risposte grezze di decision_bench.py al JSON dei risultati (niente numeri a mano).

  python metrics.py labels.jsonl raw_jev.jsonl results_jev.json [--prezzo-input 0.042] \
      [--ripetizione raw_jev_rep.jsonl] [--confronto raw_nimble.jsonl]

Per campo: accuracy con IC 95% bootstrap (2000), macro-F1, classe maggioritaria, latenza mediana e p95, errori API, token e costo
(token in ingresso × prezzo in $/M), confidenza media su corretti ed errati, ECE a 10 bin. Confidenza: il campo `confidence` per
`choice`, max(p, 1-p) per `noul`. Errori API: predizione = prima classe (come bench_email.py), contati.
--confronto: un secondo file grezzo; accordo riga per riga e McNemar esatto fra i due sistemi.
--ripetizione: le stesse righe richieste una seconda volta; stessa etichetta e risposta identica.
"""
import argparse
import json
import random
import statistics
from collections import Counter
from math import comb
from pathlib import Path

import systemone_email as so

CLASSI = {c: ([True, False] if c in so.BOOLEANI else list(so.CRITERI[c])) for c in so.CAMPI}


def jsonl(p):
    return [json.loads(l) for l in Path(p).read_text(encoding='utf-8').splitlines() if l.strip()]


def grezzi(p, tag=''):
    out = {}
    for d in jsonl(p):
        if d.get('tag', '') != tag:
            continue
        k = (d['id'], d['campo'])
        if d.get('ok') or k not in out:
            out[k] = d
    return out


def macro_f1(y, p, classi):
    f = []
    for c in classi:
        tp = sum(1 for a, b in zip(y, p) if a == c == b)
        fp = sum(1 for a, b in zip(y, p) if a != c and b == c)
        fn = sum(1 for a, b in zip(y, p) if a == c and b != c)
        f.append(2 * tp / (2 * tp + fp + fn) if tp else 0.0)
    return sum(f) / len(f)


def ece(conf, ok, bins=10):
    n, tot = len(conf), 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i, c in enumerate(conf) if lo < c <= hi or (b == 0 and c == 0)]
        if idx:
            tot += len(idx) / n * abs(sum(ok[i] for i in idx) / len(idx) - sum(conf[i] for i in idx) / len(idx))
    return tot


def boot(ok, b=2000, seed=0):
    rnd, n = random.Random(seed), len(ok)
    acc = sorted(sum(ok[rnd.randrange(n)] for _ in range(n)) / n for _ in range(b))
    return [round(acc[int(0.025 * b)], 4), round(acc[int(0.975 * b) - 1], 4)]


def mcnemar(b, c):
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def p95(xs):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(0.95 * (len(xs) - 1))))]


def confidenza(d):
    a = d.get('risposta') or {}
    if 'noul' in a:
        p = float(a['noul'])
        return max(p, 1 - p)
    if 'confidence' in a:
        return float(a['confidence'])
    pr = a.get('probabilities') or {}
    return max(pr.values()) if pr else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('labels', type=Path)
    ap.add_argument('raw', type=Path)
    ap.add_argument('out', type=Path)
    ap.add_argument('--prezzo-input', type=float, default=0.0, help='$ per milione di token in ingresso')
    ap.add_argument('--ripetizione', type=Path)
    ap.add_argument('--confronto', type=Path)
    a = ap.parse_args()
    righe = {r['id']: r for r in jsonl(a.labels) if r.get('labels')}
    g = grezzi(a.raw)
    altro = grezzi(a.confronto) if a.confronto else None
    res = {'labels': a.labels.name, 'raw': a.raw.name, 'n_righe': len(righe), 'campi': {}}
    for campo, classi in CLASSI.items():
        if not any(c == campo for _, c in g):
            continue
        y, p, conf, ok, lat, ids, err, tok = [], [], [], [], [], [], 0, 0
        for i, r in righe.items():
            vero = r['labels'].get(campo)
            if vero is None:
                continue
            d = g.get((i, campo))
            if d and d.get('ok'):
                lab, c = d['label'], confidenza(d)
                lat.append(d['lat_s'])
                tok += (d.get('usage') or {}).get('input_tokens', 0) or (d.get('usage') or {}).get('prompt_tokens', 0)
            else:
                lab, c, err = classi[0], None, err + 1
            y.append(vero); p.append(lab); conf.append(c); ok.append(int(lab == vero)); ids.append(i)
        cc = [(c, k) for c, k in zip(conf, ok) if c is not None]
        maj = Counter(y).most_common(1)[0]
        f = {'n': len(y), 'accuracy': round(sum(ok) / len(y), 4), 'ic95': boot(ok), 'macro_f1': round(macro_f1(y, p, classi), 4),
             'maggioranza': {'classe': str(maj[0]), 'accuracy': round(maj[1] / len(y), 4)},
             'latenza_s': {'mediana': round(statistics.median(lat), 3), 'p95': round(p95(lat), 3)} if lat else None,
             'errori_api': err, 'token_input': tok, 'costo_usd': round(tok * a.prezzo_input / 1e6, 6)}
        if cc:
            f['confidenza'] = {'media_corretti': round(statistics.mean([c for c, k in cc if k]), 4) if any(k for _, k in cc) else None,
                               'media_errati': round(statistics.mean([c for c, k in cc if not k]), 4) if any(not k for _, k in cc) else None,
                               'ece_10bin': round(ece([c for c, _ in cc], [k for _, k in cc]), 4)}
        if altro:
            com = [(i, yy, pp) for i, yy, pp in zip(ids, y, p) if altro.get((i, campo), {}).get('ok')]
            q = {i: altro[(i, campo)]['label'] for i, _, _ in com}
            b = sum(1 for i, yy, pp in com if pp == yy and q[i] != yy)
            c_ = sum(1 for i, yy, pp in com if q[i] == yy and pp != yy)
            f['confronto'] = {'n': len(com), 'accordo': round(sum(pp == q[i] for i, _, pp in com) / len(com), 4),
                              'acc_questo': round(sum(pp == yy for _, yy, pp in com) / len(com), 4),
                              'acc_altro': round(sum(q[i] == yy for i, yy, _ in com) / len(com), 4),
                              'solo_questo_giusto': b, 'solo_altro_giusto': c_, 'mcnemar_p_esatto': float(f'{mcnemar(b, c_):.3g}')}
        res['campi'][campo] = f
    if a.ripetizione:
        rep = grezzi(a.ripetizione, 'ripetizione')
        det = {}
        for (i, c), d in rep.items():
            x = g.get((i, c))
            if d.get('ok') and x and x.get('ok'):
                s = det.setdefault(c, {'n': 0, 'stessa_etichetta': 0, 'risposta_identica': 0})
                s['n'] += 1
                s['stessa_etichetta'] += d['label'] == x['label']
                s['risposta_identica'] += d.get('risposta') == x.get('risposta')
        res['determinismo'] = det
    a.out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
    for c, f in res['campi'].items():
        print(f"{c:15s} acc {f['accuracy']:.3f} {f['ic95']}  mF1 {f['macro_f1']:.3f}  maj {f['maggioranza']['accuracy']:.3f}  err {f['errori_api']}")
    print('scritto', a.out)


if __name__ == '__main__':
    main()
