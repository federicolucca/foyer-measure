"""
kappa_multi — accordo di uno o più annotatori ciechi con il riferimento, fra loro e κ di Fleiss, con IC 95% bootstrap.

  python kappa_multi.py reference.json results_kappa.json annotatore1.csv [annotatore2.csv]

reference.json: {id: {tipo, urgenza, serve_risposta, azione_48h, e_rumore}} (booleani true/false).
CSV: le colonne del kit cieco (id, tipo, urgenza, serve_risposta, azione_48h, e_rumore, dubbio); booleani 1/0 o sì/no.
Per ogni annotatore contro il riferimento: accordo, κ di Cohen [IC 95%] (stessa normalizzazione e bootstrap di kappa.py).
Con due annotatori: κ di Cohen fra i due e κ di Fleiss a tre (riferimento + due). Nel JSON solo conteggi.
"""
import collections
import csv
import json
import random
import sys
from pathlib import Path

CAMPI = ['tipo', 'urgenza', 'serve_risposta', 'azione_48h', 'e_rumore']
BOOL = {'serve_risposta', 'azione_48h', 'e_rumore'}
B = 2000


def norm(campo, v):
    v = str(v).strip().lower()
    if campo in BOOL:
        return '1' if v in ('1', 'si', 'sì', 'yes', 'true', 'x') else '0'
    return v


def cohen(a, b):
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = collections.Counter(a), collections.Counter(b)
    pe = sum(ca[k] * cb[k] for k in set(a) | set(b)) / (n * n)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def fleiss(colonne):
    n, m = len(colonne[0]), len(colonne)
    tot, P = collections.Counter(), []
    for i in range(n):
        cnt = collections.Counter(c[i] for c in colonne)
        tot.update(cnt)
        P.append((sum(v * v for v in cnt.values()) - m) / (m * (m - 1)))
    pe = sum((v / (n * m)) ** 2 for v in tot.values())
    return (sum(P) / n - pe) / (1 - pe) if pe < 1 else 1.0


def boot(fn, colonne, rng):
    n = len(colonne[0])
    v = []
    for _ in range(B):
        idx = [rng.randrange(n) for _ in range(n)]
        cols = [[c[i] for i in idx] for c in colonne]
        v.append(fn(*cols) if fn is cohen else fn(cols))
    v.sort()
    return [round(v[int(0.025 * B)], 3), round(v[int(0.975 * B)], 3)]


def carica(p, ref):
    out = {}
    for r in csv.DictReader(open(p, encoding='utf-8')):
        i = str(r['id'])
        if i in ref:
            out[i] = {c: norm(c, r[c]) if str(r.get(c, '')).strip() else None for c in CAMPI}
    return out


def main():
    ref_raw = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
    ref = {str(k): {c: (('1' if v[c] else '0') if c in BOOL else str(v[c])) for c in CAMPI} for k, v in ref_raw.items()}
    annot = [carica(p, ref) for p in sys.argv[3:]]
    res = {'n_riferimento': len(ref), 'annotatori': [], 'fra_annotatori': None, 'fleiss_tre': None}
    for d in annot:
        rng, r = random.Random(0), {}
        for c in CAMPI:
            ids = [i for i in d if d[i][c] is not None]
            a, b = [ref[i][c] for i in ids], [d[i][c] for i in ids]
            r[c] = {'n': len(ids), 'accordo': round(sum(x == y for x, y in zip(a, b)) / len(ids), 3),
                    'kappa': round(cohen(a, b), 3), 'ic95': boot(cohen, [a, b], rng)}
        res['annotatori'].append(r)
    if len(annot) == 2:
        d1, d2 = annot
        res['fra_annotatori'], res['fleiss_tre'] = {}, {}
        rng = random.Random(0)
        for c in CAMPI:
            ids = [i for i in d1 if i in d2 and d1[i][c] is not None and d2[i][c] is not None]
            a, b, k = [d1[i][c] for i in ids], [d2[i][c] for i in ids], [ref[i][c] for i in ids]
            res['fra_annotatori'][c] = {'n': len(ids), 'kappa': round(cohen(a, b), 3), 'ic95': boot(cohen, [a, b], rng)}
            res['fleiss_tre'][c] = {'n': len(ids), 'kappa': round(fleiss([k, a, b]), 3), 'ic95': boot(fleiss, [k, a, b], rng)}
    Path(sys.argv[2]).write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
    for j, r in enumerate(res['annotatori'], 1):
        print(f'annotatore {j}: ' + '  '.join(f"{c} κ {v['kappa']:.2f} {v['ic95']}" for c, v in r.items()))
    if res['fleiss_tre']:
        print('fra i due: ' + '  '.join(f"{c} κ {v['kappa']:.2f}" for c, v in res['fra_annotatori'].items()))
        print('Fleiss a tre: ' + '  '.join(f"{c} κ {v['kappa']:.2f}" for c, v in res['fleiss_tre'].items()))
    print('scritto', sys.argv[2])


if __name__ == '__main__':
    main()
