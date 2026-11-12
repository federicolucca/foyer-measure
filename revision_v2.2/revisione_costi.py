"""
revisione_costi — XSci review, points #11, #12, #15: cost-chosen is-noise threshold for every system that returns a
probability (B, Nimble v4-max, Jev, Tev1 0.8B; the generative Gemma models return none), chosen inside thread-grouped CV
(a) with the paper's rule 10 x missed + 1 x handled needlessly, (b) per price c with the evaluated cost itself
L(c) = c*m + t + 0.5*d; plus thread-bootstrap intervals on L_S(10) differences.

  python -I -X utf8 revision_v2.2/revisione_costi.py email_labels_v2.jsonl <data dir> email_labels_v2_corpus <jev raw answers jsonl> revisione_costi_v2.json
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cost_threshold as ct  # noqa: E402

BOOL = ('serve_risposta', 'azione_48h', 'e_rumore')
C = list(range(1, 21))
RNG = np.random.default_rng(0)


def si(v):
    return v is True or str(v).strip().lower() in ('sì', 'si', 'true', '1')


def main():
    jsonl, data, base, jev_raw, out = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], Path(sys.argv[4]), Path(sys.argv[5])
    righe = [r for r in (json.loads(l) for l in jsonl.read_text(encoding='utf-8').split(chr(10)) if l.strip()) if r.get('labels')]
    ids = [str(r['id']) for r in righe]
    L = lambda c: np.array([si(r['labels'][c]) for r in righe])
    sv, az, ru = L('serve_risposta'), L('azione_48h'), L('e_rumore')
    vera_bozza, vero_task = sv & ~ru, az & ~ru
    serve = vera_bozza | vero_task
    J = lambda f: json.loads((data / f).read_text(encoding='utf-8'))

    prev, prob = {}, {}
    cv = {c: J(f'{base}.{c}.cv.risultati.json')['previsti'] for c in BOOL}
    prev['B'] = {c: dict(zip(map(str, cv[c]['id']), cv[c]['B'])) for c in BOOL}
    for nome, frag in (('Nimble v4-max', 'prob.nimble.nimble_v4_max'), ('Tev1 0.8B', 'prob.tev1.tev1_0.8b')):
        if (data / f'{base}.e_rumore.{frag}.risultati.json').exists():
            prev[nome] = {c: {str(r['id']): r['scelta'] for r in J(f'{base}.{c}.{frag}.risultati.json')['righe']} for c in BOOL}
            prob[nome] = {str(r['id']): r['probabilita']['sì'] for r in J(f'{base}.e_rumore.{frag}.risultati.json')['righe']}
    for nome, frag in (('Gemma 4 26B', 'google_gemma_4_26b_a4b_it'), ('Gemma 4 31B', 'google_gemma_4_31b_it')):
        prev[nome] = {c: {str(k): v for k, v in J(f'{base}.{c}.online.openrouter.{frag}.risultati.json')['previsti'].items()} for c in BOOL}
    jev, pj = {c: {} for c in BOOL}, {}
    for l in jev_raw.read_text(encoding='utf-8').splitlines():
        x = json.loads(l)
        if x.get('ok') and x['campo'] in BOOL and not x.get('tag'):
            a = x['risposta']['answers'][x['campo']]
            if 'choice' in a:
                jev[x['campo']][str(x['id'])] = a['choice'] == 'sì'
                p = (a.get('probabilities') or {}).get('sì')
            else:
                p = float(a['noul'])
                jev[x['campo']][str(x['id'])] = p >= 0.5
            if x['campo'] == 'e_rumore' and p is not None:
                pj[str(x['id'])] = float(p)
    prev['Jev'] = jev
    if len(pj) == len(ids):
        prob['Jev'] = pj

    y = np.array(['sì' if v else 'no' for v in ru], dtype=object)
    g = np.array([ct.gruppo(r) for r in righe], dtype=object)
    X = np.array([ct.testo(r) for r in righe], dtype=object)
    esterni = ct.folds(y, g, np.arange(len(y)))
    # B probabilities: out-of-fold on the test part, inner-CV out-of-fold on the train part (as azioni_politiche.py)
    pB_te = np.zeros(len(y))
    pB_tr = {}
    for k, (tr, te) in enumerate(esterni):
        oof = np.zeros(len(y))
        for a_, b_ in ct.folds(y, g, tr):
            oof[b_] = ct.p_baseline(X, y, a_, b_)
        pB_tr[k] = oof
        pB_te[te] = ct.p_baseline(X, y, tr, te)

    def arr(nome, c):
        return np.array([si(prev[nome][c][i]) for i in ids])

    def costi(ps, pa, pn, idx):
        ign, bozza, task = pn & ~np.zeros_like(pn), ps & ~pn, pa & ~pn
        m = (serve & ign)[idx].sum()
        t = (task & ~vero_task)[idx].sum()
        d = (bozza & ~vera_bozza)[idx].sum()
        return m, t, d

    U = float(sys.argv[6]) if len(sys.argv) > 6 else 0.0   # unit cost of a noise email NOT ignored (0 = eq. 1 of the paper)

    def n_rum(pn, idx):
        return int((ru & ~pn)[idx].sum())

    def scegli(p_tr, ps, pa, tr, regola, c):
        cand = np.unique(np.concatenate([[0.0, 1.0001], p_tr[tr]]))
        best = None
        for t in cand:
            pn = p_tr >= t
            m, tt, d = costi(ps, pa, pn, tr)
            v = (10 * m + int((ru & ~pn)[tr].sum())) if regola == 'paper' else (c * m + tt + 0.5 * d + U * n_rum(pn, tr))
            if best is None or (v, -t) < best[0]:
                best = ((v, -t), t)
        return best[1]

    sistemi = {}
    for nome in prev:
        ps, pa, pn0 = arr(nome, 'serve_risposta'), arr(nome, 'azione_48h'), arr(nome, 'e_rumore')
        m, t, d = costi(ps, pa, pn0, np.arange(len(y)))
        n0 = n_rum(pn0, np.arange(len(y)))
        sistemi[nome] = {'pn': pn0, 'L_c': [float(c * m + t + 0.5 * d + U * n0) for c in C], 'mtd': [int(m), int(t), int(d), n0]}
        if nome not in prob and nome != 'B':
            continue
        for regola in ('paper', 'per_c'):
            Lc, mtd = [], []
            for c in (C if regola == 'per_c' else [10]):
                pn = np.zeros(len(y), bool)
                for k, (tr, te) in enumerate(esterni):
                    if nome == 'B':
                        t_ = scegli(pB_tr[k], ps, pa, tr, regola, c)
                        pn[te] = pB_te[te] >= t_
                    else:
                        p = np.array([prob[nome][i] for i in ids])
                        t_ = scegli(p, ps, pa, tr, regola, c)
                        pn[te] = p[te] >= t_
                m, t, d = costi(ps, pa, pn, np.arange(len(y)))
                nn = n_rum(pn, np.arange(len(y)))
                if regola == 'paper':
                    sistemi[f'{nome} + threshold'] = {'pn': pn, 'L_c': [float(cc * m + t + 0.5 * d + U * nn) for cc in C], 'mtd': [int(m), int(t), int(d), nn]}
                else:
                    Lc.append(float(c * m + t + 0.5 * d + U * nn))
                    mtd.append([int(m), int(t), int(d), nn])
            if regola == 'per_c':
                sistemi[f'{nome} + threshold per c'] = {'L_c': Lc, 'mtd_per_c': mtd}

    res = {'U_rumore_non_ignorato': U, 'n': len(y), 'email_con_bozza_o_task': int(serve.sum()), 'c': C, 'sistemi': {}}
    for s, v in sistemi.items():
        res['sistemi'][s] = {k: v[k] for k in v if k != 'pn'}

    def inviluppo(nomi):
        best = [min(nomi, key=lambda s: sistemi[s]['L_c'][k]) for k in range(len(C))]
        return best, [{'c': C[k], 'da': best[k - 1], 'a': best[k]} for k in range(1, len(C)) if best[k] != best[k - 1]]
    for chiave, nomi in (('senza_soglia', [s for s in sistemi if 'threshold' not in s]),
                         ('soglia_regola_paper', [s for s in sistemi if 'per c' not in s]),
                         ('soglia_per_c', [s for s in sistemi if 'threshold' not in s or 'per c' in s])):
        best, inc = inviluppo(nomi)
        res[f'piu_economico_{chiave}'] = best
        res[f'incroci_{chiave}'] = inc

    # thread bootstrap of L_S(10) differences (rule of the paper), for the systems with a per-row composition
    def per_riga(s):
        ps, pa = arr(s.replace(' + threshold', ''), 'serve_risposta'), arr(s.replace(' + threshold', ''), 'azione_48h')
        pn = sistemi[s]['pn']
        return 10 * (serve & pn).astype(float) + ((pa & ~pn) & ~vero_task) + 0.5 * ((ps & ~pn) & ~vera_bozza) + U * (ru & ~pn)
    ug = np.unique(g)
    idx = {u: np.where(g == u)[0] for u in ug}
    campioni = [np.concatenate([idx[u] for u in RNG.choice(ug, len(ug))]) for _ in range(2000)]
    rif = 'B + threshold'
    res['bootstrap_L10_thread'] = {}
    for s in sistemi:
        if 'pn' not in sistemi[s] or s == rif:
            continue
        d = per_riga(rif) - per_riga(s)
        st = np.array([d[c].sum() for c in campioni])
        res['bootstrap_L10_thread'][f'{rif} - {s}'] = {'diff': float(d.sum()), 'ic95': [float(np.percentile(st, 2.5)), float(np.percentile(st, 97.5))],
                                                     'p_due_code': float(min(1, 2 * min((st >= 0).mean(), (st <= 0).mean())))}
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
    for s, v in res['sistemi'].items():
        print(f"{s:34s} L10={v['L_c'][9]:7.1f} mtd={v.get('mtd', '')} {v.get('mtd_per_c', [''] * 10)[9]}")
    for k in ('incroci_senza_soglia', 'incroci_soglia_regola_paper', 'incroci_soglia_per_c'):
        print(k, res[k])
    for k, v in res['bootstrap_L10_thread'].items():
        print(k, v)


if __name__ == '__main__':
    main()
