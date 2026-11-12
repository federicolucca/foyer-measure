"""
azioni_politiche_v22 - v2.2 (XSci corrections): run on the 1,500-character benchmark corpus (email_labels_v*_corpus.jsonl);
cost L(c) = c*m + t + 0.5*d + 0.5*f + U*n, n = noise emails left in the inbox (U = argv[6], default 1);
the is-noise threshold of every system that returns P(noise) (B, Nimble v4-max, Jev, Tev1 0.8B) is chosen on the
training folds by minimising L(10) itself (table and curves); as a sensitivity it is also re-chosen for each c (L_c_soglia_per_c).

The agent's actions (draft, task, ignored) composed from the three switches under three policies, for every system of
the action table, and the cost L_S(c) for c = 1..20 (the cost curves).

Ground truth is the same for every policy: an email asks for a draft if needs-reply and not is-noise, for a task if
action-48h and not is-noise (labels composed with "noise wins", as in the paper). Only the composition of the
PREDICTIONS changes:
  noise_wins   (D4, the paper's): predicted is-noise -> ignored, no draft or task;
  action_wins  predicted needs-reply or action-48h win: draft/task, never ignored;
  consistency  is-noise together with needs-reply or action-48h = conflict: not ignored, no automatic action,
               flagged to the owner (cost 0.5 per flag, like a draft to read).
m = emails that asked for a draft or task and were ignored (false noise); t = tasks created without a true task;
d = drafts created without a true draft; f = conflicts flagged; n = true noise left in the inbox.

Per-row predictions: B (TF-IDF + LogReg, thread-grouped CV) from the saved .cv files; Nimble v4-max and Tev1 0.8B from
the .prob files; Gemma 4 26B and 31B from the .online files; Jev from the raw answers (v2: switches in choice format;
v3: noul). "+ threshold": is-noise = P(noise) >= t, t chosen on the thread-grouped training folds
(StratifiedGroupKFold 5, seed 0). B's probabilities are not saved and are recomputed with the same pipeline as the saved
run.

  python -I -X utf8 revision_v2.2/azioni_politiche_v22.py email_labels_v2_corpus.jsonl <data dir> email_labels_v2_corpus \
      <jev raw answers jsonl> azioni_politiche_v22_v2.json [U]
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cost_threshold as ct  # noqa: E402

BOOL = ('serve_risposta', 'azione_48h', 'e_rumore')
POLITICHE = ('noise_wins', 'action_wins', 'consistency')
C = list(range(1, 21))


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

    def carica(f):
        return json.loads((data / f).read_text(encoding='utf-8'))

    prev, prob_rumore = {}, {}
    cv = {c: carica(f'{base}.{c}.cv.risultati.json')['previsti'] for c in BOOL}
    prev['B'] = {c: dict(zip(map(str, cv[c]['id']), cv[c]['B'])) for c in BOOL}
    for nome, frag in (('Nimble v4-max', 'prob.nimble.nimble_v4_max'), ('Tev1 0.8B', 'prob.tev1.tev1_0.8b')):
        f = data / f'{base}.e_rumore.{frag}.risultati.json'
        if f.exists():
            prev[nome] = {c: {str(r['id']): r['scelta'] for r in carica(f'{base}.{c}.{frag}.risultati.json')['righe']} for c in BOOL}
            if True:
                prob_rumore[nome] = {str(r['id']): r['probabilita']['sì'] for r in carica(f.name)['righe']}
    for nome, frag in (('Gemma 4 26B', 'google_gemma_4_26b_a4b_it'), ('Gemma 4 31B', 'google_gemma_4_31b_it')):
        prev[nome] = {c: {str(k): v for k, v in carica(f'{base}.{c}.online.openrouter.{frag}.risultati.json')['previsti'].items()} for c in BOOL}
    jev, jev_p = {c: {} for c in BOOL}, {}
    for l in jev_raw.read_text(encoding='utf-8').splitlines():
        x = json.loads(l)
        if x.get('ok') and x['campo'] in BOOL and not x.get('tag'):
            a = x['risposta']['answers'][x['campo']]
            jev[x['campo']][str(x['id'])] = (a['choice'] == 'sì') if 'choice' in a else float(a['noul']) >= 0.5
            if x['campo'] == 'e_rumore':
                pj = (a.get('probabilities') or {}).get('sì') if 'choice' in a else a['noul']
                if pj is not None:
                    jev_p[str(x['id'])] = float(pj)
    prev['Jev'] = jev
    if len(jev_p) == len(ids):
        prob_rumore['Jev'] = jev_p

    # cost-chosen thresholds
    y = np.array(['sì' if v else 'no' for v in ru], dtype=object)
    g = np.array([ct.gruppo(r) for r in righe], dtype=object)
    X = np.array([ct.testo(r) for r in righe], dtype=object)
    esterni = ct.folds(y, g, np.arange(len(y)))
    U = float(sys.argv[6]) if len(sys.argv) > 6 else 1.0
    arr = lambda pv, c: np.array([si(pv[c][i]) for i in ids])

    def costo_soglia(p, ps, pa, idx, c, t):
        pn = p[idx] >= t
        return (c * int((serve[idx] & pn).sum()) + int((pa[idx] & ~pn & ~vero_task[idx]).sum())
                + 0.5 * int((ps[idx] & ~pn & ~vera_bozza[idx]).sum()) + U * int((ru[idx] & ~pn).sum()))

    def scegli(p, ps, pa, idx, c):
        cand = np.unique(np.concatenate([[0.0, 1.0001], p[idx]]))
        return float(min(cand, key=lambda t: (costo_soglia(p, ps, pa, idx, c, t), -t)))

    # B: P(noise) out-of-fold on each outer test part, inner out-of-fold on its training part
    pB_te, pB_tr = np.zeros(len(y)), []
    rifatto_b = np.zeros(len(y), bool)
    for tr, te in esterni:
        oof = np.zeros(len(y))
        for a_, b_ in ct.folds(y, g, tr):
            oof[b_] = ct.p_baseline(X, y, a_, b_)
        pB_tr.append(oof)
        pB_te[te] = ct.p_baseline(X, y, tr, te)
        rifatto_b[te] = pB_te[te] >= 0.5
    controllo_b = int(sum(si(prev['B']['e_rumore'][i]) != bool(v) for i, v in zip(ids, rifatto_b)))
    soglie = {}
    per_c = {}
    for nome in ['B'] + list(prob_rumore):
        ps, pa = arr(prev[nome], 'serve_risposta'), arr(prev[nome], 'azione_48h')
        p_all = None if nome == 'B' else np.array([prob_rumore[nome][i] for i in ids])
        per_c[nome] = []
        for c in C:
            pn = np.zeros(len(y), bool)
            ts = []
            for k, (tr, te) in enumerate(esterni):
                p_tr = pB_tr[k] if nome == 'B' else p_all
                t = scegli(p_tr, ps, pa, tr, c)
                ts.append(round(t, 4))
                pn[te] = (pB_te[te] if nome == 'B' else p_all[te]) >= t
            per_c[nome].append(pn)
            if c == 10:
                soglie[nome + ' + threshold'] = ts
        prev[nome + ' + threshold'] = {**prev[nome], 'e_rumore': {i: bool(v) for i, v in zip(ids, per_c[nome][C.index(10)])}}

    res = {'corpus': jsonl.name, 'n': len(righe), 'email_con_bozza_o_task': int(serve.sum()),
           'bozze_vere': int(vera_bozza.sum()), 'task_veri': int(vero_task.sum()),
           'controllo_B_rifatto_diverso_dal_salvato': controllo_b, 'U_rumore_in_inbox': U, 'soglie_c10_per_fold': soglie, 'c': C, 'sistemi': {}}
    riga10 = {}
    for nome, pv in prev.items():
        if not all(i in pv[c] for c in BOOL for i in ids):
            continue
        ps, pa, pn = (np.array([si(pv[c][i]) for i in ids]) for c in BOOL)
        r = {}
        for pol in POLITICHE:
            conflitto = pn & (ps | pa)
            if pol == 'noise_wins':
                ign, bozza, task, flag = pn, ps & ~pn, pa & ~pn, np.zeros(len(y), bool)
            elif pol == 'action_wins':
                ign, bozza, task, flag = pn & ~(ps | pa), ps, pa, np.zeros(len(y), bool)
            else:
                ign, bozza, task, flag = pn & ~(ps | pa), ps & ~pn, pa & ~pn, conflitto
            m, t, d, f = int((serve & ign).sum()), int((task & ~vero_task).sum()), int((bozza & ~vera_bozza).sum()), int(flag.sum())
            n = int((ru & ~ign).sum())
            costo = [c * m + t + 0.5 * d + 0.5 * f + U * n for c in C]
            extra = {}
            if pol == 'noise_wins' and nome.endswith(' + threshold'):   # sensitivity: threshold chosen again for each c
                base_ = nome[:-len(' + threshold')]
                costo_pc = []
                for k, c in enumerate(C):
                    pk = per_c[base_][k]
                    costo_pc.append(c * int((serve & pk).sum()) + int((pa & ~pk & ~vero_task).sum())
                                 + 0.5 * int((ps & ~pk & ~vera_bozza).sum()) + U * int((ru & ~pk).sum()))
                extra = {'L_c_soglia_per_c': costo_pc}
            r[pol] = {'falso_rumore': m, 'falso_rumore_ids': [i for i, x in zip(ids, serve & ign) if x], 'task_creati': int(task.sum()), 'task_inutili': t, 'bozze_create': int(bozza.sum()),
                      'bozze_inutili': d, 'segnalate': f, 'conflitti_previsti': int(conflitto.sum()),
                      'rumore_in_inbox': n, 'L_10': 10 * m + t + 0.5 * d + 0.5 * f + U * n, 'L_c': costo, **extra}
        res['sistemi'][nome] = r
        riga10[nome] = (10 * (serve & pn) + ((pa & ~pn) & ~vero_task) + 0.5 * ((ps & ~pn) & ~vera_bozza) + U * (ru & ~pn)).astype(float)
    # thread-cluster bootstrap (2,000 resamples of threads) of L(10) differences, noise wins, against B + threshold
    rng = np.random.default_rng(0)
    ug = np.unique(g)
    idx_g = {u: np.where(g == u)[0] for u in ug}
    camp = [np.concatenate([idx_g[u] for u in rng.choice(ug, len(ug))]) for _ in range(2000)]
    res['bootstrap_L10_thread_vs_B_threshold'] = {}
    for nome in riga10:
        if nome == 'B + threshold':
            continue
        dd = riga10['B + threshold'] - riga10[nome]
        st = np.array([dd[k].sum() for k in camp])
        res['bootstrap_L10_thread_vs_B_threshold'][nome] = {
            'diff_B_threshold_meno_sistema': float(dd.sum()), 'ic95': [float(np.percentile(st, 2.5)), float(np.percentile(st, 97.5))],
            'p_due_code': float(min(1.0, 2 * min((st >= 0).mean(), (st <= 0).mean())))}
    # exact McNemar on false noise (noise_wins), B + threshold vs Nimble, on the emails that asked for a draft or task
    from math import comb
    if 'B + threshold' in res['sistemi'] and 'Nimble v4-max' in res['sistemi']:
        a_ = set(res['sistemi']['B + threshold']['noise_wins']['falso_rumore_ids'])
        n_ = set(res['sistemi']['Nimble v4-max']['noise_wins']['falso_rumore_ids'])
        b, c = len(a_ - n_), len(n_ - a_)
        p = 1.0 if b + c == 0 else min(1.0, 2 * sum(comb(b + c, k) for k in range(min(b, c) + 1)) / 2 ** (b + c))
        res['mcnemar_falso_rumore_B_soglia_vs_Nimble'] = {'solo_B_soglia': b, 'solo_Nimble': c, 'p_esatto': round(p, 4)}
        print('McNemar false noise B+threshold vs Nimble:', res['mcnemar_falso_rumore_B_soglia_vs_Nimble'])
    # the cheapest system for each c (noise_wins) and where it changes
    nomi = list(res['sistemi'])
    migliore = [min(nomi, key=lambda s: res['sistemi'][s]['noise_wins']['L_c'][k]) for k in range(len(C))]
    res['piu_economico_noise_wins'] = migliore
    res['incroci_noise_wins'] = [{'c': C[k], 'da': migliore[k - 1], 'a': migliore[k]} for k in range(1, len(C)) if migliore[k] != migliore[k - 1]]
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
    print('wrote', out, '| recomputed B differs from the saved run on', controllo_b, 'rows')
    for nome, r in res['sistemi'].items():
        print(f'{nome:24s} ' + '  '.join(f"{p}: FR {r[p]['falso_rumore']}/{res['email_con_bozza_o_task']} t {r[p]['task_inutili']}/{r[p]['task_creati']} "
                                          f"d {r[p]['bozze_inutili']} f {r[p]['segnalate']} n {r[p]['rumore_in_inbox']} L10 {r[p]['L_10']}" for p in POLITICHE))
    print('incroci:', res['incroci_noise_wins'])
    for k, v in res['bootstrap_L10_thread_vs_B_threshold'].items():
        print('  boot', k, v)


if __name__ == '__main__':
    main()
