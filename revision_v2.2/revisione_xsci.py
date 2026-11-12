"""
revisione_xsci — recomputations requested by the XSci reviews of arXiv:2610.32955 v2.1 (12/11/2026).

Everything is read from the saved per-row result files (no model is re-run) plus the private v2/v3 label files.
Output: one JSON with aggregate numbers only (no email text, no addresses).

  python -I -X utf8 revision_v2.2/revisione_xsci.py email_labels_v2.jsonl email_labels_v3.jsonl <data dir> revisione_xsci.json
"""
import html
import json
import re
import sys
from collections import Counter, defaultdict
from math import comb
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import FeatureUnion, Pipeline

QUI = Path(__file__).resolve().parents[1]
F_ALL = ('tipo', 'urgenza', 'serve_risposta', 'azione_48h', 'e_rumore')
BOOL = ('serve_risposta', 'azione_48h', 'e_rumore')
RNG = np.random.default_rng(0)
NB = 2000


def si(v):
    return v is True or str(v).strip().lower() in ('sì', 'si', 'true', '1', 'yes')


def norm(campo, v):
    return si(v) if campo in BOOL else v


def carica_righe(p):
    return [r for r in (json.loads(l) for l in Path(p).read_text(encoding='utf-8').split(chr(10)) if l.strip()) if r.get('labels')]


def raw_choice(path, campo_ok=F_ALL):
    out = defaultdict(dict)
    for l in Path(path).read_text(encoding='utf-8').splitlines():
        x = json.loads(l)
        if not x.get('ok') or x.get('tag') or x['campo'] not in campo_ok:
            continue
        a = x['risposta']['answers'][x['campo']]
        out[x['campo']][str(x['id'])] = (a['choice'], a.get('probabilities'))
    return out


def previsioni_v2(data, righe):
    ids = [str(r['id']) for r in righe]
    base = 'email_labels_v2_corpus'
    J = lambda f: json.loads((data / f).read_text(encoding='utf-8'))
    P = defaultdict(dict)   # P[system][field] = {id: pred}
    for c in F_ALL:
        cv = J(f'{base}.{c}.cv.risultati.json')['previsti']
        P['B'][c] = dict(zip(map(str, cv['id']), cv['B']))
        if c in ('tipo', 'urgenza'):
            P['Production rules'][c] = dict(zip(map(str, cv['id']), cv['A']))
        for nome, frag in (('Nimble v4-max', 'prob.nimble.nimble_v4_max'), ('Tev1 0.8B', 'prob.tev1.tev1_0.8b')):
            f = data / f'{base}.{c}.{frag}.risultati.json'
            if f.exists():
                P[nome][c] = {str(r['id']): r['scelta'] for r in J(f.name)['righe']}
        for nome, frag in (('Gemma 4 26B', 'google_gemma_4_26b_a4b_it'), ('Gemma 4 31B', 'google_gemma_4_31b_it'),
                           ('Qwen2.5-7B', 'qwen_qwen_2.5_7b_instruct')):
            f = data / f'{base}.{c}.online.openrouter.{frag}.risultati.json'
            if f.exists():
                P[nome][c] = {str(k): v for k, v in J(f.name)['previsti'].items()}
    g = QUI / 'risultati' / 'grezzi'
    for nome, f1, f2 in (('Jev', 'jev_v2_raw.jsonl', 'jev_v2_booleani_choice_raw.jsonl'),
                         ('Jeva-4B', 'jeva_v2_raw.jsonl', 'jeva_v2_booleani_choice_raw.jsonl')):
        a, b = raw_choice(g / f1, ('tipo', 'urgenza')), raw_choice(g / f2, BOOL)
        for c, d in list(a.items()) + list(b.items()):
            P[nome][c] = {i: v[0] for i, v in d.items()}
        if nome == 'Jev':
            jev_prob = {i: v[1]['sì'] for i, v in b['e_rumore'].items()}
    gpu = json.loads((QUI / 'risultati_gpu' / 'gpu_risultati.json').read_text(encoding='utf-8'))
    for nome, k in (('Tev1 4B', 'tev1'), ('Nimble 9B Q8', 'nimble')):
        for c in ('tipo', 'urgenza'):
            d = gpu[f'{k}/{c}']
            P[nome][c] = dict(zip(map(str, d['ids']), d['pred']))
    # keep only complete fields
    for s in list(P):
        for c in list(P[s]):
            if not all(i in P[s][c] for i in ids):
                print('incomplete', s, c, sum(i in P[s][c] for i in ids))
                del P[s][c]
    return P, jev_prob


def corretti(P, righe, c, s):
    return np.array([norm(c, P[s][c][str(r['id'])]) == norm(c, r['labels'][c]) for r in righe])


def boot_ci(x, gruppi=None):
    x = np.asarray(x, float)
    if gruppi is None:
        st = [x[RNG.integers(0, len(x), len(x))].mean() for _ in range(NB)]
    else:
        ug = np.unique(gruppi)
        idx = {u: np.where(gruppi == u)[0] for u in ug}
        st = []
        for _ in range(NB):
            pick = RNG.choice(ug, len(ug))
            ii = np.concatenate([idx[u] for u in pick])
            st.append(x[ii].mean())
    return [round(float(np.percentile(st, 2.5)), 3), round(float(np.percentile(st, 97.5)), 3)]


def mcnemar(a, b):
    n01, n10 = int((a & ~b).sum()), int((~a & b).sum())
    chi = 0.0 if n01 + n10 == 0 else (abs(n01 - n10) - 1) ** 2 / (n01 + n10)
    k = min(n01, n10)
    p = 1.0 if n01 + n10 == 0 else min(1.0, 2 * sum(comb(n01 + n10, j) for j in range(k + 1)) / 2 ** (n01 + n10))
    return n01, n10, round(chi, 2), p


def cluster_diff_p(a, b, gruppi):
    """Thread-level paired test: bootstrap of threads on the accuracy difference; two-sided p from the sign."""
    d = a.astype(float) - b.astype(float)
    ug = np.unique(gruppi)
    idx = {u: np.where(gruppi == u)[0] for u in ug}
    st = np.array([d[np.concatenate([idx[u] for u in RNG.choice(ug, len(ug))])].mean() for _ in range(NB)])
    p = 2 * min((st <= 0).mean(), (st >= 0).mean())
    return round(float(d.mean()), 4), [round(float(np.percentile(st, 2.5)), 3), round(float(np.percentile(st, 97.5)), 3)], round(min(1.0, p), 4)


def holm(ps):
    o = sorted(range(len(ps)), key=lambda i: ps[i])
    adj, m, run = [0] * len(ps), len(ps), 0
    for r, i in enumerate(o):
        run = max(run, (m - r) * ps[i])
        adj[i] = min(1.0, run)
    return adj


def testo(r):
    s = r['state']
    corpo = html.unescape(re.sub(r'<(style|script)[^>]*>.*?</\1>', ' ', s.get('body', '') or '', flags=re.S | re.I))
    corpo = re.sub(r'<[^>]+>|https?://\S+', ' ', corpo)
    return f"{s.get('sender', '')} {s.get('subject', '')} {corpo}"


def corpo_pulito(r):
    s = r['state']
    c = html.unescape(re.sub(r'<(style|script)[^>]*>.*?</\1>', ' ', s.get('body', '') or '', flags=re.S | re.I))
    c = re.sub(r'<[^>]+>|https?://\S+', ' ', c)
    return re.sub(r'\s+', ' ', c).strip()


def modello_b(char=False):
    # the saved main run (bench_cv_pilota.py): word 1-2-grams only, max_features 20000
    w = TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2), max_features=20000)
    f = FeatureUnion([('w', w), ('c', TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), sublinear_tf=True))]) if char else w
    return Pipeline([('f', f), ('lr', LogisticRegression(max_iter=2000, class_weight='balanced', random_state=0))])


def gruppo(r):
    s = r['state']
    oggetto = re.sub(r'^\s*((re|r|fwd?|i|inoltro)\s*:\s*)+', '', s.get('subject', '').lower()).strip()
    return f"{s.get('sender', '').strip().lower()}|{oggetto}"


def mittente(r):
    m = re.search(r'<([^>]+)>', r['state'].get('sender', ''))
    return (m.group(1) if m else r['state'].get('sender', '')).strip().lower()


def main():
    p2, p3, data, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
    righe, righe3 = carica_righe(p2), carica_righe(p3)
    ids = [str(r['id']) for r in righe]
    thr = np.array([gruppo(r) for r in righe], dtype=object)
    R = {'n_v2': len(righe), 'thread_v2': len(set(thr)), 'thread_campo_jsonl': len({r.get('thread') for r in righe})}

    # ---- 0b accounting
    R['contabilita'] = {
        'per_fonte': dict(Counter(r.get('source') for r in righe)),
        'tipo': dict(Counter(r['labels']['tipo'] for r in righe)),
        'relazione': dict(Counter(r['labels'].get('relazione') for r in righe)),
        'urgenza': dict(Counter(r['labels']['urgenza'] for r in righe)),
        'split': dict(Counter(r.get('split') for r in righe)),
        'n_v3': len(righe3), 'thread_v3': len({gruppo(r) for r in righe3}), 'fonti_v3': dict(Counter(r.get('source') for r in righe3)),
    }

    # ---- 0a contacts with more than one relationship
    per_mitt = defaultdict(list)
    for r in righe:
        per_mitt[mittente(r)].append(r['labels'].get('relazione'))
    multi = {k: v for k, v in per_mitt.items() if len(v) > 1}
    per_mitt3 = defaultdict(list)
    for r in righe3:
        per_mitt3[mittente(r)].append(r['labels'].get('relazione'))
    multi3 = {k: v for k, v in per_mitt3.items() if len(v) > 1}
    R['relazione_per_contatto'] = {
        'v2_indirizzi': len(per_mitt), 'v2_indirizzi_con_piu_email': len(multi),
        'v2_indirizzi_con_piu_relazioni': sum(len(set(v)) > 1 for v in multi.values()),
        'v2_email_in_indirizzi_con_piu_relazioni': sum(len(v) for v in multi.values() if len(set(v)) > 1),
        'v3_indirizzi': len(per_mitt3), 'v3_indirizzi_con_piu_email': len(multi3),
        'v3_indirizzi_con_piu_relazioni': sum(len(set(v)) > 1 for v in multi3.values()),
        'v3_email_in_indirizzi_con_piu_relazioni': sum(len(v) for v in multi3.values() if len(set(v)) > 1),
    }
    # leave-one-out (email) vs leave-one-thread-out
    rel = [r['labels'].get('relazione') for r in righe]
    mitt = [mittente(r) for r in righe]
    magg = Counter(rel).most_common(1)[0][0]
    for nome, escludi in (('loo_email', lambda i, j: i == j), ('loo_thread', lambda i, j: thr[i] == thr[j])):
        cop = ok = ok_all = 0
        for i in range(len(righe)):
            altri = [rel[j] for j in range(len(righe)) if mitt[j] == mitt[i] and not escludi(i, j)]
            if altri:
                cop += 1
                pr = Counter(altri).most_common(1)[0][0]
                ok += pr == rel[i]
            else:
                pr = magg
            ok_all += pr == rel[i]
        R['relazione_per_contatto'][nome] = {'coperte': cop, 'acc_coperte': round(ok / cop, 4), 'acc_tutte_fallback': round(ok_all / len(righe), 4)}

    # ---- predictions, accuracies, cluster CIs, McNemar
    P, jev_prob = previsioni_v2(data, righe)
    acc = {}
    for s in P:
        for c in P[s]:
            k = corretti(P, righe, c, s)
            acc.setdefault(s, {})[c] = {'acc': round(k.mean(), 4), 'ic_righe': boot_ci(k), 'ic_thread': boot_ci(k, thr)}
    R['accuracy_v2'] = acc

    test = []
    for c in F_ALL:
        b = corretti(P, righe, c, 'B')
        for s in P:
            if s == 'B' or c not in P[s]:
                continue
            o = corretti(P, righe, c, s)
            n01, n10, chi, p = mcnemar(b, o)
            dm, ci, pc = cluster_diff_p(b, o, thr)
            test.append({'campo': c, 'vs': s, 'solo_B_giusto': n01, 'solo_altro_giusto': n10, 'chi2': chi, 'p_esatto': round(p, 5),
                         'diff': dm, 'ic_diff_thread': ci, 'p_cluster': pc})
    for t, a in zip(test, holm([t['p_esatto'] for t in test])):
        t['p_holm'] = round(a, 5)
    R['mcnemar_B_vs'] = test

    # ---- #8 bodies longer than 700 characters (as cleaned for the models)
    lung = np.array([len(corpo_pulito(r)) for r in righe])
    lunghe = lung > 700
    R['corpo_700'] = {'n_oltre_700': int(lunghe.sum()), 'n_oltre_1500': int((lung > 1500).sum()), 'per_sistema': {}}
    for s in P:
        for c in P[s]:
            k = corretti(P, righe, c, s)
            R['corpo_700']['per_sistema'].setdefault(s, {})[c] = {'acc_fino_700': round(k[~lunghe].mean(), 3), 'acc_oltre_700': round(k[lunghe].mean(), 3)}
    # majority-class accuracy in each stratum, for reference
    R['corpo_700']['maggioranza'] = {c: {'fino_700': round(max(Counter(str(righe[i]['labels'][c]) for i in np.where(~lunghe)[0]).values()) / (~lunghe).sum(), 3),
                                          'oltre_700': round(max(Counter(str(righe[i]['labels'][c]) for i in np.where(lunghe)[0]).values()) / lunghe.sum(), 3)}
                                     for c in F_ALL}

    # ---- #17 baseline with repeated stratified group k-fold
    X = np.array([testo(r) for r in righe], dtype=object)
    rip = {}
    for c, char in [(c, False) for c in F_ALL] + [('tipo', True)]:
        y = np.array([str(r['labels'][c]) for r in righe], dtype=object)
        accs, f1s = [], []
        from sklearn.metrics import f1_score
        for seme in range(5):
            pred = np.empty(len(y), dtype=object)
            for tr, te in StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seme).split(X, y, thr):
                pred[te] = modello_b(char).fit(X[tr], y[tr]).predict(X[te])
            accs.append((pred == y).mean())
            f1s.append(f1_score(y, pred, average='macro'))
        rip[c + ('_con_char' if char else '')] = {'acc': [round(a, 4) for a in accs], 'media': round(float(np.mean(accs)), 4), 'sd': round(float(np.std(accs, ddof=1)), 4),
                  'macro_f1_media': round(float(np.mean(f1s)), 4), 'macro_f1_sd': round(float(np.std(f1s, ddof=1)), 4)}
    R['B_kfold_ripetuti_5_semi'] = rip

    # ---- #6 Qwen calls
    q = {}
    for f in sorted(data.glob('*qwen*.risultati.json')):
        d = json.loads(f.read_text(encoding='utf-8'))
        q[f.name] = {'n': d.get('n'), 'previsti': len(d.get('previsti', {}) or {}), 'errori_api': d.get('errori_api'),
                     'fuori_vocab': sum(1 for v in (d.get('previsti') or {}).values() if v not in d.get('confusione_vero_su_previsto', {}) and v not in
                                        {x for row in d.get('confusione_vero_su_previsto', {}).values() for x in row})}
    R['qwen_chiamate'] = q

    # ---- #5 exact crossovers (noise wins), from the saved policy files
    inc = {}
    for nome in ('v2', 'v3'):
        r = json.loads((QUI / 'risultati' / f'azioni_politiche_{nome}.json').read_text(encoding='utf-8'))['sistemi']
        lin = {s: (v['noise_wins']['falso_rumore'], v['noise_wins']['task_inutili'] + 0.5 * v['noise_wins']['bozze_inutili']) for s, v in r.items()}
        cs = np.linspace(1, 20, 19001)
        best = [min(lin, key=lambda s: lin[s][0] * c + lin[s][1]) for c in cs]
        cambi = [{'c': round(float(cs[k]), 2), 'da': best[k - 1], 'a': best[k]} for k in range(1, len(cs)) if best[k] != best[k - 1]]
        senza = {s: v for s, v in lin.items() if 'threshold' not in s}
        best2 = [min(senza, key=lambda s: senza[s][0] * c + senza[s][1]) for c in cs]
        cambi2 = [{'c': round(float(cs[k]), 2), 'da': best2[k - 1], 'a': best2[k]} for k in range(1, len(cs)) if best2[k] != best2[k - 1]]
        inc[nome] = {'rette_m_e_intercetta': lin, 'incroci_tutti': cambi, 'incroci_senza_soglia': cambi2}
    R['incroci_continui'] = inc

    # ---- #11 thread-bootstrap CI on L_S(10) differences (v2, noise wins)
    pol = json.loads((QUI / 'risultati' / 'azioni_politiche_v2.json').read_text(encoding='utf-8'))
    a = P['Production rules']['tipo']
    R['produzione_tipo_previsto'] = dict(Counter(a.values()))

    out.write_text(json.dumps(R, ensure_ascii=False, indent=1), encoding='utf-8')
    print('written', out)


if __name__ == '__main__':
    main()
