"""
project_bench — a quale progetto appartiene un'email: una descrizione per progetto (≤120 caratteri) contro i soli nomi.

Le descrizioni stanno in un Markdown con una tabella `| slug | descrizione |`, scritte dal proprietario (cosa è il progetto e di
cosa parlano le sue email, senza nomi di persone o aziende); una riga `nessuno` descrive l'opzione «nessuno». Righe valutate:
quelle con un progetto etichettato; un progetto fuori elenco vale «nessuno». Il file delle descrizioni resta privato.

  python project_bench.py labels.jsonl descrizioni.md results_project.json --backend jev --variante desc
  python project_bench.py labels.jsonl descrizioni.md results_project.json --backend openrouter --modello google/gemma-4-26b-a4b-it --variante nomi
Lezione dal paper: con descrizioni lunghe /v1/systemone fallisce in silenzio; 120 caratteri e 19 opzioni funzionano.
"""
import argparse
import json
import os
import re
import statistics
import time
from collections import Counter
from pathlib import Path

import online_backends as ob
import systemone_email as so


def descrizioni(md: Path) -> dict:
    d = {}
    for l in md.read_text(encoding='utf-8').splitlines():
        m = re.match(r'\|\s*([a-z0-9][\w-]*)\s*\|\s*(.+?)\s*\|\s*$', l)
        if m and m.group(1) != 'slug':
            d[m.group(1)] = m.group(2)
    return d


def criteri(desc: dict, variante: str) -> dict:
    prog = [p for p in desc if p != so.NESSUNO]
    if variante == 'desc':
        c = {p: desc[p][:so.DESC_MAX] for p in prog}
        c[so.NESSUNO] = desc.get(so.NESSUNO, "l'email non riguarda nessuno dei progetti elencati")[:so.DESC_MAX]
    else:
        c = {p: f"l'email riguarda il progetto o l'area «{p}»" for p in prog}
        c[so.NESSUNO] = "l'email non riguarda nessuno dei progetti elencati"
    return c


def chiedi(backend, modello, stato, crit):
    if backend == 'openrouter':
        prompt = (so.DOMANDA['progetto'] + ' Scegli esattamente una etichetta fra queste:\n'
                  + '\n'.join(f'- {k}: {v}' for k, v in crit.items()) + '\nRispondi SOLO con JSON: {"progetto": "<etichetta>"}')
        out = ob._with_retry(lambda: ob._post('https://openrouter.ai/api/v1/chat/completions',
                             {'model': modello, 'max_tokens': 60, 'temperature': 0, 'response_format': {'type': 'json_object'},
                              'messages': [{'role': 'system', 'content': prompt},
                                           {'role': 'user', 'content': json.dumps(stato, ensure_ascii=False)}]},
                             {'Authorization': 'Bearer ' + ob._key('OPENROUTER_API_KEY')}))
        t = out['choices'][0]['message']['content'] or ''
        e = str(json.loads(t[t.find('{'):t.rfind('}') + 1]).get('progetto', '')).strip().lower()
        if e not in crit:
            raise RuntimeError(f'etichetta fuori vocabolario: {e[:40]!r}')
        return e
    if backend == 'systemone':
        url, key, mod = os.environ['SYSTEMONE_URL'], os.environ.get('SYSTEMONE_KEY', ''), modello
    else:
        url, mod, kname = ob.SERVER[backend]
        key, mod = ob._key(kname), modello or mod
    q = {'type': 'choice', 'instructions': so.DOMANDA['progetto'], 'criteria': crit}
    out = ob._with_retry(lambda: ob._post(url, {'model': mod, 'state': stato, 'questions': {'progetto': q}},
                                          {'Authorization': f'Bearer {key}'} if key else {}))
    return out['answers']['progetto']['choice']


def macro_f1(y, p):
    f = []
    for c in sorted(set(y) | set(p)):
        tp = sum(1 for a, b in zip(y, p) if a == c == b)
        fp = sum(1 for a, b in zip(y, p) if a != c and b == c)
        fn = sum(1 for a, b in zip(y, p) if a == c and b != c)
        f.append(2 * tp / (2 * tp + fp + fn) if tp else 0.0)
    return sum(f) / len(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('labels', type=Path)
    ap.add_argument('md', type=Path)
    ap.add_argument('out', type=Path)
    ap.add_argument('--backend', choices=['jev', 'nimble', 'systemone', 'openrouter'], required=True)
    ap.add_argument('--modello', default='')
    ap.add_argument('--variante', choices=['desc', 'nomi'], default='desc')
    a = ap.parse_args()
    desc = descrizioni(a.md)
    crit = criteri(desc, a.variante)
    righe = []
    for l in a.labels.read_text(encoding='utf-8').splitlines():
        if l.strip():
            r = json.loads(l)
            p = ((r.get('labels') or {}).get('progetto') or (r.get('labels') or {}).get('project') or '').strip()
            if p:
                righe.append((r, p if p in desc and p != so.NESSUNO else so.NESSUNO))
    y, pred, lat, err = [], [], [], 0
    for r, v in righe:
        t = time.perf_counter()
        try:
            pred.append(chiedi(a.backend, a.modello, so.stato(r), crit))
            lat.append(time.perf_counter() - t)
        except Exception:
            pred.append(''); err += 1
        y.append(v)
    n_p = [i for i, p in enumerate(pred) if p == so.NESSUNO]
    n_v = [i for i, v in enumerate(y) if v == so.NESSUNO]
    maj = Counter(y).most_common(1)[0]
    res = {'n': len(y), 'opzioni': len(crit), 'variante': a.variante, 'backend': a.backend, 'modello': a.modello or None,
           'maggioranza': {'classe': maj[0], 'accuracy': round(maj[1] / len(y), 4)},
           'accuracy': round(sum(p == v for p, v in zip(pred, y)) / len(y), 4), 'macro_f1': round(macro_f1(y, pred), 4),
           'nessuno': {'previsti': len(n_p), 'veri': len(n_v),
                       'precision': round(sum(y[i] == so.NESSUNO for i in n_p) / max(1, len(n_p)), 4),
                       'recall': round(sum(pred[i] == so.NESSUNO for i in n_v) / max(1, len(n_v)), 4)},
           'errori_api': err, 'latenza_s_mediana': round(statistics.median(lat), 3) if lat else None}
    a.out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
    print(json.dumps({k: res[k] for k in ('n', 'opzioni', 'accuracy', 'macro_f1', 'errori_api')}), '->', a.out)


if __name__ == '__main__':
    main()
