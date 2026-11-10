"""
curve_split — learning curve: fixed test set, nested training subsets, typed examples.

  python curve/curve_split.py split    --data labels.jsonl --out <folder>
  python curve/curve_split.py examples --data labels.jsonl --split <folder>/split_paper3.json --out <file outside the repo>

- split: fixed test set = 20% of the threads, stratified by type (type of a thread = the most frequent among its
  rows), seed 7; the rest is the training pool. Nested subsets of the pool (each size contains the previous one),
  taken as whole threads in an order stratified by type, for seeds 7 and 11: 50, 100, 200, 300, 400, all.
  Writes split_paper3.json and curve_subsets.json with the numeric row ids only (refuses if an id is not an
  integer: the files may be published and must not contain addresses or subjects).
- examples: every row → five examples (one per field) in the decision format of Tev1/Jev: state = sender, subject,
  cleaned body at 1,500 characters; question and options with the paper's Italian definitions (systemone_email.py);
  answer = letter of the option. The file contains email text: it refuses to write it inside the repository.

Thread = the `thread` field of the corpus (649 threads over 722 rows in the paper). It is coarser than the
sender+subject grouping of decision_model.gruppo() (696 groups): each group lies in a single thread, so no pair of
emails of the same thread, under either definition, ends up split between train and test.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

_HERE = Path(__file__).resolve().parent
for _p in (_HERE, _HERE.parent):   # decision_model.py next to this file, systemone_email.py at the repository root
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import systemone_email as so  # noqa: E402
from decision_model import carica  # noqa: E402

SEME_TEST = 7
FRAZIONE_TEST = 0.2
SEMI_CURVA = (7, 11)
TAGLIE = (50, 100, 200, 300, 400, None)   # None = the whole pool
REPO = _HERE.parent


def thread_di(r: dict) -> str:
    return r['thread']


def per_thread(righe: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for r in sorted(righe, key=lambda r: r['id']):
        out.setdefault(thread_di(r), []).append(r)
    return out


def tipo_thread(rr: list[dict]) -> str:
    c = Counter(r['labels']['tipo'] for r in rr)
    return min(c, key=lambda k: (-c[k], k))


def split_test(righe: list[dict], seme: int = SEME_TEST, frazione: float = FRAZIONE_TEST):
    """For each type: threads in random order (seed), the first round(fraction · n_threads) go to the test set."""
    th = per_thread(righe)
    classi: dict[str, list[str]] = {}
    for t in sorted(th):
        classi.setdefault(tipo_thread(th[t]), []).append(t)
    rnd = random.Random(seme)
    test = set()
    for c in sorted(classi):
        lista = classi[c][:]
        rnd.shuffle(lista)
        test.update(lista[:round(frazione * len(lista))])
    test_ids = sorted(r['id'] for t in test for r in th[t])
    pool_ids = sorted(r['id'] for t in th if t not in test for r in th[t])
    return test_ids, pool_ids


def ordine_stratificato(righe: list[dict], seme: int) -> list[int]:
    """Ids of the pool rows in an order made of whole threads: at each step the type furthest below its share of
    the pool takes its next thread (threads of each type in random order with the seed). Every prefix is therefore
    stratified by type as far as whole threads allow; the sizes are prefixes, hence nested."""
    th = per_thread(righe)
    classi: dict[str, list[str]] = {}
    for t in sorted(th):
        classi.setdefault(tipo_thread(th[t]), []).append(t)
    rnd = random.Random(seme)
    for c in sorted(classi):
        rnd.shuffle(classi[c])
    tot = sum(len(v) for v in th.values())
    quota = {c: sum(len(th[t]) for t in classi[c]) / tot for c in classi}
    presi = {c: 0 for c in classi}
    pos = {c: 0 for c in classi}
    ordine: list[int] = []
    while len(ordine) < tot:
        aperte = [c for c in sorted(classi) if pos[c] < len(classi[c])]
        n_dopo = {c: len(ordine) + len(th[classi[c][pos[c]]]) for c in aperte}
        c = max(aperte, key=lambda c: quota[c] * n_dopo[c] - presi[c])
        t = classi[c][pos[c]]
        pos[c] += 1
        presi[c] += len(th[t])
        ordine += [r['id'] for r in th[t]]
    return ordine


def sha_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def conteggi(righe: list[dict], ids) -> dict:
    s = set(ids)
    rr = [r for r in righe if r['id'] in s]
    return {'righe': len(rr), 'thread': len({thread_di(r) for r in rr}),
            'per_campo': {c: dict(sorted(Counter(r['labels'][c] for r in rr).items())) for c in so.CAMPI}}


def solo_id(ids) -> list[int]:
    if not all(isinstance(i, int) and not isinstance(i, bool) for i in ids):
        raise SystemExit('An id is not an integer: the split files would carry more than ids; nothing written.')
    return list(ids)


def cmd_split(a):
    dati = Path(a.dati)
    righe = carica(dati, '1')
    test_ids, pool_ids = split_test(righe)
    # keys and descriptive values below are the published file format (Italian, kept as they are)
    split = {'descrizione': 'Paper 3, curva di apprendimento: test fisso e pool di training (id delle righe).',
             'dati_sha256': sha_file(dati), 'righe': len(righe), 'thread': len({thread_di(r) for r in righe}),
             'raggruppamento': 'campo thread del corpus v3', 'stratificazione': 'tipo del thread (più frequente)',
             'seme': SEME_TEST, 'frazione_test_thread': FRAZIONE_TEST,
             'test_ids': solo_id(test_ids), 'pool_ids': solo_id(pool_ids),
             'conteggi': {'test': conteggi(righe, test_ids), 'pool': conteggi(righe, pool_ids)}}
    pool = [r for r in righe if r['id'] in set(pool_ids)]
    sub = {'descrizione': 'Paper 3: sottoinsiemi annidati del pool (id delle righe); all = tutto il pool.',
           'dati_sha256': split['dati_sha256'], 'semi': {}, 'taglie_effettive': {}, 'conteggi': {}}
    for seme in SEMI_CURVA:
        ordine = ordine_stratificato(pool, seme)
        taglie = {('all' if n is None else str(n)): sorted(ordine if n is None else ordine[:n]) for n in TAGLIE}
        prec = set()
        for k, ids in taglie.items():   # nested by construction: checked anyway
            assert prec <= set(ids), k
            prec = set(ids)
        sub['semi'][str(seme)] = {k: solo_id(v) for k, v in taglie.items()}
        sub['taglie_effettive'][str(seme)] = {k: len(v) for k, v in taglie.items()}
        sub['conteggi'][str(seme)] = {k: conteggi(righe, v)['per_campo']['tipo'] for k, v in taglie.items()}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for nome, obj in (('split_paper3.json', split), ('curve_subsets.json', sub)):
        (out / nome).write_text(json.dumps(obj, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    print('test', split['conteggi']['test']['righe'], 'rows /', split['conteggi']['test']['thread'], 'threads;',
          'pool', split['conteggi']['pool']['righe'], 'rows; sizes', sub['taglie_effettive'])


def esempi(r: dict, campi=so.CAMPI, extra_stato: dict | None = None) -> list[dict]:
    """Five examples per email: chat messages (system + JSON decision) and answer = letter."""
    out = []
    for c in campi:
        dec = so.decisione(r['state'], c, extra_stato)
        out.append({'id': r['id'], 'campo': c, 'messages': so.messaggi_tev1(dec),
                    'completion': so.lettera(c, r['labels'][c]), 'key': r['labels'][c]})
    return out


def cmd_esempi(a):
    out = Path(a.out).resolve()
    if REPO in out.parents:
        raise SystemExit(f'{out} is inside the repository: the examples contain email text and stay outside.')
    righe = carica(Path(a.dati), '1')
    split = json.loads(Path(a.split).read_text(encoding='utf-8'))
    test = set(split['test_ids'])
    n = 0
    with out.open('w', encoding='utf-8', newline='\n') as f:
        for r in righe:
            for e in esempi(r):
                e['split'] = 'test' if r['id'] in test else 'pool'
                f.write(json.dumps(e, ensure_ascii=False) + '\n')
                n += 1
    print(f'{n} examples from {len(righe)} rows in {out}')


def main():
    ap = argparse.ArgumentParser(description=__doc__.strip().split('\n')[0])
    sp = ap.add_subparsers(dest='cmd', required=True)
    s = sp.add_parser('split', help='fixed test set and nested subsets (ids only)')
    s.add_argument('--data', '--dati', dest='dati', required=True, help='the labelled corpus (JSONL)')
    s.add_argument('--out', required=True, help='folder for split_paper3.json and curve_subsets.json')
    e = sp.add_parser('examples', aliases=['esempi'], help='typed examples (contain email text)')
    e.add_argument('--data', '--dati', dest='dati', required=True, help='the labelled corpus (JSONL)')
    e.add_argument('--split', required=True, help='split_paper3.json')
    e.add_argument('--out', required=True, help='JSONL file outside the repository')
    a = ap.parse_args()
    (cmd_split if a.cmd == 'split' else cmd_esempi)(a)


if __name__ == '__main__':
    main()
