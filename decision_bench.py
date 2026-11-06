"""
decision_bench — un sistema online sui cinque campi del paper, una domanda per chiamata; risposte grezze in un JSONL
(si riprende da dove si era interrotto). Le metriche le calcola metrics.py.

  python decision_bench.py run    labels.jsonl raw_jev.jsonl --backend jev
  python decision_bench.py run    labels.jsonl raw_nimble_noul.jsonl --backend nimble --formato noul --campi serve_risposta,azione_48h,e_rumore
  python decision_bench.py run    labels.jsonl raw_gemma.jsonl --backend openrouter --modello google/gemma-4-26b-a4b-it
  python decision_bench.py repeat labels.jsonl raw_jev_rep.jsonl --backend jev --n 100      # determinismo
  python decision_bench.py run    labels.jsonl raw_jeva.jsonl --backend jeva --jeva-cache hybrid   # v2.1, local GPU, 1 worker
Chiavi solo da variabili d'ambiente (vedi online_backends.py).
"""
from __future__ import annotations

import argparse
import json
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import online_backends as ob
import systemone_email as so


def carica(p: Path) -> list[dict]:
    return [r for r in (json.loads(l) for l in p.read_text(encoding='utf-8').splitlines() if l.strip()) if r.get('labels')]


def esegui(righe, out: Path, backend, modello, formato, campi, workers, tag=''):
    fatti = set()
    if out.exists():
        for l in out.read_text(encoding='utf-8').splitlines():
            d = json.loads(l)
            if d.get('ok'):
                fatti.add((d['id'], d['campo'], d.get('tag', '')))
    lavori = [(r, c) for r in righe for c in campi if (r['id'], c, tag) not in fatti]
    print(f'{len(lavori)} chiamate da fare ({len(fatti)} già fatte)', flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    lock, n = threading.Lock(), [0]

    def uno(rc):
        r, c = rc
        d = {'id': r['id'], 'campo': c, 'tag': tag, 'formato': formato if c in so.BOOLEANI else 'choice'}
        t = time.perf_counter()
        try:
            lab, info = ob.classify(backend, so.stato(r), c, modello, formato)
            d.update(ok=True, lat_s=time.perf_counter() - t, label=lab, risposta=info.get('raw'), usage=info.get('usage', {}))
        except Exception as e:
            d.update(ok=False, errore=f'{type(e).__name__}: {str(e)[:200]}')
        with lock:
            with out.open('a', encoding='utf-8') as f:
                f.write(json.dumps(d, ensure_ascii=False) + '\n')
            n[0] += 1
            if n[0] % 100 == 0 or not d['ok']:
                print(f'  {n[0]}/{len(lavori)}' + ('' if d['ok'] else f"  errore id={r['id']} {c}: {d['errore'][:100]}"), flush=True)

    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(uno, lavori))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['run', 'repeat'])
    ap.add_argument('jsonl', type=Path)
    ap.add_argument('out', type=Path)
    ap.add_argument('--backend', choices=list(ob.BACKENDS), required=True)
    ap.add_argument('--modello', default='')
    ap.add_argument('--formato', choices=['choice', 'noul'], default='choice', help='formato dei booleani')
    ap.add_argument('--campi', default=','.join(so.CAMPI))
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--n', type=int, default=100)
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--jeva-cache', choices=['hybrid', 'none'], default='hybrid',
                    help='jeva only: reuse option encodings across calls (hybrid) or encode them on every call (none)')
    a = ap.parse_args()
    if a.backend == 'jeva':
        import jeva_backend
        jeva_backend.configure(cache=a.jeva_cache)
        a.workers = 1   # one model in this process, on one GPU
    righe, campi = carica(a.jsonl), a.campi.split(',')
    if a.cmd == 'repeat':
        righe = random.Random(a.seed).sample(righe, min(a.n, len(righe)))
    esegui(righe, a.out, a.backend, a.modello, a.formato, campi, a.workers, 'ripetizione' if a.cmd == 'repeat' else '')


if __name__ == '__main__':
    main()
