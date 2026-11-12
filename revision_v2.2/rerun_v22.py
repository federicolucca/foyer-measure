"""
rerun_v22 — hosted runs for paper v2.2 (XSci corrections, point 1d):
  * negative control: ask the sender's relationship from the text of one email (8 classes), zero-shot, on the
    v2 corpus, with Gemma 4 26B (OpenRouter), Jev (TypeSafe) and Nimble v4-max (Bespoke) — same cleaned state,
    same question style and backends as bench_online_pilota.py;
  * one repeated run of Gemma 4 26B on type (spread of a single-run hosted estimate).
Keys are read from the environment only, never printed or written.

  python -I -X utf8 revision_v2.2/rerun_v22.py <corpus jsonl> <out dir> {rel-gemma|rel-jev|rel-nimble|tipo-gemma-rip}
"""
import json
import os
import sys
import time
from pathlib import Path

# The pilot harness the paper's hosted runs used (bench_online_pilota.py, bench_baseline_pilota.py); set its folder in
# FOYER_PILOT_DIR. online_backends.py in this repository is its public port. Keys come from the environment only
# (TYPESAFE_API_KEY, BESPOKE_API_KEY, OPENROUTER_API_KEY) and are never printed or written.
sys.path.insert(0, os.environ.get('FOYER_PILOT_DIR', str(Path(__file__).resolve().parent)))
os.environ.setdefault('SYSTEMONE_URL', 'https://api.bespokelabs.ai/v1/systemone')
os.environ.setdefault('SYSTEMONE_KEY', os.environ.get('BESPOKE_API_KEY', ''))

import bench_online_pilota as bo  # noqa: E402
from bench_baseline_pilota import metriche  # noqa: E402

# Relationship of the sender to the recipient (the owner); classes as in the label guidelines (D1/D3).
bo.CRITERI['relazione'] = {
    'cliente': 'chi dà lavoro al destinatario: un cliente effettivo, che paga, o un suo referente (non un cliente potenziale)',
    'fornitore': 'chi fornisce al destinatario un lavoro o un servizio su misura (consulente, professionista, subfornitore)',
    'socio': 'socio o collaboratore stretto del destinatario in un progetto o in un\'impresa comune',
    'famiglia': 'un familiare del destinatario',
    'istituzione': 'ente pubblico, università, ente di ricerca o associazione',
    'servizio': 'azienda o piattaforma di cui il destinatario usa il servizio (banca, software, negozio online, newsletter)',
    'monitoraggio': 'notifica automatica di un sistema gestito dal destinatario (deploy, build, server, console)',
    'me': 'il destinatario stesso: messaggio inviato da lui',
}
bo.DOMANDA['relazione'] = 'Che relazione ha il mittente con il destinatario di questa email?'

LAVORI = {
    'rel-gemma': ('relazione', 'openrouter', 'google/gemma-4-26b-a4b-it'),
    'rel-jev': ('relazione', 'jev', 'jev-latest'),
    'rel-nimble': ('relazione', 'systemone', 'nimble-v4-max'),
    'tipo-gemma-rip': ('tipo', 'openrouter', 'google/gemma-4-26b-a4b-it'),
}


def main():
    jsonl, outdir, lavoro = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
    campo, backend, modello = LAVORI[lavoro]
    fn = bo.BACKEND[backend][0]
    righe = [r for r in (json.loads(l) for l in jsonl.read_text(encoding='utf-8').split(chr(10)) if l.strip()) if r.get('labels')]
    y = [r['labels'][campo] for r in righe]
    pred, err, costo, lat = [], [], 0.0, []
    for r in righe:
        t = time.perf_counter()
        try:
            p, c = fn(campo, r, modello)
            costo += c
        except Exception as e:  # noqa: BLE001 — counted as a wrong (empty) prediction, as in bench_online_pilota
            p = ''
            err.append(f'id {r["id"]}: {str(e)[:80]}')
        pred.append(p)
        lat.append(round(1000 * (time.perf_counter() - t), 1))
        if len(pred) == 3 and len(err) == 3:
            raise SystemExit('first 3 calls failed: ' + '; '.join(err))
    maggioranza = max(set(y), key=y.count)
    res = {'lavoro': lavoro, 'campo': campo, 'backend': backend, 'modello': modello, 'n': len(righe),
           **metriche(y, pred), 'maggioranza': {'classe': maggioranza, 'accuracy': round(y.count(maggioranza) / len(y), 4)},
           'errori_api': len(err), 'esempi_errori': err[:5], 'latenza_ms_mediana': sorted(lat)[len(lat) // 2],
           'costo_usd_totale': round(costo, 6), 'token_input_totali': bo.TOKEN['input'],
           'criteri': bo.CRITERI[campo] if campo == 'relazione' else None, 'domanda': bo.DOMANDA[campo],
           'previsti': {str(r['id']): p for r, p in zip(righe, pred)}, 'data': time.strftime('%Y-%m-%d %H:%M')}
    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / f'{jsonl.stem}.{lavoro}.risultati.json'
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
    print(f"{lavoro}: n={res['n']} acc={res['accuracy']} macroF1={res['macro_f1']} maj={res['maggioranza']} err={len(err)} cost=${res['costo_usd_totale']}")


if __name__ == '__main__':
    main()
