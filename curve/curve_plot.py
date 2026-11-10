"""
curve_plot — the learning-curve figure (16:9, two panels), from the result JSON files only.

  python curve/curve_plot.py --results <results folder> --out <figure folder>

Reads <results>/curve_baseline.json and every <results>/curve_<model>_<seed>*.json (one file per run: several runs
of the same model and seed are merged by size, checking that test set and data coincide). No number typed by hand.

Two panels: type; mean of the three switches (serve_risposta, azione_48h, e_rumore). x axis = training labels
(log), with the zero-shot (size 0) drawn on the left as a separate point. Curves: TF-IDF baseline and the
fine-tuned models; band = min/max across seeds; horizontal line: majority class of the test set.
Writes curva.pdf (vector) and curva.png (1600×900). The paper's own figure is drawn by curve_tables.py.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

INTERRUTTORI = ['serve_risposta', 'azione_48h', 'e_rumore']
ORDINE = ['0', '50', '100', '200', '300', '400', 'all']

# validated categorical palette (blue, orange, teal); references in grey
COLORI = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100']
MARKER = ['o', 's', '^', 'D']
GRIGIO, TESTO, SECONDARIO, GRIGLIA = '#8a8984', '#1f1f1e', '#5f5e59', '#e6e5e0'
X_ZERO = 25   # position of the zero-shot point on the log axis


# ─── reading the result JSON files ───────────────────────────────────────────────────────────────

def carica(cartella: Path) -> dict:
    base = json.loads((cartella / 'curve_baseline.json').read_text(encoding='utf-8'))
    modelli: dict[str, dict] = {}
    for p in sorted(cartella.glob('curve_*.json')):
        if p.name == 'curve_baseline.json':
            continue
        j = json.loads(p.read_text(encoding='utf-8'))
        if j['test']['sha256_id'] != base['test']['sha256_id'] or j['dati_sha256'] != base['dati_sha256']:
            raise SystemExit(f'{p.name}: test set or data different from the baseline')
        m = modelli.setdefault(j['modello'], {'base_model': j['base_model'], 'iperparametri': j['iperparametri'],
                                              'semi': {}, 'file': []})
        m['file'].append(p.name)
        semi = m['semi'].setdefault(str(j['seme']), {})
        for t, rec in j['taglie'].items():
            if t in semi:
                raise SystemExit(f'{p.name}: size {t} of seed {j["seme"]} already present in another file')
            semi[t] = rec
    return {'baseline': base, 'modelli': modelli}


def n_etichette(t: str, pool: int) -> int:
    return pool if t == 'all' else int(t)


def valore(rec: dict, campo: str, metrica: str = 'accuracy') -> float:
    """One field, or 'interruttori' = mean of the three switches."""
    if campo == 'interruttori':
        return float(np.mean([rec['campi'][c][metrica] for c in INTERRUTTORI]))
    return rec['campi'][campo][metrica]


def curva(semi: dict, campo: str, metrica: str = 'accuracy', solo_semi=None) -> dict:
    """size → (mean, min, max, seeds used) across the seeds present (or only those given)."""
    out = {}
    for t in ORDINE:
        v = {s: valore(semi[s][t], campo, metrica) for s in semi if t in semi[s] and (not solo_semi or s in solo_semi)}
        if v:
            out[t] = (float(np.mean(list(v.values()))), min(v.values()), max(v.values()), sorted(v))
    return out


# ─── figure ──────────────────────────────────────────────────────────────────────────────────────

def etichetta_modello(nome: str, base: str) -> str:
    return f'LoRA {base.split("/")[-1]}'


def disegna(ax, dati, campo, titolo, pool):
    base = dati['baseline']
    serie = [('TF-IDF + logistic regression', base['semi'])] + \
            [(etichetta_modello(m, v['base_model']), v['semi']) for m, v in dati['modelli'].items()]
    for k, (nome, semi) in enumerate(serie):
        c = curva(semi, campo)
        punti = [t for t in ORDINE if t in c and t != '0']
        x = [n_etichette(t, pool) for t in punti]
        y = [c[t][0] for t in punti]
        lo = [c[t][1] for t in punti]
        hi = [c[t][2] for t in punti]
        ax.fill_between(x, lo, hi, color=COLORI[k], alpha=0.15, linewidth=0)
        ax.plot(x, y, color=COLORI[k], linewidth=2, marker=MARKER[k], markersize=7,
                markeredgecolor='white', markeredgewidth=1.5, label=nome, zorder=3)
        if '0' in c:
            ax.plot([X_ZERO], [c['0'][0]], color=COLORI[k], marker=MARKER[k], markersize=7, linestyle='none',
                    markeredgecolor='white', markeredgewidth=1.5, zorder=3)
        if punti:   # direct label to the right of the last point
            ax.annotate(nome.replace('LoRA ', ''), (x[-1], y[-1]), xytext=(8, 0), textcoords='offset points',
                        va='center', fontsize=9, color=TESTO)
    maj = base['maggioranza_test']
    v = float(np.mean([maj[c]['accuracy'] for c in INTERRUTTORI])) if campo == 'interruttori' else maj[campo]['accuracy']
    ax.axhline(v, color=GRIGIO, linestyle=':', linewidth=1.5)
    ax.annotate('majority class (test)', (X_ZERO * 0.9, v), xytext=(0, 4), textcoords='offset points',
                fontsize=8.5, color=SECONDARIO)
    ax.set_xscale('log')
    ticks = [X_ZERO, 50, 100, 200, 300, 400, pool]
    ax.set_xticks(ticks)
    ax.set_xticklabels(['0\n(zero-shot)'] + [str(t) for t in ticks[1:]])
    ax.minorticks_off()
    ax.set_xlim(X_ZERO * 0.8, pool * 1.6)
    ax.set_title(titolo, loc='left', fontsize=12, color=TESTO)
    ax.set_xlabel('training labels', color=SECONDARIO)
    ax.grid(axis='y', color=GRIGLIA, linewidth=0.8)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(GRIGLIA)
    ax.tick_params(colors=SECONDARIO)


def main():
    ap = argparse.ArgumentParser(description=__doc__.strip().split('\n')[0])
    ap.add_argument('--results', required=True, help='folder with curve_baseline.json and curve_<model>_<seed>.json')
    ap.add_argument('--out', required=True, help='output folder for curva.pdf and curva.png')
    a = ap.parse_args()
    dati = carica(Path(a.results))
    pool = dati['baseline']['semi']['7']['all']['etichette']
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'pdf.fonttype': 42})
    fig, axs = plt.subplots(1, 2, figsize=(16, 9), dpi=100, sharey=False)
    disegna(axs[0], dati, 'tipo', 'Type (6 classes): accuracy on the fixed test set', pool)
    disegna(axs[1], dati, 'interruttori', 'Mean of the three yes/no decisions: accuracy on the fixed test set', pool)
    axs[0].set_ylabel('accuracy', color=SECONDARIO)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc='upper center', ncol=len(l), frameon=False, fontsize=10)
    fig.text(0.01, 0.01, f'Fixed test set of {dati["baseline"]["test"]["righe"]} emails; bands = min/max across '
             'seeds.', fontsize=8.5, color=SECONDARIO)
    fig.tight_layout(rect=(0, 0.03, 1, 0.94))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / 'curva.pdf')
    fig.savefig(out / 'curva.png', dpi=100)
    print('wrote', out / 'curva.pdf', out / 'curva.png')


if __name__ == '__main__':
    main()
