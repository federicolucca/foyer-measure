"""
figura_costi_v22 - v2.2: the cost curves. Each L_S(c) is linear in c (slope = false noise m, intercept = t + 0.5 d + U n),
so lines are continuous and crossovers exact; "+ threshold" for every P(noise) system (Jev and Tev1 0.8B included).
Policy "noise wins", v2 and v3 in two panels, vector PDF. Marks where the cheapest system changes: among all systems
(filled) and among systems without a threshold (open). Also writes the crossovers to a JSON, from which the text takes
its numbers.

  python -I revision_v2.2/figura_costi_v22.py azioni_politiche_v22_v2.json azioni_politiche_v22_v3.json cost_curves.pdf crossovers.json
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt  # noqa: E402

# reference palette (validated, slots 1-6, light mode); one fixed colour per system
COLORE = {'B': '#2a78d6', 'Nimble v4-max': '#eb6834', 'Gemma 4 26B': '#1baf7a', 'Gemma 4 31B': '#eda100',
          'Tev1 0.8B': '#e87ba4', 'Jev': '#008300'}
ETICHETTA = {'B': 'B: TF-IDF + LogReg', 'B + threshold': 'B + threshold', 'Nimble v4-max': 'Nimble v4-max',
             'Nimble v4-max + threshold': 'Nimble + threshold', 'Gemma 4 26B': 'Gemma 4 26B', 'Gemma 4 31B': 'Gemma 4 31B',
             'Tev1 0.8B': 'Tev1 0.8B', 'Jev': 'Jev', 'Jev + threshold': 'Jev + threshold', 'Tev1 0.8B + threshold': 'Tev1 0.8B + threshold'}
TESTO, TESTO2, GRIGLIA = '#0b0b0b', '#52514e', '#e4e3df'


def retta(r, s):
    x = r['sistemi'][s]['noise_wins']
    m = x['falso_rumore']
    return m, x['L_10'] - 10 * m


def inviluppo(r, nomi, c0=1.0, c1=20.0):
    """Exact lower envelope of straight lines on [c0, c1]: list of crossovers (c, from, to, L)."""
    def migliore(c):
        return min(nomi, key=lambda s: (retta(r, s)[0] * c + retta(r, s)[1], retta(r, s)[0]))
    out, c, cur = [], c0, migliore(c0)
    while True:
        m0, q0 = retta(r, cur)
        cand = []
        for s in nomi:
            m, q = retta(r, s)
            if m < m0:
                x = (q - q0) / (m0 - m)
                if c < x < c1:
                    cand.append((x, m, s))
        if not cand:
            return out
        x, _, s = min(cand)
        out.append({'c': round(x, 2), 'da': cur, 'a': s, 'L': round(retta(r, s)[0] * x + retta(r, s)[1], 2)})
        c, cur = x, s


def main():
    v2, v3 = (json.loads(Path(p).read_text(encoding='utf-8')) for p in sys.argv[1:3])
    out_pdf, out_json = Path(sys.argv[3]), Path(sys.argv[4])
    plt.rcParams.update({'font.family': 'serif', 'font.size': 7.5, 'axes.edgecolor': TESTO2, 'axes.labelcolor': TESTO,
                         'xtick.color': TESTO2, 'ytick.color': TESTO2, 'pdf.fonttype': 42})
    fig, assi = plt.subplots(1, 2, figsize=(7.0, 2.7), sharey=True)
    incroci = {}
    for ax, (nome_c, r) in zip(assi, (('v2 (231 emails)', v2), ('v3 (722 emails, adversarial)', v3))):
        c = [1, 20]
        for s in r['sistemi']:
            base = s.replace(' + threshold', '')
            m, q = retta(r, s)
            ax.plot(c, [m * x + q for x in c], color=COLORE[base], lw=1.6,
                    ls='--' if s.endswith('threshold') else '-', label=ETICHETTA[s], solid_capstyle='round')
        tutti = inviluppo(r, list(r['sistemi']))
        senza = inviluppo(r, [s for s in r['sistemi'] if not s.endswith('threshold')])
        for p in tutti:
            ax.plot(p['c'], p['L'], 'o', ms=6, mfc=TESTO, mec='white', mew=1, zorder=5)
        for p in senza:
            ax.plot(p['c'], p['L'], 'o', ms=6, mfc='white', mec=TESTO, mew=1.2, zorder=5)
            ax.annotate(f"c={p['c']:g}", (p['c'], p['L']), xytext=(0, 7), textcoords='offset points', fontsize=6.5, color=TESTO2, ha='center')
        for p in tutti:
            ax.annotate(f"c={p['c']:g}", (p['c'], p['L']), xytext=(0, -12), textcoords='offset points', fontsize=6.5, color=TESTO, ha='center')
        incroci[nome_c] = {'tutti_i_sistemi': tutti, 'senza_soglie': senza,
                           'piu_economico_c1': min(r['sistemi'], key=lambda s: sum(retta(r, s))),
                           'piu_economico_c10': min(r['sistemi'], key=lambda s: r['sistemi'][s]['noise_wins']['L_10']),
                           'piu_economico_senza_soglie_c10': min([s for s in r['sistemi'] if not s.endswith('threshold')],
                                                                 key=lambda s: r['sistemi'][s]['noise_wins']['L_10'])}
        ax.set_yscale('log')
        ax.set_title(nome_c, fontsize=8, color=TESTO, loc='left')
        ax.set_xlabel('cost of a missed email, $c$')
        ax.set_xlim(1, 20)
        ax.set_xticks([1, 5, 10, 15, 20])
        ax.grid(True, which='major', color=GRIGLIA, lw=0.6)
        ax.grid(True, which='minor', axis='y', color=GRIGLIA, lw=0.3)
        for lato in ('top', 'right'):
            ax.spines[lato].set_visible(False)
    assi[0].set_ylabel('total cost $L_S(c)$ (log scale)')
    h, lab = assi[0].get_legend_handles_labels()
    fig.legend(h, lab, loc='lower center', ncol=5, frameon=False, fontsize=7, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf)
    fig.savefig(out_pdf.with_suffix('.png'), dpi=200)
    out_json.write_text(json.dumps(incroci, ensure_ascii=False, indent=1), encoding='utf-8')
    print('wrote', out_pdf, out_json)
    print(json.dumps(incroci, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
