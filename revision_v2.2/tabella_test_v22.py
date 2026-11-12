"""
tabella_test_v22 — v2.2 appendix table: accuracy on every field of v2 with 95% thread-cluster bootstrap intervals
(2,000 resamples of threads) and the exact McNemar test of each system against B, Holm-corrected over all 38 tests.
Also prints the thread-cluster intervals used in Table 2.

  python -I -X utf8 revision_v2.2/tabella_test_v22.py revisione_xsci.json <out tex>
"""
import json
import sys
from pathlib import Path

CAMPI = ('tipo', 'urgenza', 'serve_risposta', 'azione_48h', 'e_rumore')
ORDINE = ['B', 'Production rules', 'Tev1 0.8B', 'Tev1 4B', 'Nimble 9B Q8', 'Jeva-4B', 'Nimble v4-max', 'Jev', 'Qwen2.5-7B',
          'Gemma 4 26B', 'Gemma 4 31B']
NOME = {'B': 'B: TF-IDF + LogReg', 'Production rules': 'Stored production type', 'Gemma 4 26B': 'Gemma 4 26B-A4B'}


def f3(x):
    return f'{x:.3f}'.replace('0.', '.', 1) if x < 1 else f'{x:.2f}'


def main():
    r = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
    acc = r['accuracy_v2']
    mc = {(m['campo'], m['vs']): m for m in r['mcnemar_B_vs']}
    righe = [r'\begin{table*}[t]', r'\centering\small',
             r'\caption{All systems on v2 (231 emails): accuracy with 95\% thread-cluster bootstrap intervals (2,000 resamples of the '
             r'207 threads). $^{*}$: exact McNemar test against B significant at 0.05 after Holm correction over all ' + str(len(mc)) +
             r' comparisons in the table; $(n_{B}{:}n_{S})$ = emails only B / only the system got right. B is the seed-0 run '
             r'(over five seeds of the split, type $0.771\pm0.021$). Empty cells: field not asked.}',
             r'\label{tab:tests}', r'\resizebox{\textwidth}{!}{%', r'\begin{tabular}{lccccc}', r'\toprule',
             r'System & Type & Urgency & Needs-reply & Action-48h & Is-noise ' + '\\\\', r'\midrule']
    for s in ORDINE:
        celle = []
        for c in CAMPI:
            v = acc.get(s, {}).get(c)
            if not v:
                celle.append('')
                continue
            lo, hi = v['ic_thread']
            cella = f"{v['acc']:.3f} [{f3(lo)}, {f3(hi)}]"
            m = mc.get((c, s))
            if m:
                cella += f" ({m['solo_B_giusto']}:{m['solo_altro_giusto']})" + ('$^{*}$' if m['p_holm'] < 0.05 else '')
            celle.append(cella)
        righe.append(f"{NOME.get(s, s)} & " + ' & '.join(celle) + ' \\\\')
    righe += [r'\bottomrule', r'\end{tabular}}', r'\end{table*}', '']
    Path(sys.argv[2]).write_text('\n'.join(righe), encoding='utf-8')
    print('wrote', sys.argv[2])
    for s in ORDINE:
        print(s, {c: acc[s][c]['ic_thread'] for c in ('tipo', 'urgenza') if c in acc.get(s, {})})


if __name__ == '__main__':
    main()
