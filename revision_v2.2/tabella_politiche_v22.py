"""
tabella_politiche_v22 - the policy table (tab_policy.tex) from azioni_politiche_v22_v{2,3}.json (no hand-typed numbers).

  python -I revision_v2.2/tabella_politiche_v22.py azioni_politiche_v22_v2.json azioni_politiche_v22_v3.json tab_policy.tex
"""
import json
import sys
from pathlib import Path

ORDINE = ['B', 'B + threshold', 'Nimble v4-max', 'Nimble v4-max + threshold', 'Jev', 'Jev + threshold', 'Gemma 4 26B', 'Gemma 4 31B', 'Tev1 0.8B', 'Tev1 0.8B + threshold']
NOME = {'B': 'B: TF-IDF + LogReg', 'B + threshold': 'B + threshold', 'Nimble v4-max': 'Nimble v4-max',
        'Nimble v4-max + threshold': 'Nimble + threshold', 'Jev': 'Jev', 'Gemma 4 26B': 'Gemma 4 26B-A4B',
        'Gemma 4 31B': 'Gemma 4 31B', 'Tev1 0.8B': 'Tev1 0.8B', 'Jev + threshold': 'Jev + threshold',
        'Tev1 0.8B + threshold': 'Tev1 0.8B + threshold'}


def num(x):
    return f'{x:g}'


def main():
    v2, v3 = (json.loads(Path(p).read_text(encoding='utf-8')) for p in sys.argv[1:3])
    r = [r'\begin{table*}[t]', r'\centering\small',
         r'\caption{Composition policies for the three switches, v2 and v3. The truth is fixed (labels composed with \emph{is-noise} wins); '
         r'only the composition of the predictions changes. \emph{Noise wins}: a predicted is-noise ignores the email (the policy of '
         r'Table~\ref{tab:actions}); \emph{action wins}: a predicted reply or task overrides is-noise; \emph{consistency}: is-noise together '
         r'with a reply or a task is not ignored and not acted on, but flagged to the owner. FR = false noise (emails that needed a reply or a '
         r'task and were ignored), t = useless tasks, d = useless drafts, n = noise emails left in the inbox, f = flagged conflicts, '
         r'$L_{10}$ = $L_S(10)$ of Eq.~(\ref{eq:cost}) with a flag costing 0.5. Thresholds and Jev formats as in Figure~\ref{fig:cost}; '
         r'Gemma returns no probability and is not thresholdable as served.}',
         r'\label{tab:policy}', r'\resizebox{\textwidth}{!}{%', r'\begin{tabular}{l rrrrr rrrr rrrr}', r'\toprule',
         r' & \multicolumn{5}{c}{Noise wins} & \multicolumn{4}{c}{Action wins} & \multicolumn{4}{c}{Consistency} \\',
         r'\cmidrule(lr){2-6}\cmidrule(lr){7-10}\cmidrule(lr){11-14}',
         r'System & FR & t & d & n & $L_{10}$ & FR & t & d & $L_{10}$ & FR & t & f & $L_{10}$ \\', r'\midrule']
    for nome, d in (('v2', v2), ('v3', v3)):
        r.append(r'\multicolumn{14}{l}{\emph{' + nome + f" ({d['n']} emails, {d['email_con_bozza_o_task']} needing a reply or a task)" + r'}} \\')
        for s in ORDINE:
            if s not in d['sistemi']:
                continue
            x = d['sistemi'][s]
            nw, aw, co = x['noise_wins'], x['action_wins'], x['consistency']
            r.append(f"{NOME[s]} & {nw['falso_rumore']} & {nw['task_inutili']} & {nw['bozze_inutili']} & {nw['rumore_in_inbox']} & {num(nw['L_10'])} & "
                     f"{aw['falso_rumore']} & {aw['task_inutili']} & {aw['bozze_inutili']} & {num(aw['L_10'])} & "
                     f"{co['falso_rumore']} & {co['task_inutili']} & {co['segnalate']} & {num(co['L_10'])} \\\\")
    r += [r'\bottomrule', r'\end{tabular}}', r'\end{table*}', '']
    Path(sys.argv[3]).write_text('\n'.join(r), encoding='utf-8')
    print('wrote', sys.argv[3])


if __name__ == '__main__':
    main()
