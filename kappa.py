"""
kappa.py — accordo fra il secondo annotatore (CSV compilato) e le etichette di riferimento.

Uso:  python kappa.py annotazione_60_compilata.csv chiave_60_NON_INVIARE.json

Per ogni campo: accordo grezzo, kappa di Cohen con IC 95% bootstrap (2000 ricampionamenti),
matrice di confusione, e le righe di disaccordo (id, riferimento, annotatore, dubbio).
Nessuna dipendenza oltre la libreria standard.
"""
import csv, json, sys, random, collections

CAMPI = ['tipo', 'urgenza', 'serve_risposta', 'azione_48h', 'e_rumore']
BOOL = {'serve_risposta', 'azione_48h', 'e_rumore'}

def norm(campo, v):
    v = str(v).strip().lower()
    if campo in BOOL:
        return '1' if v in ('1', 'si', 'sì', 'yes', 'true', 'x') else '0'
    return v

def kappa(a, b):
    n = len(a)
    if n == 0: return float('nan')
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = collections.Counter(a), collections.Counter(b)
    pe = sum(ca[k] * cb[k] for k in set(a) | set(b)) / (n * n)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0

def main(csv_path, key_path):
    key = {int(k): v for k, v in json.load(open(key_path, encoding='utf-8')).items()}
    rows = list(csv.DictReader(open(csv_path, encoding='utf-8')))
    rng = random.Random(0)
    print(f"righe compilate: {len(rows)}\n")
    for campo in CAMPI:
        a, b, dis = [], [], []
        for r in rows:
            i = int(r['id'])
            if i not in key or not str(r.get(campo, '')).strip(): continue
            ref = key[i][campo]; ref = ('1' if ref else '0') if campo in BOOL else str(ref)
            ann = norm(campo, r[campo])
            a.append(ref); b.append(ann)
            if ref != ann: dis.append((i, ref, ann, r.get('dubbio', ''), (r.get('oggetto') or '')[:50]))
        if not a:
            print(f"== {campo}: nessuna riga compilata\n"); continue
        k = kappa(a, b); po = sum(x == y for x, y in zip(a, b)) / len(a)
        boots = []
        for _ in range(2000):
            idx = [rng.randrange(len(a)) for _ in a]
            boots.append(kappa([a[i] for i in idx], [b[i] for i in idx]))
        boots.sort(); lo, hi = boots[int(0.025 * len(boots))], boots[int(0.975 * len(boots))]
        print(f"== {campo}: n={len(a)}  accordo={po:.3f}  kappa={k:.3f} [IC95% {lo:.3f}, {hi:.3f}]")
        conf = collections.Counter(zip(a, b))
        labs = sorted(set(a) | set(b))
        print("   confusione (riferimento \\ annotatore):")
        print("   " + " " * 14 + " ".join(f"{l[:7]:>8s}" for l in labs))
        for x in labs:
            print(f"   {x[:14]:14s}" + " ".join(f"{conf[(x, y)]:8d}" for y in labs))
        if dis:
            print(f"   disaccordi ({len(dis)}):")
            for i, ref, ann, d, ogg in dis:
                print(f"     id {i:5d}  rif={ref:14s} ann={ann:14s} dubbio={d or '-':2s} {ogg}")
        print()
    print("Lettura: kappa < 0.4 debole, 0.4–0.6 moderato, 0.6–0.8 sostanziale, > 0.8 quasi perfetto (Landis & Koch).")

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print(__doc__); sys.exit(1)
    main(sys.argv[1], sys.argv[2])
