"""
bench_email — confronto su uno stesso JSONL fra:
  A) baseline classica: TF-IDF (parole + caratteri) + regressione logistica, addestrata sullo split train
  B) modello decisionale via Ollama /v1/systemone (Tev1 / Nimble), zero-shot
  C) (opzionale) etichette gia' predette dalla pipeline attuale, se presenti nel JSONL ("predicted")

Input: il JSONL di `python manage.py dataset_email --esporta file.jsonl`, una riga per email:
  {"id", "split": "train|val|test", "state": {sender, subject, body, date}, "labels": {"tipo", ...}, "predicted": {...}}

Uso:
  python bench_email.py dataset.jsonl --campo tipo --modello tev1:0.8b --out risultati.json
  python bench_email.py dataset.jsonl --campo tipo --no-systemone        # solo baseline

Ogni esecuzione scrive un JSON con metriche, matrice di confusione e tempi: e' la documentazione
dell'esperimento per il paper (sezione Results). Niente numeri a mano.
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from collections import Counter, defaultdict
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import FeatureUnion, Pipeline

import systemone_email as so
import online_backends as ob

CLASSI = {
    "tipo": list(so.TIPI),
    "urgenza": so.URGENZA_LABEL,
}


def testo(state: dict) -> str:
    return f"{state.get('sender', '')}\n{state.get('subject', '')}\n{state.get('body', '')}"


def carica(path: Path) -> list[dict]:
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    return [r for r in rows if r.get("labels")]


def baseline(train: list[dict], test: list[dict], campo: str) -> tuple[list[str], float, float]:
    """TF-IDF parole(1-2) + caratteri(3-5) + LogReg bilanciata. Ritorna (predizioni, s_train, s_pred)."""
    pipe = Pipeline([
        ("feat", FeatureUnion([
            ("w", TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)),
            ("c", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1, sublinear_tf=True)),
        ])),
        ("clf", LogisticRegression(max_iter=2000, C=4.0, class_weight="balanced")),
    ])
    t = time.time()
    pipe.fit([testo(r["state"]) for r in train], [r["labels"][campo] for r in train])
    s_train = time.time() - t
    t = time.time()
    pred = list(pipe.predict([testo(r["state"]) for r in test]))
    s_pred = time.time() - t
    return pred, s_train, s_pred


def systemone(test: list[dict], campo: str, modello: str) -> tuple[list[str], list[float], float]:
    pred, tempi = [], []
    for r in test:
        t = time.time()
        if campo == "tipo":
            a = so.ask(so.snapshot(r["state"]), "tipo", {"type": "choice", "instructions": "What kind of email is this?", "criteria": so.TIPI}, modello)
            pred.append(a["choice"])
        elif campo == "urgenza":
            a = so.ask(so.snapshot(r["state"]), "urgenza", {"type": "score", "instructions": "How urgent is this email for the recipient?", "criteria": so.URGENZA}, modello)
            pred.append(so.URGENZA_LABEL[round(a["score"])])
        else:
            raise SystemExit(f"campo {campo} non supportato per systemone")
        tempi.append(time.time() - t)
    return pred, tempi, sum(tempi)


def valuta(nome: str, y, pred, classi) -> dict:
    return {
        "sistema": nome,
        "n": len(y),
        "accuracy": round(accuracy_score(y, pred), 4),
        "macro_f1": round(f1_score(y, pred, labels=classi, average="macro", zero_division=0), 4),
        "per_classe": classification_report(y, pred, labels=classi, output_dict=True, zero_division=0),
        "confusione": {"classi": classi, "matrice": confusion_matrix(y, pred, labels=classi).tolist()},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl", type=Path)
    ap.add_argument("--campo", default="tipo", choices=list(CLASSI))
    ap.add_argument("--modello", default=so.MODEL)
    ap.add_argument("--no-systemone", action="store_true")
    ap.add_argument("--online", choices=list(ob.BACKENDS), default=None, help="aggiunge un sistema online (serve la chiave in env)")
    ap.add_argument("--test-split", default="test", help="test | val | test,val")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    rows = carica(a.jsonl)
    splits = set(a.test_split.split(","))
    train = [r for r in rows if r["split"] == "train"]
    test = [r for r in rows if r["split"] in splits]
    campo, classi = a.campo, CLASSI[a.campo]
    y = [r["labels"][campo] for r in test]
    if not train or not test:
        raise SystemExit(f"train={len(train)} test={len(test)}: servono entrambi")

    print(f"dataset: {len(rows)} righe · train {len(train)} · test {len(test)} ({a.test_split}) · campo {campo}")
    print("distribuzione train:", dict(Counter(r["labels"][campo] for r in train)))
    print("distribuzione test: ", dict(Counter(y)))

    risultati = {"dataset": str(a.jsonl), "campo": campo, "n_train": len(train), "n_test": len(test),
                 "test_split": a.test_split, "macchina": platform.platform(), "quando": time.strftime("%Y-%m-%d %H:%M"),
                 "sistemi": []}

    # A) baseline classica
    pred_a, s_fit, s_pred = baseline(train, test, campo)
    ra = valuta("tfidf+logreg", y, pred_a, classi)
    ra["tempi_s"] = {"addestramento": round(s_fit, 2), "predizione_totale": round(s_pred, 3), "per_email_ms": round(1000 * s_pred / len(test), 2)}
    risultati["sistemi"].append(ra)

    # C) predizioni della pipeline attuale, se ci sono
    if all(r.get("predicted", {}).get(campo) for r in test):
        pred_c = [r["predicted"][campo] for r in test]
        risultati["sistemi"].append(valuta("pipeline_attuale", y, pred_c, classi))

    # B) modello decisionale
    if not a.no_systemone:
        pred_b, tempi, tot = systemone(test, campo, a.modello)
        rb = valuta(f"systemone:{a.modello}", y, pred_b, classi)
        rb["tempi_s"] = {"predizione_totale": round(tot, 2), "per_email_s": round(tot / len(test), 2),
                         "min_s": round(min(tempi), 2), "max_s": round(max(tempi), 2)}
        risultati["sistemi"].append(rb)
        # dove i due sistemi divergono: materiale per l'analisi degli errori
        risultati["divergenze"] = [
            {"id": r.get("id"), "subject": r["state"].get("subject", "")[:80], "vero": yy, "tfidf": pa, "systemone": pb}
            for r, yy, pa, pb in zip(test, y, pred_a, pred_b) if pa != pb
        ]

    # D/E/F) sistema online
    if a.online:
        pred_o, tempi_o, errori = [], [], 0
        for r in test:
            t = time.time()
            try:
                lab, _ = ob.classify(a.online, so.snapshot(r["state"]), campo, a.modello)
            except Exception as e:
                lab, errori = classi[0], errori + 1   # fallback alla prima classe, contato
                print(f"  online errore su id={r.get('id')}: {str(e)[:90]}")
            pred_o.append(lab); tempi_o.append(time.time() - t)
        ro = valuta(f"online:{a.online}:{a.modello}", y, pred_o, classi)
        ro["tempi_s"] = {"predizione_totale": round(sum(tempi_o), 2), "per_email_s": round(sum(tempi_o) / len(test), 2), "errori_api": errori}
        risultati["sistemi"].append(ro)

    print()
    print(f"{'sistema':28s} {'acc':>6s} {'macroF1':>8s}  tempi")
    for s in risultati["sistemi"]:
        print(f"{s['sistema']:28s} {s['accuracy']:6.3f} {s['macro_f1']:8.3f}  {s.get('tempi_s', '')}")

    out = a.out or a.jsonl.with_suffix(f".{campo}.risultati.json")
    out.write_text(json.dumps(risultati, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nscritto {out}")


if __name__ == "__main__":
    main()
