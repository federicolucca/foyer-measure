"""
systemone_email — classificazione email con un modello decisionale (Jev-style) via Ollama.

Usa l'endpoint POST /v1/systemone (Ollama >= 0.35) con Tev1 / Nimble / Clef in locale.
Pensato per sostituire il "LLM pass 1" di classify_emails sulle domande CHIUSE:
    tipo, urgenza, serve_risposta, azione_48h, e_rumore, progetto (scelta fra quelli esistenti)
Il modello generativo resta solo per i casi "progetto = nessuno" (nome, descrizione, task).

Lezioni dai test del 20/10/2026 (Ollama 0.40.6, tev1:0.8b, CPU 1 core, 3 GB):
- Una richiesta con >= 3 domande fallisce con "the model could not evaluate the decision";
  con 1-2 domande funziona. Quindi: una domanda per chiamata (lo stato viene ricalcolato, ma
  su GPU sono ~100 ms a domanda; su CPU 1-3 s).
- Dopo un errore il modello resta in uno stato sporco: la chiamata successiva fallisce anche se
  valida. Rimedio: una chiamata con keep_alive=0 (scarica il modello) e poi si riprova.
- Il primo tentativo a freddo puo' fallire: retry.
- Tev1 accetta al massimo 24 opzioni per domanda; ogni opzione DEVE avere una descrizione.
  Con piu' di ~22 progetti si fa una pre-selezione (ultimi aggiornati, o per contatto).
- Le descrizioni in italiano e gli accenti vanno bene.

Uso:
    from systemone_email import classify_email
    out = classify_email({"sender": ..., "subject": ..., "body": ..., "date": ...},
                         projects=[{"slug": "acme-site", "name": "...", "description": "..."}, ...])
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request

OLLAMA_URL = "http://localhost:11434/v1/systemone"
MODEL = "tev1:0.8b"          # oppure "tev1" (4B) o "nimble" (9B, Q8 ~9.5 GB)
MAX_OPTIONS = 22             # Tev1: 24 opzioni max, teniamo margine per "nessuno"
BODY_MAX = 1500              # stesso taglio di EmailLabel.snapshot
TIMEOUT = 300

TIPI = {
    "richiesta": "Someone asks the recipient to do something",
    "aggiornamento": "A status update on ongoing work, no action requested",
    "offerta": "A commercial offer, quote or proposal",
    "cliente": "A message from a paying client about their project",
    "altro": "Anything else from a real person",
    "spam": "Unsolicited marketing",
    "newsletter": "Automated newsletter or notification",
}
URGENZA = ["bassa: no deadline, can wait days", "media: should be handled this week", "alta: needs action today"]
URGENZA_LABEL = ["bassa", "media", "alta"]


class SystemOneError(RuntimeError):
    pass


def _post(body: dict) -> dict:
    req = urllib.request.Request(OLLAMA_URL, data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        return json.load(r)


def _reset(model: str) -> None:
    """Scarica il modello: dopo un 500 lo slot resta sporco e ogni chiamata successiva fallisce."""
    try:
        _post({"model": model, "state": "ok", "keep_alive": 0,
               "questions": {"q": {"type": "noul", "instructions": "ok?"}}})
    except Exception:
        pass


def ask(state: dict | str, name: str, question: dict, model: str = MODEL, tries: int = 3) -> dict:
    """Una domanda, una chiamata. Ritorna la risposta grezza di Ollama per quella domanda."""
    body = {"model": model, "state": state, "questions": {name: question}}
    last = None
    for i in range(tries):
        try:
            return _post(body)["answers"][name]
        except urllib.error.HTTPError as e:
            last = e.read().decode(errors="replace")[:200]
            _reset(model)
            time.sleep(0.5)
    raise SystemOneError(f"{name}: {last}")


def snapshot(email: dict) -> dict:
    """Lo stesso 'stato' congelato di dataset_email: quello che il modello vede, e nient'altro."""
    return {
        "sender": (email.get("sender") or "")[:255],
        "subject": (email.get("subject") or "")[:512],
        "body": (email.get("body") or email.get("context") or "")[:BODY_MAX],
        "date": email.get("date") or "",
    }


def project_criteria(projects: list[dict]) -> dict:
    """Progetti esistenti -> opzioni con descrizione (obbligatoria per Tev1), piu' 'nessuno'."""
    crit = {}
    for p in projects[:MAX_OPTIONS]:
        desc = (p.get("description") or "").strip() or p.get("name", "")
        crit[p["slug"]] = f"{p.get('name', p['slug'])}: {desc[:160]}"
    crit["nessuno"] = "None of these projects (a new work relationship, or personal/non-project mail)"
    return crit


def classify_email(email: dict, projects: list[dict], model: str = MODEL) -> dict:
    """
    Sei decisioni per un'email. Ritorna etichette + probabilita' + confidenza + tempi.
    Chiavi: tipo, urgenza, serve_risposta, azione_48h, e_rumore, progetto, meta.
    """
    state = snapshot(email)
    t0 = time.time()
    out: dict = {"meta": {"model": model, "per_domanda_s": {}}}

    def timed(name, q):
        t = time.time()
        a = ask(state, name, q, model)
        out["meta"]["per_domanda_s"][name] = round(time.time() - t, 2)
        return a

    a = timed("tipo", {"type": "choice", "instructions": "What kind of email is this?", "criteria": TIPI})
    out["tipo"] = {"label": a["choice"], "p": a["probabilities"], "conf": a["confidence"]}

    a = timed("urgenza", {"type": "score", "instructions": "How urgent is this email for the recipient?", "criteria": URGENZA})
    out["urgenza"] = {"label": URGENZA_LABEL[round(a["score"])], "score": a["score"], "p": a["probabilities"], "conf": a["confidence"]}

    for name, instr in (
        ("serve_risposta", "Does this email require a written reply from the recipient?"),
        ("azione_48h", "Does it require a concrete action by the recipient within 48 hours?"),
        ("e_rumore", "Is this email noise (automated notification, marketing, irrelevant)?"),
    ):
        a = timed(name, {"type": "noul", "instructions": instr})
        out[name] = {"label": a["noul"] >= 0.5, "p_true": a["noul"]}

    if projects:
        a = timed("progetto", {"type": "choice", "instructions": "Which existing work project does this email belong to?",
                               "criteria": project_criteria(projects)})
        out["progetto"] = {"label": a["choice"], "p": a["probabilities"], "conf": a["confidence"],
                           "nuovo": a["choice"] == "nessuno"}
    out["meta"]["totale_s"] = round(time.time() - t0, 2)
    return out


if __name__ == "__main__":
    import sys
    # Synthetic demo data (no real emails are shipped with this repository).
    projects = [
        {"slug": "acme-site", "name": "ACME website", "description": "redesign of a client's corporate website"},
        {"slug": "infra", "name": "Infrastructure", "description": "cloud services, deployments, monitoring alerts"},
        {"slug": "taxes", "name": "Taxes", "description": "accountant, filings, invoices"},
    ]
    emails = [
        {"sender": "Jane Doe <jane.doe@example.com>", "subject": "Re: homepage draft", "date": "2026-09-14",
         "body": "Hi, could you send the updated homepage draft by Thursday? We need it for the board meeting. Thanks, Jane"},
        {"sender": "Deploy Bot <noreply@example.net>", "subject": "Deploy failed on main", "date": "2026-10-16",
         "body": "Pipeline #42 failed at step build. See logs for details."},
        {"sender": "Newsletter <news@example.org>", "subject": "October product update", "date": "2026-10-07",
         "body": "This month: new features, webinars and a 20% discount on annual plans. Unsubscribe here."},
    ]
    model = sys.argv[1] if len(sys.argv) > 1 else MODEL
    for e in emails:
        r = classify_email(e, projects, model)
        print(f"\n{e['subject'][:60]}")
        print(f"  tipo={r['tipo']['label']} ({r['tipo']['conf']:.2f})  urgenza={r['urgenza']['label']} ({r['urgenza']['score']:.2f})"
              f"  risposta={r['serve_risposta']['p_true']:.2f}  48h={r['azione_48h']['p_true']:.2f}  rumore={r['e_rumore']['p_true']:.2f}")
        print(f"  progetto={r['progetto']['label']} ({r['progetto']['conf']:.2f}) {json.dumps({k: round(v, 2) for k, v in r['progetto']['p'].items()})}")
        print(f"  {r['meta']['totale_s']}s")
