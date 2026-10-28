"""
systemone_email — classificazione email con un modello decisionale (Jev-style) via /v1/systemone.

Domande, definizioni e pulizia del corpo sono quelle del paper (Appendice B, v2): le stesse date a tutti i sistemi.
  - tipo (6 classi) e urgenza (3 livelli) come `choice`;
  - i tre interruttori (serve_risposta, azione_48h, e_rumore) come `choice` sì/no con una definizione per opzione;
    `noul` è disponibile per il confronto di formato (nel paper: Jev è sensibile al formato, Nimble no);
  - progetto come `choice` fra i progetti esistenti + «nessuno», con una descrizione di al massimo 120 caratteri.
Stato = {mittente, oggetto, corpo}: corpo senza blocchi style/script, tag e URL, spazi compattati, 1500 caratteri.

Lezioni pratiche (Ollama 0.40.6 / 0.35+, tev1:0.8b su CPU):
- Una richiesta con >= 3 domande può fallire con "the model could not evaluate the decision": una domanda per chiamata.
- Dopo un errore il modello resta in uno stato sporco: una chiamata con keep_alive=0 (scarica il modello), poi si riprova.
- Tev1 accetta al massimo 24 opzioni per domanda; ogni opzione DEVE avere una descrizione.
- Descrizioni lunghe fanno fallire in silenzio l'endpoint: tenerle sotto ~120 caratteri.

Uso:
    from systemone_email import classify_email
    out = classify_email({"sender": ..., "subject": ..., "body": ...},
                         projects=[{"slug": "infra", "description": "..."}, ...])
"""
from __future__ import annotations

import html
import json
import re
import time
import urllib.error
import urllib.request

OLLAMA_URL = "http://localhost:11434/v1/systemone"
MODEL = "tev1:0.8b"          # oppure "tev1" (4B) o "nimble" (9B, Q8 ~9.5 GB)
MAX_OPTIONS = 22             # Tev1: 24 opzioni max, teniamo margine per «nessuno»
BODY_MAX = 1500
DESC_MAX = 120
TIMEOUT = 300

# Appendice B.1–B.2 del paper (verbatim). «cliente» non è più un tipo: è una relazione del mittente.
CRITERI = {
    'tipo': {
        'richiesta': 'chi scrive chiede al destinatario qualcosa, o serve una sua azione o risposta',
        'aggiornamento': 'informazione, notifica, conferma o esito senza richiesta (codici di sicurezza, avvisi di accesso, prenotazioni, report di stato, thread in copia)',
        'offerta': 'proposta commerciale mirata al destinatario (preventivo, offerta personale)',
        'newsletter': 'invio di massa periodico o promozionale da un brand o servizio',
        'spam': 'commerciale non richiesto da mittente sconosciuto, o phishing',
        'altro': 'tutto il resto, compresi i messaggi inviati dal destinatario stesso',
    },
    'urgenza': {
        'alta': 'serve un\'azione entro circa 48 ore o c\'è un appuntamento in giornata',
        'media': 'serve un\'azione o una lettura attenta senza scadenza vicina (avvisi di sicurezza, conferme di orario)',
        'bassa': 'nessuna azione richiesta',
    },
    'serve_risposta': {
        'sì': 'il destinatario dovrebbe rispondere a questa email: qualcuno aspetta una sua risposta',
        'no': 'nessuna risposta necessaria (notifica, informazione, invio di massa, messaggio già chiuso)',
    },
    'azione_48h': {
        'sì': 'il destinatario deve fare qualcosa entro circa 48 ore (scadenza, appuntamento, pagamento, verifica)',
        'no': 'nessuna azione con scadenza nelle prossime 48 ore',
    },
    'e_rumore': {
        'sì': 'rumore: notifica automatica, newsletter, pubblicità o messaggio che si può ignorare',
        'no': 'merita l\'attenzione del destinatario',
    },
}
DOMANDA = {'tipo': 'Che tipo di messaggio è questa email per il suo destinatario?',
           'urgenza': 'Quanto è urgente agire su questa email?',
           'serve_risposta': 'Il destinatario deve rispondere a questa email?',
           'azione_48h': 'Il destinatario deve fare qualcosa per questa email entro 48 ore?',
           'e_rumore': 'Questa email è rumore che si può ignorare?',
           'progetto': 'A quale progetto del destinatario appartiene questa email?'}
CAMPI = ['tipo', 'urgenza', 'serve_risposta', 'azione_48h', 'e_rumore']
BOOLEANI = ['serve_risposta', 'azione_48h', 'e_rumore']
NESSUNO = 'nessuno'

# compatibilità con la v1 dell'harness
TIPI = CRITERI['tipo']
URGENZA_LABEL = ['bassa', 'media', 'alta']


class SystemOneError(RuntimeError):
    pass


def domanda(campo: str, formato: str = 'choice') -> dict:
    """La domanda del paper per un campo; per i booleani formato 'choice' (sì/no con definizioni) o 'noul'."""
    if campo in BOOLEANI and formato == 'noul':
        return {'type': 'noul', 'instructions': DOMANDA[campo]}
    return {'type': 'choice', 'instructions': DOMANDA[campo], 'criteria': CRITERI[campo]}


def etichetta(campo: str, risposta: dict):
    """Dalla risposta System One all'etichetta: stringa per tipo/urgenza, bool per i booleani."""
    if 'noul' in risposta:
        return float(risposta['noul']) >= 0.5
    return risposta['choice'] == 'sì' if campo in BOOLEANI else risposta['choice']


def stato(email: dict) -> dict:
    """Quello che il modello vede, e nient'altro (Appendice B.3)."""
    s = email.get('state', email)
    corpo = html.unescape(re.sub(r'<(style|script)[^>]*>.*?</\1>', ' ', s.get('body', '') or '', flags=re.S | re.I))
    corpo = re.sub(r'\s+', ' ', re.sub(r'<[^>]+>|https?://\S+', ' ', corpo)).strip()
    return {'mittente': s.get('sender', ''), 'oggetto': s.get('subject', ''), 'corpo': corpo[:BODY_MAX]}


snapshot = stato   # nome della v1


def _post(body: dict, url: str = OLLAMA_URL, headers: dict | None = None) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
                                 headers={"Content-Type": "application/json", **(headers or {})})
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
    """Una domanda, una chiamata (Ollama locale). Ritorna la risposta grezza per quella domanda."""
    body = {"model": model, "state": state, "questions": {name: question}}
    last = None
    for _ in range(tries):
        try:
            return _post(body)["answers"][name]
        except urllib.error.HTTPError as e:
            last = e.read().decode(errors="replace")[:200]
            _reset(model)
            time.sleep(0.5)
    raise SystemOneError(f"{name}: {last}")


def project_criteria(projects: list[dict], descrizioni: bool = True) -> dict:
    """Progetti esistenti -> opzioni (descrizione troncata a 120 caratteri, o solo il nome) più «nessuno»."""
    crit = {}
    for p in projects[:MAX_OPTIONS]:
        desc = (p.get("description") or "").strip()
        crit[p["slug"]] = desc[:DESC_MAX] if descrizioni and desc else f"l'email riguarda il progetto o l'area «{p.get('name', p['slug'])}»"
    crit[NESSUNO] = "l'email non riguarda nessuno dei progetti elencati"
    return crit


def classify_email(email: dict, projects: list[dict] | None = None, model: str = MODEL, formato: str = 'choice') -> dict:
    """Le decisioni di un'email, una domanda per chiamata: etichetta, risposta grezza e tempo per campo."""
    st = stato(email)
    out: dict = {"meta": {"model": model, "per_domanda_s": {}}}
    t0 = time.time()
    for campo in CAMPI:
        t = time.time()
        a = ask(st, campo, domanda(campo, formato), model)
        out["meta"]["per_domanda_s"][campo] = round(time.time() - t, 2)
        out[campo] = {"label": etichetta(campo, a), "raw": a}
    if projects:
        q = {"type": "choice", "instructions": DOMANDA['progetto'], "criteria": project_criteria(projects)}
        a = ask(st, "progetto", q, model)
        out["progetto"] = {"label": a["choice"], "raw": a, "nuovo": a["choice"] == NESSUNO}
    out["meta"]["totale_s"] = round(time.time() - t0, 2)
    return out


if __name__ == "__main__":
    import sys
    # Dati sintetici (nessuna email reale è inclusa in questo repository).
    projects = [
        {"slug": "infrastruttura", "description": "servizi cloud, deploy, monitoraggio, avvisi dei sistemi"},
        {"slug": "personale", "description": "posta personale non di lavoro: acquisti, viaggi, servizi"},
    ]
    emails = [
        {"sender": "Jane Doe <jane.doe@example.com>", "subject": "Re: homepage draft",
         "body": "Hi, could you send the updated homepage draft by Thursday? We need it for the board meeting. Thanks, Jane"},
        {"sender": "Deploy Bot <noreply@example.net>", "subject": "Deploy failed on main",
         "body": "Pipeline #42 failed at step build. See logs for details."},
    ]
    model = sys.argv[1] if len(sys.argv) > 1 else MODEL
    for e in emails:
        r = classify_email(e, projects, model)
        print(e["subject"][:60], {c: r[c]["label"] for c in CAMPI}, r["progetto"]["label"], f"{r['meta']['totale_s']}s")
