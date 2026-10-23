"""
online_backends — gli stessi campi (tipo, urgenza) chiesti a modelli ONLINE, per chiudere il confronto:

  D) Jev (TypeSafe)         POST https://api.typesafe.ai/v1/systemone   Authorization: Bearer $TYPESAFE_API_KEY
  E) Nimble / Clef / altro  qualsiasi server System One (stesso wire format di Ollama): $SYSTEMONE_URL [+ $SYSTEMONE_KEY]
                            es. Ollama cloud  https://ollama.com/v1/systemone  model clef:cloud  (serve account)
                            es. Nimble demo    https://bespokelabs--nimble-sglang-nimble.us-west.modal.direct/v1/systemone (spesso spento)
  F) LLM generativo         OpenRouter chat/completions in JSON mode: $OPENROUTER_API_KEY, modello a scelta
                            (es. qwen/qwen2.5-7b-instruct = stessi pesi del locale; google/gemma-4-26b-a4b-it:free)

Tutti ritornano l'etichetta nello stesso vocabolario di systemone_email (TIPI, URGENZA_LABEL), così
bench_email.py li valuta con le stesse metriche. Le chiavi NON vanno nel repo: solo variabili d'ambiente.

Uso da bench_email.py:
  python bench_email.py dati.jsonl --campo tipo --online jev
  python bench_email.py dati.jsonl --campo tipo --online systemone --modello nimble
  python bench_email.py dati.jsonl --campo tipo --online openrouter --modello qwen/qwen2.5-7b-instruct
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request

import systemone_email as so

TIMEOUT = 120


def _post(url: str, body: dict, headers: dict | None = None) -> dict:
    h = {"Content-Type": "application/json", **(headers or {})}
    req = urllib.request.Request(url, data=json.dumps(body, ensure_ascii=False).encode("utf-8"), headers=h)
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        return json.load(r)


def _with_retry(fn, tries=3, wait=3.0):
    last = None
    for i in range(tries):
        try:
            return fn()
        except urllib.error.HTTPError as e:
            body = e.read().decode(errors="replace")[:300]
            last = f"HTTP {e.code}: {body}"
            if e.code in (400, 401, 402, 403):   # errori nostri o di chiave: non riprovare
                raise RuntimeError(last)
        except (urllib.error.URLError, TimeoutError) as e:
            last = str(e)
        time.sleep(wait * (i + 1))
    raise RuntimeError(last or "fallito")


# ---------------------------------------------------------------- System One remoti (Jev, Nimble, Clef…)

def _question(campo: str) -> tuple[str, dict]:
    if campo == "tipo":
        return "tipo", {"type": "choice", "instructions": "What kind of email is this?", "criteria": so.TIPI}
    if campo == "urgenza":
        return "urgenza", {"type": "score", "instructions": "How urgent is this email for the recipient?", "criteria": so.URGENZA}
    raise ValueError(campo)


def _systemone_label(campo: str, answer: dict) -> str:
    if campo == "tipo":
        return answer["choice"]
    return so.URGENZA_LABEL[round(answer["score"])]


def systemone_remote(state: dict, campo: str, url: str, model: str, key: str = "") -> tuple[str, dict]:
    """Un server System One qualsiasi (Jev, Nimble su SGLang, Ollama cloud…): stesso body di Ollama."""
    name, q = _question(campo)
    body = {"model": model, "state": state, "questions": {name: q}}
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    out = _with_retry(lambda: _post(url, body, headers))
    a = out["answers"][name]
    return _systemone_label(campo, a), a


def jev(state: dict, campo: str, model: str = "jev-latest") -> tuple[str, dict]:
    key = os.environ.get("TYPESAFE_API_KEY", "")
    if not key:
        raise RuntimeError("TYPESAFE_API_KEY mancante")
    url = os.environ.get("TYPESAFE_BASE_URL", "https://api.typesafe.ai") + "/v1/systemone"
    return systemone_remote(state, campo, url, model, key)


def systemone_env(state: dict, campo: str, model: str) -> tuple[str, dict]:
    url = os.environ.get("SYSTEMONE_URL", "")
    if not url:
        raise RuntimeError("SYSTEMONE_URL mancante (es. https://ollama.com/v1/systemone)")
    return systemone_remote(state, campo, url, model, os.environ.get("SYSTEMONE_KEY", ""))


# ---------------------------------------------------------------- LLM generativo (OpenRouter)

_SYS_TIPO = ("Classify the email into exactly one type. Types and definitions:\n"
             + "\n".join(f"- {k}: {v}" for k, v in so.TIPI.items())
             + '\nReply ONLY with JSON: {"tipo": "<one of the types>"}')
_SYS_URG = ("Rate the urgency of the email for the recipient. Levels:\n"
            + "\n".join(f"- {l}" for l in so.URGENZA)
            + '\nReply ONLY with JSON: {"urgenza": "bassa" | "media" | "alta"}')


def openrouter(state: dict, campo: str, model: str) -> tuple[str, dict]:
    key = os.environ.get("OPENROUTER_API_KEY", "")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY mancante")
    system = _SYS_TIPO if campo == "tipo" else _SYS_URG
    user = json.dumps(state, ensure_ascii=False)
    body = {"model": model, "max_tokens": 60, "temperature": 0,
            "response_format": {"type": "json_object"},
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
    headers = {"Authorization": f"Bearer {key}", "X-Title": "foyer-measure"}
    out = _with_retry(lambda: _post("https://openrouter.ai/api/v1/chat/completions", body, headers))
    text = out["choices"][0]["message"]["content"]
    try:
        obj = json.loads(text[text.find("{"): text.rfind("}") + 1])
    except Exception:
        raise RuntimeError(f"risposta non JSON: {text[:120]}")
    label = str(obj.get(campo, "")).strip().lower()
    valid = list(so.TIPI) if campo == "tipo" else so.URGENZA_LABEL
    if label not in valid:
        raise RuntimeError(f"etichetta fuori vocabolario: {label!r}")
    return label, {"raw": obj, "usage": out.get("usage", {})}


BACKENDS = {"jev": jev, "systemone": systemone_env, "openrouter": openrouter}


def classify(backend: str, state: dict, campo: str, model: str) -> tuple[str, dict]:
    fn = BACKENDS[backend]
    if backend == "jev":
        return fn(state, campo, model or "jev-latest")
    return fn(state, campo, model)
