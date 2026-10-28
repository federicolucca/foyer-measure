"""
online_backends — gli stessi campi chiesti a sistemi ONLINE, con le domande del paper (Appendice B).

  jev         TypeSafe Jev       POST https://api.typesafe.ai/v1/systemone        $TYPESAFE_API_KEY
  nimble      Bespoke Nimble     POST https://api.bespokelabs.ai/v1/systemone      $BESPOKE_API_KEY   (modello nimble-v4-max)
  systemone   qualsiasi server System One (stesso wire format di Ollama): $SYSTEMONE_URL [+ $SYSTEMONE_KEY]
  openrouter  LLM generativo in JSON mode, temperatura 0: $OPENROUTER_API_KEY, modello a scelta

Ogni backend ritorna l'etichetta nel vocabolario di systemone_email (stringa per tipo/urgenza, bool per i booleani),
così bench_email.py e decision_bench.py li valutano con le stesse metriche. Le chiavi NON vanno nel repo:
solo variabili d'ambiente.
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request

import systemone_email as so

TIMEOUT = 120
SERVER = {
    'jev': ('https://api.typesafe.ai/v1/systemone', 'jev-latest', 'TYPESAFE_API_KEY'),
    'nimble': ('https://api.bespokelabs.ai/v1/systemone', 'nimble-v4-max', 'BESPOKE_API_KEY'),
}


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


def _key(name: str) -> str:
    k = os.environ.get(name, "")
    if not k:
        raise RuntimeError(f"{name} mancante")
    return k


# ---------------------------------------------------------------- System One remoti (Jev, Nimble, altri)

def systemone_remote(state: dict, campo: str, url: str, model: str, key: str = "", formato: str = "choice"):
    body = {"model": model, "state": state, "questions": {campo: so.domanda(campo, formato)}}
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    out = _with_retry(lambda: _post(url, body, headers))
    a = out["answers"][campo]
    return so.etichetta(campo, a), {"raw": a, "usage": out.get("usage", {})}


def hosted(nome: str):
    def fn(state, campo, model="", formato="choice"):
        url, default, key = SERVER[nome]
        if nome == 'jev' and os.environ.get("TYPESAFE_BASE_URL"):
            url = os.environ["TYPESAFE_BASE_URL"] + "/v1/systemone"
        return systemone_remote(state, campo, url, model or default, _key(key), formato)
    return fn


def systemone_env(state: dict, campo: str, model: str, formato: str = "choice"):
    url = os.environ.get("SYSTEMONE_URL", "")
    if not url:
        raise RuntimeError("SYSTEMONE_URL mancante (es. https://ollama.com/v1/systemone)")
    return systemone_remote(state, campo, url, model, os.environ.get("SYSTEMONE_KEY", ""), formato)


# ---------------------------------------------------------------- LLM generativo (OpenRouter), Appendice B.4

def _sistema(campo: str) -> str:
    return (so.DOMANDA[campo] + ' Scegli esattamente una etichetta. Definizioni:\n'
            + '\n'.join(f'- {k}: {v}' for k, v in so.CRITERI[campo].items())
            + f'\nRispondi SOLO con JSON: {{"{campo}": "<etichetta>"}}')


def openrouter(state: dict, campo: str, model: str, formato: str = "choice"):
    body = {"model": model, "max_tokens": 60, "temperature": 0, "usage": {"include": True},
            "response_format": {"type": "json_object"},
            "messages": [{"role": "system", "content": _sistema(campo)},
                         {"role": "user", "content": json.dumps(state, ensure_ascii=False)}]}
    headers = {"Authorization": f"Bearer {_key('OPENROUTER_API_KEY')}", "X-Title": "foyer-measure"}
    out = _with_retry(lambda: _post("https://openrouter.ai/api/v1/chat/completions", body, headers))
    text = out["choices"][0]["message"]["content"] or ""
    try:
        obj = json.loads(text[text.find("{"): text.rfind("}") + 1])
    except Exception:
        raise RuntimeError(f"risposta non JSON: {text[:120]}")
    label = str(obj.get(campo, "")).strip().lower()
    if label not in so.CRITERI[campo]:
        raise RuntimeError(f"etichetta fuori vocabolario: {label!r}")
    return (label == 'sì') if campo in so.BOOLEANI else label, {"raw": obj, "usage": out.get("usage", {})}


BACKENDS = {"jev": hosted('jev'), "nimble": hosted('nimble'), "systemone": systemone_env, "openrouter": openrouter}


def classify(backend: str, state: dict, campo: str, model: str = "", formato: str = "choice"):
    return BACKENDS[backend](state, campo, model, formato)
