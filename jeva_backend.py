"""
jeva_backend — Jeva-4B (Q. Li, X. Wang, arXiv:2610.18773) answering the paper's questions in-process, no HTTP server.

The same code that produced the Jeva-4B rows of the paper (v2.1: Tables 2, 3 and 8), on a Kaggle T4.

  Install   git clone https://github.com/LiQiiiii/Jeva && cd Jeva && git checkout e2104e0 && pip install -e .
            (JEVA_COMMIT below; an NVIDIA GPU is needed in practice: BF16 weights, FP16 on cards without BF16)
  Weights   LIQIIIII/Jeva-4B from the Hugging Face Hub (downloaded on first use, never stored in this repository)
  Used by   decision_bench.py --backend jeva   and   project_bench.py --backend jeva

The question is the one every System One server gets in this harness ({"type": "choice" | "noul", "instructions",
"criteria": {option id: description}}), asked through Jeva's own typed API (model.system_one), and the answer comes back
in the /v1/systemone shape ({"choice" | "noul", "confidence", "probabilities", ...}), so metrics.py reads it unchanged.

How "none" is passed (project assignment):
  null_mode="option"  (default, the paper's Table 8 row) "none" is an ordinary option with its description,
                      allow_null=False: the fixed-class comparison every other system got.
  null_mode="native"  (--native-null) "none" is removed from the options and the question is asked with allow_null=True;
                      Jeva's null outcome is mapped back to "none". This is an open-set setting: its null threshold should be
                      calibrated on an independent validation split; without one the result is UNCALIBRATED and not
                      comparable with the fixed-class rows.

Option-encoding cache: cache="hybrid" (default, as in the paper) reuses the option encodings across calls; cache="none"
encodes every option on every call. Same decisions, different latency.
"""
from __future__ import annotations

import os
import time

JEVA_REPO = 'https://github.com/LiQiiiii/Jeva'
JEVA_COMMIT = 'e2104e0'
CHECKPOINT = 'LIQIIIII/Jeva-4B'
NULL_MODES = ('option', 'native')
CACHES = ('hybrid', 'none')

_CONFIG = {'cache': os.environ.get('JEVA_CACHE', 'hybrid'), 'dtype': os.environ.get('JEVA_DTYPE') or None,
           'device': os.environ.get('JEVA_DEVICE', 'cuda')}
_MODELS: dict = {}


def configure(cache: str | None = None, dtype: str | None = None, device: str | None = None) -> None:
    """Set the cache mode, dtype and device before the first question (decision_bench / project_bench flags)."""
    if cache is not None:
        if cache not in CACHES:
            raise ValueError(f'cache must be one of {CACHES}')
        _CONFIG['cache'] = cache
    if dtype is not None:
        _CONFIG['dtype'] = dtype
    if device is not None:
        _CONFIG['device'] = device


def load_jeva(checkpoint: str = CHECKPOINT, device: str = 'cuda', dtype: str | None = None,
              cache: str = 'hybrid', truncation: str = 'truncate'):
    """Load Jeva once. dtype None: bfloat16 where the GPU supports it, else float16 (the paper ran bfloat16 on a T4)."""
    from jeva import Jeva
    if dtype is None:
        import torch
        dtype = 'bfloat16' if device.startswith('cuda') and torch.cuda.is_bf16_supported() else 'float16'
    return Jeva.from_pretrained(checkpoint, device=device, merge_lora=True, cache=cache, dtype=dtype,
                                truncation=truncation)


def model(checkpoint: str = '') -> object:
    """The loaded model for this process (one per checkpoint), with the configured cache and dtype."""
    key = checkpoint or CHECKPOINT
    if key not in _MODELS:
        _MODELS[key] = load_jeva(key, device=_CONFIG['device'], dtype=_CONFIG['dtype'], cache=_CONFIG['cache'])
    return _MODELS[key]


def _jeva_question(question: dict, none_label: str | None, null_mode: str):
    from jeva import Choice, Noul
    kind = question['type']
    if kind == 'noul':
        return Noul(instructions=question['instructions'])
    if kind != 'choice':
        raise ValueError(f'unsupported question type {kind!r}')
    if null_mode not in NULL_MODES:
        raise ValueError(f'null_mode must be one of {NULL_MODES}')
    crit = dict(question['criteria'])
    allow_null = null_mode == 'native' and none_label is not None and none_label in crit
    if allow_null:
        crit.pop(none_label)
    return Choice(instructions=question['instructions'], criteria=crit, allow_null=allow_null)


def ask_jeva(m, state: dict | str, question: dict, none_label: str | None = None, null_mode: str = 'option') -> dict:
    """One typed question -> a /v1/systemone-shaped answer plus Jeva's own latency and token usage."""
    q = _jeva_question(question, none_label, null_mode)
    t = time.perf_counter()
    resp = m.system_one(state=state, questions={'q': q})
    wall = time.perf_counter() - t
    a = resp.answers['q']
    out = {'type': question['type'], 'confidence': a.confidence, 'probabilities': dict(a.probabilities),
           'null_probability': a.null_probability, 'latency_ms': resp.latency_ms, 'wall_s': wall,
           'usage': {k: getattr(resp.usage, k) for k in resp.usage.__dataclass_fields__}}
    if question['type'] == 'choice':
        out['choice'] = none_label if a.choice is None else a.choice
        out['null_native'] = a.choice is None
    else:
        out['noul'] = a.noul
    return out


def classify(state: dict, campo: str, checkpoint: str = '', formato: str = 'choice'):
    """decision_bench interface: (label, {"raw": answer, "usage": usage}) like the hosted backends in online_backends.py."""
    import systemone_email as so
    a = ask_jeva(model(checkpoint), state, so.domanda(campo, formato))
    usage = a.pop('usage')
    return so.etichetta(campo, a), {'raw': a, 'usage': usage}


def ask_project(state: dict, question: dict, none_label: str, checkpoint: str = '', null_mode: str = 'option') -> str:
    """project_bench interface: the chosen option id ("none" for Jeva's native null when null_mode="native")."""
    return ask_jeva(model(checkpoint), state, question, none_label=none_label, null_mode=null_mode)['choice']
