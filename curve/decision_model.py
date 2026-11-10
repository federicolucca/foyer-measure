"""
decision_model — learning curve on a fixed test set: TF-IDF baseline or LoRA on a typed decision model.

  python curve/decision_model.py --user 1 --consent u1=consent_u1.json --data labels.jsonl \
      --split split_paper3.json --subsets curve_subsets.json --system baseline|lora [--base-model <hf id>] --out results

Fixed test set and nested training subsets from curve_split.py, seeds 7 and 11 (curve_baseline.py is the same run
with --system baseline; curve_lora.py runs --system lora in a Kaggle kernel):
- baseline = word TF-IDF 1-2 + character TF-IDF 3-5 (char_wb), no feature cap, balanced logistic regression; plus the
  majority class;
- lora = a single LoRA per training size on the five questions (Tev1 native format, early stop on a validation split
  taken from the training subset); size 0 = zero-shot of the base model. Needs a GPU.

Every system sees the same test ids (split_paper3.json) and the same nested subsets (curve_subsets.json); the test
set selects nothing (fixed hyper-parameters, early stop on the validation taken from the training subset).

Data: JSONL in the paper's format (see README); only rows verified by a person are used (source other than
'system'). Consent: a JSON {"owner": "1", "train_own": true, "updated_at": ...} for the user whose labels are used;
without the flag the run does not start. Output in --out: curve_baseline.json or curve_<model>_<seed>.json (metrics
only). No weights are ever saved: the best epoch is kept in memory.

This is the learning-curve path of the trainer used for the paper; the per-user cross-validation, shared and
transfer modes of the private trainer are not part of this repository.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import html
import json
import math
import random
import re
import sys
import time
from collections import Counter
from datetime import datetime
from functools import partial
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

_HERE = Path(__file__).resolve().parent
for _p in (_HERE, _HERE.parent):   # systemone_email.py: next to this file (Kaggle) or at the repository root
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import systemone_email as so  # noqa: E402

CAMPI = so.CAMPI
BOOLEANI = set(so.BOOLEANI)
SEME = 0
# The decision model's name → Hugging Face id of the weights the LoRA starts from. Passed with --base-model
# or with the environment variable FOYER_BASE_<NAME>, e.g. FOYER_BASE_TEV1_4B: no id is guessed here.
BASE_DEFAULT = 'tev1:4b'
CONSENSO = 'train_own'


# ─── data ────────────────────────────────────────────────────────────────────────────────────────

def norm(campo: str, v) -> str:
    if campo in BOOLEANI:
        return 'sì' if v is True or str(v).strip().lower() in ('sì', 'si', 'true', '1', 'yes') else 'no'
    return str(v).strip().lower()


def carica(path: Path, utente: str) -> list[dict]:
    """The rows labelled by a person; uncorrected system decisions are not labels."""
    righe = []
    # split('\n') and not splitlines(): a corpus row contains U+2028, which splitlines() would break
    for l in path.read_text(encoding='utf-8').split('\n'):
        if not l.strip():
            continue
        r = json.loads(l)
        if not r.get('labels') or r.get('source') == 'system':
            continue
        if any(not str(r['labels'].get(c, '')).strip() for c in CAMPI):
            continue
        r['labels'] = {c: norm(c, r['labels'][c]) for c in CAMPI}
        r['utente'] = utente
        righe.append(r)
    return righe


def gruppo(r: dict) -> str:
    """Thread = sender + subject without reply/forward prefixes (as in paper v2)."""
    s = r['state']
    oggetto = re.sub(r'^\s*((re|r|fwd?|i|inoltro)\s*:\s*)+', '', s.get('subject', '').lower()).strip()
    return f"{r['utente']}|{s.get('sender', '').strip().lower()}|{oggetto}"


def testo(r: dict, con_utente: bool = False) -> str:
    s = r['state']
    corpo = html.unescape(re.sub(r'<(style|script)[^>]*>.*?</\1>', ' ', s.get('body', '') or '', flags=re.S | re.I))
    corpo = re.sub(r'<[^>]+>|https?://\S+', ' ', corpo)
    prefisso = f"owner_{r['utente']} " if con_utente else ''
    return f"{prefisso}{s.get('sender', '')} {s.get('subject', '')} {corpo}"


def sha_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


# ─── metrics (same definitions as the paper's metrics.py) ────────────────────────────────────────

def macro_f1(y, p, classi):
    return sum(f1_classe(y, p, c) for c in classi) / len(classi)


def f1_classe(y, p, c):
    tp = sum(1 for a, b in zip(y, p) if a == c == b)
    fp = sum(1 for a, b in zip(y, p) if a != c and b == c)
    fn = sum(1 for a, b in zip(y, p) if a == c and b != c)
    return 2 * tp / (2 * tp + fp + fn) if tp else 0.0


def ece(conf, ok, bins=10):
    n, tot = len(conf), 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i, c in enumerate(conf) if lo < c <= hi or (b == 0 and c == 0)]
        if idx:
            tot += len(idx) / n * abs(sum(ok[i] for i in idx) / len(idx) - sum(conf[i] for i in idx) / len(idx))
    return tot


def metriche(y, pred, conf, ms, classi) -> dict:
    ok = [a == b for a, b in zip(y, pred)]
    return {'n': len(y), 'accuracy': round(sum(ok) / len(y), 4), 'macro_f1': round(macro_f1(y, pred, classi), 4),
            'ece': round(ece(conf, ok), 4), 'ms_per_email': round(ms, 3),
            'f1_per_classe': {c: round(f1_classe(y, pred, c), 4) for c in classi},
            'esempi_per_classe': {c: sum(1 for a in y if a == c) for c in classi}}


# ─── systems ─────────────────────────────────────────────────────────────────────────────────────

def carica_base(base: str, dtype):
    """The base model and the modules to exclude from the LoRA. Qwen3.5 (qwen3_5) is image-text-to-text: if
    AutoModelForCausalLM does not map it, it is loaded with the multimodal class, only the text is used and the LoRA
    does not touch the vision tower."""
    import torch
    from transformers import AutoModelForCausalLM
    # on the CPU (tests only) no device_map: 'auto' offloads part of the weights to disk (meta) and backward fails
    dm = 'auto' if torch.cuda.is_available() else None
    try:
        return AutoModelForCausalLM.from_pretrained(base, dtype=dtype, device_map=dm), None
    except (ValueError, KeyError) as e:
        from transformers import AutoModelForImageTextToText
        print(f'AutoModelForCausalLM does not load {base} ({type(e).__name__}: {str(e)[:200]}); '
              'using AutoModelForImageTextToText', flush=True)
        return (AutoModelForImageTextToText.from_pretrained(base, dtype=dtype, device_map=dm),
                r'.*(visual|vision).*')


class BaselineCurva:
    """The curve baseline: word TF-IDF 1-2 and character TF-IDF 3-5 (char_wb), no feature cap, balanced logistic
    regression. With a single class in the training subset it predicts that class (confidence 1)."""

    def __init__(self, seme=SEME):
        self.seme = seme

    def fit(self, righe, campo):
        from sklearn.pipeline import FeatureUnion
        y = [r['labels'][campo] for r in righe]
        self.costante = y[0] if len(set(y)) == 1 else None
        if self.costante is None:
            feat = FeatureUnion([('parole', TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2))),
                                 ('caratteri', TfidfVectorizer(sublinear_tf=True, analyzer='char_wb', ngram_range=(3, 5)))])
            self.m = Pipeline([('tfidf', feat), ('lr', LogisticRegression(max_iter=3000, class_weight='balanced',
                                                                          random_state=self.seme))])
            self.m.fit([testo(r) for r in righe], y)
        return self

    def predict(self, righe):
        if self.costante is not None:
            return [self.costante] * len(righe), [1.0] * len(righe)
        P = self.m.predict_proba([testo(r) for r in righe])
        cl = list(self.m.classes_)
        return [cl[int(k)] for k in P.argmax(1)], [float(x) for x in P.max(1)]


class Lora:
    """LoRA on a decision model: prompt = state + typed question, answer = the option. Loss on the answer only.

    format 'testo': free-text prompt, answer = key of the option; at prediction time the log-probability of each
    key. format 'tev1' (the curve's default): Tev1's native format (chat template, fixed system instruction, JSON
    decision with lettered options, no thinking), answer = letter + EOS; at prediction time the distribution over the
    option letters at the answer position, i.e. the constrained choice of /v1/systemone.
    fit() accepts one field or a list of fields (five questions per email → five examples, a single LoRA); with zero
    rows it trains nothing (zero-shot of the base model). val_frac > 0: early stop on the loss of a validation split
    taken by thread from the training rows (the best epoch is kept, in memory; no weights on disk)."""

    def __init__(self, base: str, con_utente=False, epoche=2, lr=2e-4, rango=16, batch=4, max_len=768,
                 formato='testo', val_frac=0.0, seme=SEME):
        self.base, self.con_utente = base, con_utente
        self.epoche, self.lr, self.rango, self.batch, self.max_len = epoche, lr, rango, batch, max_len
        self.formato, self.val_frac, self.seme = formato, val_frac, seme
        self.storia_val, self.epoca_migliore, self.s_per_passo = [], None, None

    def _prompt(self, r, campo):
        stato = so.stato({'sender': r['state'].get('sender', ''), 'subject': r['state'].get('subject', ''),
                          'body': r['state'].get('body', '')})
        if self.con_utente:
            stato = {'owner': f"u{r['utente']}", **stato}
        opzioni = so.CRITERI[campo]
        righe_op = '\n'.join(f'- {k}: {v}' for k, v in opzioni.items())
        # prompt text sent to the model (format 'testo'): kept in Italian, like the questions
        return (f"Stato: {json.dumps(stato, ensure_ascii=False)}\nDomanda: {so.DOMANDA[campo]}\n"
                f"Opzioni:\n{righe_op}\nRisposta:")

    def _opzione(self, campo, valore):
        # the corpus's boolean labels are sì/no, and so are the option keys of systemone_email
        return valore

    def _risposta(self, campo, valore):
        if self.formato == 'tev1':
            return self.tok(so.lettera(campo, valore), add_special_tokens=False)['input_ids'] + [self.tok.eos_token_id]
        return self.tok(' ' + self._opzione(campo, valore) + self.tok.eos_token, add_special_tokens=False)['input_ids']

    def _esempio(self, r, campo):
        """(prompt + answer ids, number of answer tokens): the loss is on those only."""
        a = self._risposta(campo, r['labels'][campo])
        return self._ids_prompt(r, campo) + a, len(a)

    def _split_val(self, righe):
        """Validation for the early stop: whole threads in random order (seed) until they cover val_frac of train."""
        if self.val_frac <= 0 or len(righe) < 10:
            return righe, []
        th = sorted({r.get('thread') or gruppo(r) for r in righe})
        random.Random(self.seme).shuffle(th)
        val_t, n = set(), 0
        for t in th:
            if n >= self.val_frac * len(righe):
                break
            val_t.add(t)
            n += sum(1 for r in righe if (r.get('thread') or gruppo(r)) == t)
        return ([r for r in righe if (r.get('thread') or gruppo(r)) not in val_t],
                [r for r in righe if (r.get('thread') or gruppo(r)) in val_t])

    def _loss(self, x, na):
        import torch
        ids = torch.tensor([x], device=self.m.device)
        # logits only at the positions that predict the answer: the same loss (mean cross-entropy over the
        # answer tokens, in fp32) without projecting the prompt onto the vocabulary
        logit = self.m(input_ids=ids, attention_mask=torch.ones_like(ids),
                       logits_to_keep=na + 1).logits[0, :-1].float()
        return torch.nn.functional.cross_entropy(logit, ids[0, -na:])

    def fit(self, righe, campo):
        import torch
        from peft import LoraConfig, get_peft_model
        from transformers import AutoTokenizer
        torch.manual_seed(self.seme)
        campi = [campo] if isinstance(campo, str) else list(campo)
        self.campo = campi[0]
        self.tok = AutoTokenizer.from_pretrained(self.base)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        # Kaggle's T4 has no bf16: base weights in fp16, LoRA parameters in fp32; on the CPU (tests only) all fp32
        if not torch.cuda.is_available():
            dtype = torch.float32
        else:
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        m, escludi = carica_base(self.base, dtype)
        if not righe:   # size 0: zero-shot of the base model, same prediction
            self.m = m.eval()
            return self
        m.gradient_checkpointing_enable()
        m.enable_input_require_grads()
        cfg = LoraConfig(r=self.rango, lora_alpha=2 * self.rango, lora_dropout=0.05, task_type='CAUSAL_LM',
                         target_modules='all-linear', exclude_modules=escludi)
        self.m = get_peft_model(m, cfg)
        for q in self.m.parameters():
            if q.requires_grad:
                q.data = q.data.float()
        train, val = self._split_val(righe)
        esempi = [self._esempio(r, c) for r in train for c in campi]
        esempi_val = [self._esempio(r, c) for r in val for c in campi]
        allenabili = [(n, q) for n, q in self.m.named_parameters() if q.requires_grad]
        opt = torch.optim.AdamW([q for _, q in allenabili], lr=self.lr)
        rnd = random.Random(self.seme)
        t0, passi, migliore, stato_migliore = time.perf_counter(), 0, math.inf, None
        for epoca in range(self.epoche):
            self.m.train()
            rnd.shuffle(esempi)
            for k in range(0, len(esempi), self.batch):
                lotto = esempi[k:k + self.batch]
                # one example at a time with gradient accumulation: the effective batch stays self.batch, but the
                # full-vocabulary logits of several 768-token examples do not fit in the 15 GB of a T4
                for x, na in lotto:
                    (self._loss(x, na) / len(lotto)).backward()
                opt.step()
                opt.zero_grad()
                passi += 1
            if esempi_val:
                self.m.eval()
                with torch.no_grad():
                    lv = sum(float(self._loss(x, na)) for x, na in esempi_val) / len(esempi_val)
                self.storia_val.append(round(lv, 5))
                if lv < migliore:
                    migliore, self.epoca_migliore = lv, epoca + 1
                    stato_migliore = {n: q.detach().clone() for n, q in allenabili}
        if stato_migliore is not None and self.epoca_migliore != self.epoche:
            with torch.no_grad():
                for n, q in allenabili:
                    q.copy_(stato_migliore[n])
        del stato_migliore
        self.s_per_passo = (time.perf_counter() - t0) / max(1, passi)
        self.m.eval()
        return self

    def _token_opzioni(self):
        return [self.tok(' ' + o + self.tok.eos_token, add_special_tokens=False)['input_ids']
                for o in so.CRITERI[self.campo]]

    def _token_lettere(self):
        ids = [self.tok(o['label'], add_special_tokens=False)['input_ids'] for o in so.opzioni(self.campo)]
        if any(len(x) != 1 for x in ids):
            raise SystemExit(f'the option letters are not one token each for {self.base}: {ids}')
        return [x[0] for x in ids]

    def _lp_cache(self, p, opzioni_tok):
        """The prompt once only; each option restarts from a copy of its cache. Qwen3.5's cache has state (conv and
        recurrent state of the gated delta layers, updated in place): that is why it is copied per option."""
        import copy
        import torch
        dev = self.m.device
        with torch.no_grad():
            out = self.m(input_ids=torch.tensor([p], device=dev), use_cache=True, logits_to_keep=1)
            primo = out.logits[0, -1].float().log_softmax(-1)
            cache = out.past_key_values
            lp = []
            for a in opzioni_tok:
                s = float(primo[a[0]])
                if len(a) > 1:
                    o = self.m(input_ids=torch.tensor([a[:-1]], device=dev), past_key_values=copy.deepcopy(cache),
                               use_cache=True)
                    logit = o.logits[0].float().log_softmax(-1)
                    s += sum(float(logit[j, t]) for j, t in enumerate(a[1:]))
                lp.append(s)
        return lp

    def _ids_prompt(self, r, campo=None):
        campo = campo or self.campo
        if self.formato != 'tev1':
            return self.tok(self._prompt(r, campo), add_special_tokens=True)['input_ids'][-(self.max_len - 8):]
        # Tev1: left truncation would cut the system instruction; the body is shortened instead
        extra = {'owner': f"u{r['utente']}"} if self.con_utente else None
        lim = so.BODY_MAX
        while True:
            dec = so.decisione(r['state'], campo, extra)
            dec['state']['corpo'] = dec['state']['corpo'][:lim]
            txt = self.tok.apply_chat_template(so.messaggi_tev1(dec), tokenize=False, add_generation_prompt=True,
                                               enable_thinking=False)
            ids = self.tok(txt, add_special_tokens=False)['input_ids']
            if len(ids) <= self.max_len - 4 or lim == 0:
                return ids
            lim = int(lim * 0.8) if lim > 50 else 0

    def _lp_lettere(self, p, lettere):
        """Tev1: one forward pass of the prompt, log-probabilities of the option letters at the answer position."""
        import torch
        with torch.no_grad():
            out = self.m(input_ids=torch.tensor([p], device=self.m.device), logits_to_keep=1)
            lp = out.logits[0, -1].float().log_softmax(-1)
        return [float(lp[t]) for t in lettere]

    def predict(self, righe, campo=None):
        if campo:
            self.campo = campo
        opzioni = list(so.CRITERI[self.campo])
        ot = self._token_lettere() if self.formato == 'tev1' else self._token_opzioni()
        pred, conf = [], []
        self.latenze_ms = []
        for r in righe:
            t = time.perf_counter()
            p = self._ids_prompt(r)
            lp = self._lp_lettere(p, ot) if self.formato == 'tev1' else self._lp_cache(p, ot)
            self.latenze_ms.append(1000 * (time.perf_counter() - t))
            z = max(lp)
            pr = [math.exp(x - z) for x in lp]
            s = sum(pr)
            k = int(np.argmax(pr))
            pred.append(opzioni[k])
            conf.append(pr[k] / s)
        return pred, conf


def versioni() -> dict:
    """Versions matter: the same CV gives 0.801 on type with scikit-learn 1.8 and 0.788 with 1.9 in paper v2."""
    import platform
    from importlib.metadata import PackageNotFoundError, version
    v = {'python': platform.python_version()}
    for p in ('scikit-learn', 'numpy', 'torch', 'transformers', 'peft', 'flash-linear-attention', 'triton'):
        try:
            v[p] = version(p)
        except PackageNotFoundError:
            pass
    return v


def base_hf(nome: str, esplicito: str | None) -> str:
    import os
    if esplicito:
        return esplicito
    var = 'FOYER_BASE_' + re.sub(r'[^A-Z0-9]', '_', nome.upper())
    if os.environ.get(var):
        return os.environ[var]
    raise SystemExit(f'Missing the weights id of {nome}: pass --base-model or the variable {var}.')


# ─── run ─────────────────────────────────────────────────────────────────────────────────────────

def verifica_consensi(utenti: list[str], file_consenso: dict[str, Path]) -> list[dict]:
    flag = CONSENSO
    verificati = []
    for u in utenti:
        if u not in file_consenso:
            raise SystemExit(f'User {u}: no consent file, the run does not start.')
        c = json.loads(file_consenso[u].read_text(encoding='utf-8'))
        if str(c.get('owner')) != u:
            raise SystemExit(f'User {u}: the consent file belongs to user {c.get("owner")}.')
        if c.get(flag) is not True:
            raise SystemExit(f'User {u}: consent "{flag}" missing or revoked, the run does not start.')
        verificati.append({'owner': u, 'flag': flag, 'valore': True, 'updated_at': c.get('updated_at'),
                           'file_sha256': sha_file(file_consenso[u])})
    return verificati


def libera():
    """Between one task and the next: without it, on the T4 (16 GB) the previous model stays in memory and OOMs."""
    gc.collect()
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _lavoratore(gpu, fn, compiti, risultati):
    """One process per GPU (CUDA_VISIBLE_DEVICES set by the parent before the spawn): takes tasks while there are
    any. Of the data, only the rows passed by the parent (inside fn) stay in memory, no file."""
    import traceback
    while True:
        x = compiti.get()
        if x is None:
            return
        k, args = x
        try:
            risultati.put((k, fn(*args, gpu=gpu), None))
        except Exception:
            risultati.put((k, None, traceback.format_exc(limit=8)))
            return


def esegui_parallelo(fn, compiti, gpus, peso=None, al_termine=None):
    """fn(*args) for each args in compiti; with several GPUs one process per GPU and a dynamic queue (heaviest
    tasks first: the order does not change the results). al_termine(k, outcome) as soon as a task finishes."""
    if gpus <= 1:
        esiti = []
        for k, args in enumerate(compiti):
            esiti.append(fn(*args))
            if al_termine:
                al_termine(k, esiti[-1])
        return esiti
    import multiprocessing as mp
    import os
    import queue
    ctx = mp.get_context('spawn')
    cq, rq = ctx.Queue(), ctx.Queue()
    for k in sorted(range(len(compiti)), key=lambda k: -(peso(compiti[k]) if peso else 0)):
        cq.put((k, compiti[k]))
    proc, prima = [], os.environ.get('CUDA_VISIBLE_DEVICES')
    for gpu in range(gpus):
        cq.put(None)
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
        p = ctx.Process(target=_lavoratore, args=(gpu, fn, cq, rq))
        p.start()
        proc.append(p)
    if prima is None:
        os.environ.pop('CUDA_VISIBLE_DEVICES', None)
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = prima
    esiti = {}
    while len(esiti) < len(compiti):
        try:
            k, r, err = rq.get(timeout=60)
        except queue.Empty:
            if not any(p.is_alive() for p in proc):
                raise SystemExit(f'the GPU processes ended with {len(compiti) - len(esiti)} tasks without outcome')
            continue
        if err:
            for p in proc:
                p.terminate()
            raise SystemExit(f'task {k} failed:\n{err}')
        esiti[k] = r
        if al_termine:
            al_termine(k, r)
    for p in proc:
        p.join()
    return [esiti[k] for k in range(len(compiti))]


# ─── learning curve on the fixed test set ───────────────────────────────────────────────────────

def metriche_curva(y, pred, conf, lat_ms, classi) -> dict:
    """The metrics of metriche() plus median latency and mean confidence of the correct and wrong answers."""
    m = metriche(y, pred, conf, float(np.mean(lat_ms)) if lat_ms else 0.0, classi)
    ok = [a == b for a, b in zip(y, pred)]
    giusti, errati = [c for c, o in zip(conf, ok) if o], [c for c, o in zip(conf, ok) if not o]
    m['latenza_mediana_ms'] = round(float(np.median(lat_ms)), 3) if lat_ms else None
    m['conf_media_corretti'] = round(float(np.mean(giusti)), 4) if giusti else None
    m['conf_media_errati'] = round(float(np.mean(errati)), 4) if errati else None
    return m


def maggioranza(y_train, y_test, classi) -> dict:
    """Always predicts the most frequent class of y_train (ties: the first in alphabetical order)."""
    c = Counter(y_train)
    top = min(c, key=lambda k: (-c[k], k))
    return {'classe': top, 'accuracy': round(sum(1 for a in y_test if a == top) / len(y_test), 4),
            'macro_f1': round(macro_f1(y_test, [top] * len(y_test), classi), 4)}


def _compito_curva(righe, campi, crea, sub, te, nome, gpu=None):
    """One size of one seed: one fit on all the rows of the subset, then the prediction of the fixed test per field.
    The baseline (one model per field) goes through here with crea building one system per field."""
    nome = nome if gpu is None else f'{nome} gpu {gpu}'
    t0 = time.perf_counter()
    out, info = {}, {}
    if getattr(crea, 'func', None) is Lora:
        s = crea().fit([righe[i] for i in sub], campi)
        info = {'fit_s': round(time.perf_counter() - t0, 1), 's_per_passo': s.s_per_passo,
                'loss_val_per_epoca': s.storia_val, 'epoca_migliore': s.epoca_migliore}
        for c in campi:
            p, conf = s.predict([righe[i] for i in te], c)
            out[c] = (p, conf, s.latenze_ms)
        del s
        libera()
    else:
        for c in campi:
            s = crea().fit([righe[i] for i in sub], c)
            t = time.perf_counter()
            p, conf = s.predict([righe[i] for i in te])
            out[c] = (p, conf, [1000 * (time.perf_counter() - t) / max(1, len(te))] * len(te))
    print(f'  {nome}: {len(sub)} rows in train, {time.perf_counter() - t0:.0f}s', flush=True)
    return out, info


def esegui_curva(a, utenti, verificati):
    """Same test ids (split_paper3.json) and nested subsets (curve_subsets.json) for every system; the test set
    selects nothing (fixed hyper-parameters, early stop on the validation taken from the training subset)."""
    dati = Path(a.dati)
    split = json.loads(Path(a.split).read_text(encoding='utf-8'))
    subsets = json.loads(Path(a.subsets).read_text(encoding='utf-8'))
    if split['dati_sha256'] != sha_file(dati) or subsets['dati_sha256'] != sha_file(dati):
        raise SystemExit('The data are not those of the split: different sha256.')
    righe = carica(dati, utenti[0])
    pos = {r['id']: i for i, r in enumerate(righe)}
    te = [pos[i] for i in split['test_ids']]
    if set(split['test_ids']) & set(subsets['semi'][next(iter(subsets['semi']))]['all']):
        raise SystemExit('Test set and pool overlap.')
    campi = [c for c in a.campi.split(',') if c in CAMPI]
    semi = [s.strip() for s in a.semi.split(',')]
    taglie = [t.strip() for t in a.taglie.split(',')]
    tfidf = a.sistema == 'baseline'
    if tfidf:
        taglie = [t for t in taglie if t != '0']
        crea = None
    else:
        base = base_hf(a.model, a.base_model)
        if a.gpus > 1 and not Path(base).exists():
            from huggingface_hub import snapshot_download
            snapshot_download(base)   # once, before the processes: they must not download in parallel to one cache
    y_test = {c: [righe[i]['labels'][c] for i in te] for c in campi}
    classi = {c: sorted(set(y_test[c])) for c in campi}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    comune = {'dati_sha256': split['dati_sha256'], 'test': {'righe': len(te), 'sha256_id': hashlib.sha256(
        json.dumps(sorted(split['test_ids'])).encode()).hexdigest()}, 'campi': campi,
        'consensi_verificati': verificati, 'versioni': versioni(), 'creato': datetime.now().isoformat(timespec='seconds')}
    compiti = []
    for s in semi:
        for t in taglie:
            sub = [] if t == '0' else [pos[i] for i in subsets['semi'][s][t]]
            crea_s = (partial(BaselineCurva, seme=int(s)) if tfidf else
                      partial(Lora, base, epoche=a.epoche, lr=a.lr, rango=a.rango, batch=a.batch, max_len=a.max_len,
                              formato=a.formato, val_frac=a.val_frac, seme=int(s)))
            compiti.append((s, t, sub, crea_s))
    risultati: dict[str, dict] = {s: {} for s in semi}

    def scrivi():
        # JSON keys and values are the format of the published results (Italian keys, kept as they are)
        if tfidf:
            media = {}
            for t in taglie:
                fatti = [s for s in semi if t in risultati[s]]
                if fatti:
                    media[t] = {c: {k: round(float(np.mean([risultati[s][t]['campi'][c][k] for s in fatti])), 4)
                                    for k in ('accuracy', 'macro_f1')} | {'semi': len(fatti)} for c in campi}
            obj = {'sistema': 'tfidf parole 1-2 + caratteri 3-5, logreg bilanciata', **comune,
                   'maggioranza_test': {c: maggioranza(y_test[c], y_test[c], classi[c]) for c in campi},
                   'semi': risultati, 'media_semi': media}
            (out / 'curve_baseline.json').write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding='utf-8')
        else:
            for s in semi:
                obj = {'sistema': 'lora', 'modello': a.model, 'base_model': base, 'seme': int(s), **comune,
                       'iperparametri': {'formato': a.formato, 'rango': a.rango, 'alpha': 2 * a.rango,
                                         'dropout': 0.05, 'moduli': 'all-linear (attenzione e MLP), torre visiva esclusa',
                                         'lr': a.lr, 'epoche': a.epoche, 'batch_effettivo': a.batch,
                                         'micro_batch': 1, 'max_len': a.max_len, 'val_frac': a.val_frac,
                                         'early_stop': 'miglior epoca per loss di validazione'},
                       'taglie': risultati[s]}
                nome = re.sub(r'[^A-Za-z0-9.]+', '-', a.model)
                (out / f'curve_{nome}_{s}.json').write_text(json.dumps(obj, ensure_ascii=False, indent=1),
                                                             encoding='utf-8')

    def al_termine(k, esito):
        s, t, sub, _ = compiti[k]
        per_campo, info = esito
        rec = {'etichette': len(sub), 'campi': {}, **info}
        for c in campi:
            p, conf, lat = per_campo[c]
            rec['campi'][c] = metriche_curva(y_test[c], p, conf, lat, classi[c])
            if sub:
                rec['campi'][c]['maggioranza_train'] = maggioranza([righe[i]['labels'][c] for i in sub],
                                                                   y_test[c], classi[c])
        risultati[s][t] = rec
        print(f'seed {s} size {t}:', {c: rec['campi'][c]['accuracy'] for c in campi}, flush=True)
        scrivi()   # after every size: an interrupted kernel does not lose the sizes already done

    righe_fn = partial(_compito_curva_args, righe, campi, te)
    esegui_parallelo(righe_fn, [(sub, crea_s, f'seed {s} size {t}') for s, t, sub, crea_s in compiti],
                     1 if tfidf else a.gpus, peso=lambda x: len(x[0]), al_termine=al_termine)
    scrivi()
    print('wrote', out)


def _compito_curva_args(righe, campi, te, sub, crea, nome, gpu=None):
    return _compito_curva(righe, campi, crea, sub, te, nome, gpu=gpu)


def main(argv=None, sistema: str | None = None):
    """CLI of the curve run. sistema fixes --system (curve_baseline.py passes 'baseline')."""
    ap = argparse.ArgumentParser(description=__doc__.strip().split('\n')[0])
    ap.add_argument('--user', required=True, help='the user whose labels are used (e.g. 1); checked in the consent')
    ap.add_argument('--consent', '--consenso', dest='consenso', action='append', default=[], required=True,
                    help='uN=file.json with {"owner": "N", "train_own": true}')
    ap.add_argument('--data', '--dati', dest='dati', required=True, help='the labelled corpus (JSONL)')
    ap.add_argument('--split', required=True, help='split_paper3.json (from curve_split.py split)')
    ap.add_argument('--subsets', required=True, help='curve_subsets.json (from curve_split.py split)')
    ap.add_argument('--out', required=True, help='results folder (curve_baseline.json, curve_<model>_<seed>.json)')
    if sistema is None:
        ap.add_argument('--system', '--sistema', dest='sistema', choices=['lora', 'baseline'], default='lora')
    ap.add_argument('--model', default=BASE_DEFAULT, help='name of the decision model, e.g. tev1:4b | tev1:0.8b')
    ap.add_argument('--base-model', help='Hugging Face id (or folder) of the weights the LoRA starts from')
    ap.add_argument('--fields', '--campi', dest='campi', default=','.join(CAMPI))
    ap.add_argument('--seeds', '--semi', dest='semi', default='7,11')
    ap.add_argument('--sizes', '--taglie', dest='taglie', default='0,50,100,200,300,400,all',
                    help='0 = zero-shot (LoRA only)')
    ap.add_argument('--gpus', type=int, default=1,
                    help='LoRA: how many GPUs to use, one process per GPU; the (seed, size) tasks are spread across them')
    ap.add_argument('--format', '--formato', dest='formato', choices=['tev1', 'testo'], default='tev1')
    ap.add_argument('--epochs', '--epoche', dest='epoche', type=int, default=3)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--rank', '--rango', dest='rango', type=int, default=16)
    ap.add_argument('--batch', type=int, default=16, help='effective batch (micro-batch 1 with accumulation)')
    ap.add_argument('--max-len', type=int, default=1024)
    ap.add_argument('--val-frac', type=float, default=0.1)
    a = ap.parse_args(argv)
    if sistema is not None:
        a.sistema = sistema

    utenti = [u.strip().lstrip('u') for u in [a.user] if u and u.strip()]
    if not utenti:
        raise SystemExit('--user is required.')
    consenso = {k.lstrip('u'): Path(v) for k, v in (x.split('=', 1) for x in a.consenso)}
    verificati = verifica_consensi(utenti, consenso)   # before reading a single label
    return esegui_curva(a, utenti, verificati)


if __name__ == '__main__':
    main()
