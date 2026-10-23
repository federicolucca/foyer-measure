# foyer-measure — measurement harness for a personal inbox agent

Code and protocol accompanying the preprint

> F. Lucca, *Measure Before You Train: Typed Decision Models vs. a Classical Baseline vs. a Generative LLM for Routing a Personal Multilingual Inbox*, arXiv, October 2026.

The paper measures how well different systems — production keyword rules, a TF-IDF + logistic-regression baseline, local and hosted typed decision models (Tev1, Nimble, Jev) and hosted generative LLMs — make the routing decisions of a personal email agent: message type, urgency, and three switches (needs-reply, action-48h, is-noise). This repository contains everything needed to run the same measurement on **your own** inbox. **No emails, labels or per-email results are included: the data in the paper are the author's private correspondence and are not released.**

## What is here

| File | Purpose |
|---|---|
| `systemone_email.py` | Client for the typed-decision endpoint (`/v1/systemone`, Ollama ≥ 0.35 or a hosted server): one typed question per call, automatic reload on server errors, the exact questions and class definitions used in the paper (Italian, with English glosses in the paper's Appendix B). |
| `bench_email.py` | The benchmark: TF-IDF + logistic regression (out-of-fold), a local decision model, the production prediction and optional online systems, all on the same labelled JSONL, same splits, same metrics (accuracy, macro-F1, latency). Writes `<field>.risultati.json`. |
| `online_backends.py` | Backends for hosted systems: TypeSafe Jev, any System One server (Nimble, Clef, Ollama cloud), generative LLMs through OpenRouter in JSON mode. Keys are read from environment variables only. |
| `kappa.py` | Cohen's κ with bootstrap intervals, confusion matrices and disagreement lists, for a second annotator's CSV against a reference. No dependencies. |
| `INSTRUCTIONS_annotator.md` / `ISTRUZIONI_annotatore.md` | The labelling instructions given to annotators (English / Italian), with the label definitions. |

## The protocol in one page

1. **Sample** from the production database, stratified by the field you want to measure (120–250 emails per round is enough to start).
2. **Freeze a snapshot** of exactly what the model will see (sender, subject, body truncated to 1,500 characters, date) and **redact secrets** before storing it (passwords, tokens, account numbers).
3. **Split by thread**, not by email id: a reply is a near-duplicate of its parent. Keep the fixed split for later; use thread-grouped cross-validation while the test split is small.
4. **Label by hand** in the agent's own review loop: a model proposal, then a human confirms or corrects. Record which was proposed and which was kept.
5. **Pre-register** the hypotheses of each new round before opening its labels (we commit a file with thresholds and a verification script).
6. **Count before training**: majority class, the production system, a trained baseline, and the zero-shot systems — on the same rows, under the same protocol.
7. **Compose the labels into actions** (draft / task / ignore / nothing) and count the costly error — an email that needed action and was ignored — not just accuracy. Write down the price of each error; choose thresholds for it.

## Input format

One JSON object per line:

```json
{"id": 123, "split": "train", "thread": "normalised subject|sender address",
 "state": {"sender": "...", "subject": "...", "body": "...", "date": "2026-10-01"},
 "labels": {"tipo": "aggiornamento", "urgenza": "bassa", "relazione": "servizio", "progetto": null,
            "serve_risposta": false, "azione_48h": false, "e_rumore": true},
 "predicted": {"tipo": "...", "urgenza": "..."}}
```

Label vocabularies (Italian in the data, as shown to the models; English in the paper): `tipo` ∈ {richiesta, aggiornamento, offerta, newsletter, spam, altro}; `urgenza` ∈ {alta, media, bassa}; `relazione` ∈ {cliente, fornitore, socio, famiglia, istituzione, servizio, monitoraggio, me}.

## Running

```bash
pip install scikit-learn
ollama pull tev1:0.8b          # or tev1 (4B), nimble (9B)
python bench_email.py labels.jsonl --campo tipo --modello tev1:0.8b
python bench_email.py labels.jsonl --campo urgenza --online openrouter --modello google/gemma-4-26b-a4b-it   # OPENROUTER_API_KEY
python bench_email.py labels.jsonl --campo tipo --online jev                                                 # TYPESAFE_API_KEY
python kappa.py annotator.csv reference.json
```

Two practical notes from the paper: the System One endpoint silently fails when option descriptions are long (keep each `criteria` value under ~120 characters), and run one request at a time on CPU (`OLLAMA_NUM_PARALLEL=1`).

## Licence

Code: MIT. Instructions and documentation: CC BY 4.0.
