# foyer-measure — measurement harness for a personal inbox agent

Code and protocol accompanying the preprint

> F. Lucca, *Measure Before You Train: Typed Decision Models vs. a Classical Baseline vs. a Generative LLM for Routing a Personal Multilingual Inbox*, arXiv:2610.32955 (v2, October 2026; paper v2.1, November 2026, adds Jeva-4B as the eleventh system — code tag `v2.1`).

The paper measures how well different systems — production keyword rules, a TF-IDF + logistic-regression baseline, local and hosted typed decision models (Tev1, Nimble, Jeva, Jev) and hosted generative LLMs — make the routing decisions of a personal email agent: message type, urgency, and three switches (needs-reply, action-48h, is-noise). This repository contains everything needed to run the same measurement on **your own** inbox. **No emails, labels or per-email results are included: the data in the paper are the author's private correspondence and are not released.**

## What is here

| File | Purpose |
|---|---|
| `systemone_email.py` | The questions, class definitions and body cleaning used in the paper (Appendix B, Italian, verbatim): type (6 classes) and urgency (3 levels) as `choice`, the three switches as `choice` yes/no with one definition per option (`noul` available for the format comparison), project as `choice` with descriptions of at most 120 characters. Client for a local `/v1/systemone` (Ollama ≥ 0.35): one question per call, reload after server errors. |
| `online_backends.py` | The same questions for hosted systems: TypeSafe Jev, Bespoke Nimble, any System One server, generative LLMs through OpenRouter in JSON mode (the Italian prompt of Appendix B.4). Keys from environment variables only. |
| `jeva_backend.py` | *(v2.1)* Jeva-4B in-process on a local GPU (`model.system_one`), weights `LIQIIIII/Jeva-4B` from the Hub, Jeva repository pinned to commit `e2104e0`. Answers in the `/v1/systemone` shape; used by `decision_bench.py` and `project_bench.py` with `--backend jeva`. Options `--jeva-cache hybrid\|none` and, for projects, `--native-null`. |
| `bench_email.py` | TF-IDF + logistic regression, a local decision model, the production prediction and an optional online system on the same labelled JSONL, same splits, same metrics; all five fields. |
| `decision_bench.py` | *(v2)* An online system (or, from v2.1, Jeva-4B on a local GPU) on all five fields, one question per call, in parallel and resumable; raw answers to a JSONL. `repeat` re-asks a random subset to measure determinism. |
| `metrics.py` | *(v2)* From raw answers to results: accuracy with bootstrap 95% intervals, macro-F1, majority class, median and p95 latency, API errors, token cost, mean confidence on correct vs. wrong answers, ECE (10 bins), per-email agreement and exact McNemar against a second system, determinism. |
| `cost_threshold.py` | *(v2)* The is-noise threshold chosen for a written cost (missed email × 10 + email handled needlessly × 1) inside thread-grouped cross-validation, with the same rule for the baseline (inner CV) and for zero-shot systems (their own probabilities). |
| `project_bench.py` | *(v2)* Project assignment with one owner-written description per project (≤ 120 characters) against names only. The descriptions file stays private. |
| `kappa.py` | Cohen's κ with bootstrap intervals, confusion matrices and disagreement lists, for a second annotator's CSV against a reference. No dependencies. |
| `kappa_multi.py` | *(v2)* One or two blind annotators: κ against the reference, κ between them and Fleiss' κ for three raters, with bootstrap intervals. No dependencies. |
| `INSTRUCTIONS_annotator.md` / `ISTRUZIONI_annotatore.md` | The labelling instructions given to annotators (English / Italian), with the label definitions. |

## The protocol in one page

1. **Sample** from the production database, stratified by the field you want to measure (120–250 emails per round is enough to start).
2. **Freeze a snapshot** of exactly what the model will see (sender, subject, body truncated to 1,500 characters, date) and **redact secrets** before storing it (passwords, tokens, account numbers).
3. **Split by thread**, not by email id: a reply is a near-duplicate of its parent. Keep the fixed split for later; use thread-grouped cross-validation while the test split is small.
4. **Label by hand** in the agent's own review loop: a model proposal, then a human confirms or corrects. Record which was proposed and which was kept. Give a sample to a **blind second annotator** (sender, subject, date, body only) and report κ.
5. **Pre-register** the hypotheses of each new round before opening its labels (we commit a file with thresholds and a verification script).
6. **Count before training**: majority class, the production system, a trained baseline, and the zero-shot systems — on the same rows, under the same protocol and the **same question format**.
7. **Compose the labels into actions** (draft / task / ignore / nothing) and count the costly error — an email that needed action and was ignored — not just accuracy. Write down the price of each error; choose thresholds for it, **for every system that returns probabilities, not only for the trained one**.

## Input format

One JSON object per line:

```json
{"id": 123, "split": "train", "thread": "normalised subject|sender address",
 "state": {"sender": "...", "subject": "...", "body": "...", "date": "2026-10-01"},
 "labels": {"tipo": "aggiornamento", "urgenza": "bassa", "relazione": "servizio", "progetto": null,
            "serve_risposta": false, "azione_48h": false, "e_rumore": true},
 "predicted": {"tipo": "...", "urgenza": "..."}}
```

Label vocabularies (Italian in the data, as shown to the models; English in the paper): `tipo` ∈ {richiesta, aggiornamento, offerta, newsletter, spam, altro}; `urgenza` ∈ {alta, media, bassa}; `relazione` ∈ {cliente, fornitore, socio, famiglia, istituzione, servizio, monitoraggio, me}. The descriptions file for `project_bench.py` is a Markdown table `| slug | descrizione |`, with an optional `nessuno` row.

## Running

```bash
pip install scikit-learn numpy
ollama pull tev1:0.8b          # or tev1 (4B), nimble (9B)
python bench_email.py labels.jsonl --campo tipo --modello tev1:0.8b
python bench_email.py labels.jsonl --campo urgenza --online openrouter --modello google/gemma-4-26b-a4b-it   # OPENROUTER_API_KEY

# v2: a hosted decision model on all fields, then the metrics
python decision_bench.py run labels.jsonl raw_jev.jsonl --backend jev                                        # TYPESAFE_API_KEY
python decision_bench.py repeat labels.jsonl raw_jev.jsonl --backend jev --n 100
python decision_bench.py run labels.jsonl raw_nimble.jsonl --backend nimble                                  # BESPOKE_API_KEY
python metrics.py labels.jsonl raw_jev.jsonl results_jev.json --prezzo-input 0.042 --ripetizione raw_jev.jsonl --confronto raw_nimble.jsonl

# v2: same switches in the other format, cost threshold for everyone, projects, annotators
python decision_bench.py run labels.jsonl raw_jev_noul.jsonl --backend jev --formato noul --campi serve_risposta,azione_48h,e_rumore
python cost_threshold.py labels.jsonl results_threshold.json --sistema jev=raw_jev.jsonl --sistema nimble=raw_nimble.jsonl
python project_bench.py labels.jsonl descriptions.md results_project.json --backend nimble --variante desc
python kappa_multi.py reference.json results_kappa.json annotator1.csv annotator2.csv
```

Practical notes from the paper:
- **Short option descriptions:** the System One endpoint silently fails when option descriptions are long. Keep each `criteria` value under ~120 characters; 19 options are fine.
- **One request at a time on CPU:** set `OLLAMA_NUM_PARALLEL=1`.
- **Question format:** it is part of the measurement. In the paper, Jev's is-noise accuracy moves by 11 points between `noul` and `choice`, while Nimble's does not move. Compare systems in the same format.
- **Hosted models are not deterministic:** some hosted endpoints return the same label but different probabilities on a repeated request. Use `repeat`.
- **Small differences on re-runs:** re-running the baseline in another environment (scikit-learn version, snapshot of the text) can change a handful of out-of-fold predictions. On the paper's data, 5 of 231 is-noise predictions differed.

## Jeva-4B (v2.1)

The Jeva-4B rows of the paper (Tables 2, 3 and 8) were produced with `jeva_backend.py` on a single NVIDIA T4 (Kaggle), bfloat16.

```bash
git clone https://github.com/LiQiiiii/Jeva && cd Jeva && git checkout e2104e0 && pip install -e . && cd ..
# weights: LIQIIIII/Jeva-4B, downloaded from the Hugging Face Hub on first use

# Tables 2 and 3: type, urgency and the three switches (switches as choice yes/no, the format of Table 3)
python decision_bench.py run labels.jsonl raw_jeva.jsonl --backend jeva
python metrics.py labels.jsonl raw_jeva.jsonl results_jeva.json
# Table 3 note: the switches asked as noul
python decision_bench.py run labels.jsonl raw_jeva_noul.jsonl --backend jeva --formato noul --campi serve_risposta,azione_48h,e_rumore
python metrics.py labels.jsonl raw_jeva_noul.jsonl results_jeva_noul.json

# Table 8: project assignment, "nessuno" as an ordinary option (allow_null=False), descriptions and names only
python project_bench.py labels.jsonl descriptions.md results_project_jeva.json --backend jeva --variante desc
python project_bench.py labels.jsonl descriptions.md results_project_jeva_names.json --backend jeva --variante nomi
# same decisions without the option-encoding cache (latency comparison)
python project_bench.py labels.jsonl descriptions.md results_project_jeva_nocache.json --backend jeva --variante desc --jeva-cache none
```

Notes:
- **Language.** Jeva-4B was trained mostly on English decision data (the authors, personal communication, November 2026). The questions and options are in Italian, as for every other system in the paper (Appendix B), so for Jeva the input is out of distribution. Ablations with English questions are possible and were not run.
- **Native null (`--native-null`): open-set, uncalibrated.** With `--native-null` the `nessuno` option is removed and the question is asked with `allow_null=True`; Jeva's null outcome is mapped back to `nessuno`. This is an open-set setting: its null threshold has to be calibrated on an independent validation split. Without one, as in the paper, the result is **uncalibrated** and not comparable with the fixed-class rows. The paper reports it separately, in the caption of Table 8.
- **Cache.** `--jeva-cache hybrid` (default) reuses option encodings across calls; `none` re-encodes every option on every call. On the paper's data both give the same decisions, at about a third to a fifth of the latency with the cache.
- One model per process: `decision_bench.py` runs Jeva with a single worker.

## Changelog

- **v2.1** (November 2026): `jeva_backend.py` and `--backend jeva` in `decision_bench.py` and `project_bench.py` (`--jeva-cache`, `--native-null`), for the Jeva-4B rows of paper v2.1.
- **v2** (October 2026): the paper's questions verbatim, all five fields, hosted decision-model bench, metrics, cost threshold, project assignment and multi-annotator agreement.
- **v1**: System One client, benchmark harness, online backends, Cohen's κ, annotator instructions.

## Licence

Code: MIT. Instructions and documentation: CC BY 4.0.
