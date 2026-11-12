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

## v2.2 — learning curve (Train After You Measure)

The scripts of the follow-up paper (*Train After You Measure*): a fixed thread-grouped test set, nested training subsets, a TF-IDF baseline and LoRA fine-tunes of Tev1-0.8B and Tev1-4B at 0 (zero-shot), 50, 100, 200, 300, 400 and all labels, two seeds (7, 11), and the paper's tables, macros, hypothesis decisions and figure generated from the result files. They live in `curve/` and use the questions and definitions of `systemone_email.py` (Appendix B, Italian, verbatim), which from v2.2 also writes them in Tev1's native decision format (`decisione`, `messaggi_tev1`, `lettera`); the v2/v2.1 tools are unchanged.

| File | Purpose | Origin (private foyer repo, branch `v3` @ `3d4ad13`) |
|---|---|---|
| `curve/curve_split.py` | `split`: fixed test set (20% of threads, stratified by the thread's type, seed 7) and nested subsets of the pool (whole threads, type-stratified order, seeds 7 and 11) → `split_paper3.json`, `curve_subsets.json` (row ids only). `examples`: the five typed examples per email in Tev1's format (contain email text; refuses to write inside the repo). | `backend/scripts/paper3/curva_dati.py` |
| `curve/curve_baseline.py` | Entry point for the baseline on the fixed test set: word 1-2 + character 3-5 TF-IDF, balanced logistic regression, one model per field and size; majority class of the test set → `curve_baseline.json`. CPU only. | `backend/scripts/paper3/train_decision_model.py` (`--mode curva --sistema baseline`) |
| `curve/curve_lora.py` | Kaggle kernel (private, GPU T4 ×2): installs `peft`/`transformers`, tests the flash-linear-attention kernel against the PyTorch path, runs `decision_model.py --system lora` → `curve_<model>_<seed>.json`. Edit the constants (base model, seed, sizes) per session. | `backend/scripts/paper3/kaggle/foyer_paper3_lora.py` (`ESPERIMENTO='curva'` path only) |
| `curve/decision_model.py` | *(supporting module)* The trainer of the curve: data loading, metrics, the baseline, the LoRA (Tev1 chat template, answer = option letter, loss on the answer only, early stop on a 10% thread-grouped validation taken from the training subset, best epoch kept in memory, prediction = distribution over the option letters), one process per GPU. | `backend/scripts/paper3/train_decision_model.py` (`--mode curva` path; the per-user CV, shared and transfer modes are not included) |
| `curve/curve_plot.py` | A 16:9 two-panel figure (type; mean of the three switches) with min/max bands across seeds → `curva.pdf`, `curva.png`. | `backend/scripts/paper3/curve_plot.py` + `curva_risultati.py` |
| `curve/curve_tables.py` | Every number of the paper: `numbers.tex` (macros), `tab_main.tex`, `tab_hyp.tex` (the five pre-registered hypotheses decided by script), appendix tables and `fig_curve.pdf/png`. | `make_curve_table.py` of the paper's sources (setup counts now read from `split_paper3.json`) |

Running (Python 3.12; the paper used scikit-learn 1.8.0, numpy 1.26.4; on Kaggle the latest `peft`/`transformers`):

```bash
pip install scikit-learn numpy matplotlib
# 1. split: fixed test set and nested subsets (ids only)
python curve/curve_split.py split --data labels.jsonl --out split/
# 2. baseline on the fixed test set (CPU, about a minute)
python curve/curve_baseline.py --user 1 --consent u1=consent_u1.json --data labels.jsonl \
    --split split/split_paper3.json --subsets split/curve_subsets.json --out results/
# 3. LoRA on Kaggle (T4 x2): a private dataset with curve/decision_model.py, systemone_email.py, labels.jsonl,
#    split_paper3.json, curve_subsets.json and consenso_u1.json; kernel = curve/curve_lora.py.
#    The same run elsewhere: python curve/decision_model.py --system lora --model tev1-4b \
#    --base-model togethercomputer/Tev1-4B-experimental --seeds 7 --sizes 0,100,400,all --gpus 1 ... (same inputs)
#    Copy the curve_tev1-*_<seed>.json files into results/.
# 4. tables, macros, hypotheses and figures
python -X utf8 curve/curve_tables.py --results results/ --split split/split_paper3.json --prereg-date 2026-11-05 \
    --registration-m <folder with tab_lora_u1.tex and tab_baseline_u1.tex> --out generated/
python curve/curve_plot.py --results results/ --out figure/
```

Inputs:
- **Corpus:** the same JSONL as above (`id` integer, `thread`, `state`, `labels` with all five fields; booleans may be `true`/`false` or `sì`/`no`). Rows with `"source": "system"` (an uncorrected system decision) are not labels and are skipped.
- **`split_paper3.json`** (ids only): `{"dati_sha256": "<sha256 of the corpus file>", "righe": N, "thread": T, "seme": 7, "frazione_test_thread": 0.2, "test_ids": [...], "pool_ids": [...], "conteggi": {"test": {"righe", "thread", "per_campo"}, "pool": {...}}}`.
- **`curve_subsets.json`** (ids only): `{"dati_sha256": "...", "semi": {"7": {"50": [...], "100": [...], ..., "all": [...]}, "11": {...}}, "taglie_effettive": {...}, "conteggi": {...}}`; each size contains the previous one.
- **Consent:** `{"owner": "1", "train_own": true, "updated_at": "..."}` for the person whose labels are used; without it the trainer does not start. Its sha256 is recorded in the results.
- The runs check the corpus sha256 against the split files. The result JSON keys are in Italian (`semi`, `taglie`, `campi`, `etichette`, ...), as in the paper's runs.

**No data and no weights are released:** no emails, labels, split files, per-email predictions or LoRA weights are in this repository. The LoRA weights were never saved (the best epoch is kept in memory and discarded after evaluation), and the private Kaggle dataset was deleted after each run.

## Changelog

- **v2.3** (November 2026): `revision_v2.2/` — the scripts behind paper v2.2 (arXiv v4): Holm-corrected tests, cost-chosen thresholds on the 1,500-character corpus, policy table and cost curves, repeated-CV baseline, relationship control with the paying-client definition. See `revision_v2.2/README.md`.

- **v2.2** (November 2026): `curve/` — learning-curve scripts of *Train After You Measure* (split, baseline, Kaggle LoRA kernel, trainer, figure, tables and hypothesis decisions); Tev1 native decision format (`decisione`, `messaggi_tev1`, `lettera`) added to `systemone_email.py`, comments translated to English.
- **v2.1** (November 2026): `jeva_backend.py` and `--backend jeva` in `decision_bench.py` and `project_bench.py` (`--jeva-cache`, `--native-null`), for the Jeva-4B rows of paper v2.1.
- **v2** (October 2026): the paper's questions verbatim, all five fields, hosted decision-model bench, metrics, cost threshold, project assignment and multi-annotator agreement.
- **v1**: System One client, benchmark harness, online backends, Cohen's κ, annotator instructions.

## Licence

Code: MIT. Instructions and documentation: CC BY 4.0.
