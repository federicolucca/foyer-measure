# curve/ — learning curve of *Train After You Measure*

The scripts of the pre-registered learning curve in *Train After You Measure: a Pre-Registered Learning Curve for
Fine-Tuning Typed Decision Models on One User's Labels* (companion to arXiv:2610.32955). The paper cites code tag
`v2.2`. These six files are the ones at that tag. This README was added on `main` right after the tag, and the same
information is in the top-level README of the tag (section "v2.2").

**No data are released:** no emails, labels, split files, per-row predictions or result JSON files. **No weights are
released either:** the LoRA adapters were never saved. Neither are the owner's project descriptions.

## Files

| File | What it does |
|---|---|
| `curve_split.py` | `split`: the fixed test set (20% of the threads of each type, seed 7) and the nested training subsets of the pool (50, 100, 200, 300, 400, all; whole threads in a type-stratified order; seeds 7 and 11). Writes `split_paper3.json` and `curve_subsets.json`, which contain row ids only. `examples`: the five typed examples per email in Tev1's native format, for inspection. They contain email text, so the script refuses to write them inside the repository. |
| `curve_baseline.py` | The baseline on the fixed test set: TF-IDF of word 1–2-grams and character 3–5-grams with a class-balanced logistic regression, one model per field and training size. Writes `curve_baseline.json` and runs on CPU. |
| `curve_lora.py` | Kaggle kernel (private, GPU T4 ×2). Installs `peft`/`transformers`, checks the flash-linear-attention kernel against the PyTorch path, then runs `decision_model.py --system lora` for one model and seed. Writes `curve_<model>_<seed>.json`. Set the constants at the top for each session. |
| `decision_model.py` | The trainer used by the two scripts above: data loading and consent check, metrics (accuracy, macro-F1, per-class F1, 10-bin ECE, latency), the baseline, and the LoRA. |
| `curve_plot.py` | The two-panel figure (type; mean of the three switches), with min–max bands across seeds. Reads the result JSON files and writes `curva.pdf` and `curva.png`. |
| `curve_tables.py` | Every number of the paper, from the result JSON files: `numbers.tex` (macros), the two body tables, the appendix tables, the five pre-registered hypotheses decided by rule, and `fig_curve.pdf`/`.png`. |

## Setup used in the paper

- **Models (Hugging Face):** `togethercomputer/Tev1-0.8B-experimental` and `togethercomputer/Tev1-4B-experimental`. Each budget 0 is the same model without an adapter (zero-shot).
- **LoRA:** rank 16, alpha 32, dropout 0.05, all linear layers. Learning rate 2e-4 (constant, AdamW), effective batch 16 (micro-batch 1 with accumulation), at most 1024 tokens, at most 3 epochs.
- **Training details:** early stopping on a 10% thread-grouped validation split taken from the training subset, keeping the best epoch in memory. The loss is on the answer letter only. Prediction is the distribution over the option letters.
- **Seeds:** 7 for the test split. 7 and 11 for the subsets, the validation split, initialisation and data order.
- **Sizes run:** the 0.8B model ran 0, 50, 100, 200, 300, 400 and all. The 4B model ran 0, 100, 400 and all. Zero-shot used seed 7 only.
- **Hardware:** Kaggle, NVIDIA T4 ×2 (one size per GPU), fp16 base weights with the LoRA in fp32. The baseline ran on CPU.

## Reproducing the figure and the tables from the result files

The result JSON files (`curve_baseline.json`, `curve_tev1-0.8b_{7,11}.json`, `curve_tev1-4b_{7,11}.json`) and the
split file stay private, so the steps below need your own runs (or the author's files).

```bash
pip install scikit-learn numpy matplotlib
python curve/curve_split.py split --data labels.jsonl --out split/
python curve/curve_baseline.py --user 1 --consent u1=consent_u1.json --data labels.jsonl \
    --split split/split_paper3.json --subsets split/curve_subsets.json --out results/
# LoRA: run curve/curve_lora.py as a private Kaggle kernel (T4 x2), or locally:
python curve/decision_model.py --system lora --model tev1-4b --base-model togethercomputer/Tev1-4B-experimental \
    --user 1 --consent u1=consent_u1.json --data labels.jsonl --split split/split_paper3.json \
    --subsets split/curve_subsets.json --seeds 7 --sizes 0,100,400,all --out results/
# tables, macros, hypotheses and the paper's figure
python -X utf8 curve/curve_tables.py --results results/ --split split/split_paper3.json --prereg-date 2026-11-05 \
    --registration-m <folder with tab_lora_u1.tex and tab_baseline_u1.tex> --out generated/
python curve/curve_plot.py --results results/ --out figure/
```

The input format (corpus JSONL, the split files and the consent file) is described in the top-level README.
