<!--
English note (added on publication, 16 Nov 2026). This is registration C of "Train After You Measure"
(arXiv companion of 2610.32955). It is the file docs/paper3/PREREGISTRAZIONE_curva.md as committed in the author's
private repository foyer (branch v3): the registration in commit 96a3b7b (2026-11-05 11:19:28 +0100, lines up to
the section "Codice"), and Amendment 1 appended in commit 3d88d63 (2026-11-05 12:53:29 +0100; it replaces the first
version of the amendment, commit 046cf50, 11:46:26, as the amendment itself explains). The text is kept verbatim in
Italian. Only privacy redactions were made, each marked [redatto: ...]: the hash prefix of the private corpus, two
local file paths and the name of the private Kaggle dataset; the only addition is the word "(private)" after the
address of the original repository. Hypotheses, thresholds, dates and design are unchanged.
-->

# Pre-registrazione — paper 3, curva di apprendimento su test fisso ("Train After You Measure")

Committata il 05/11/2026 sul branch `v3` di foyer, **prima di qualsiasi run LoRA della curva e prima di aver
calcolato la baseline sul test fisso** (blocco B). Ipotesi, soglie e regole qui sotto non si cambiano dopo aver
visto i risultati; un'ipotesi aggiunta dopo va segnata come esplorativa, con la data. Un emendamento prima dei
run LoRA è ammesso solo come commit separato e datato.

Domanda: quante etichette servono perché un decision model tipizzato fine-tunato batta la regressione TF-IDF, e
quanto del divario zero-shot era mancanza di dati?

## Cosa si sapeva già al momento del commit (H1 non è cieca)

- **C1 su questi stessi dati** (`docs/paper3/REPORT_C1_lora_u1.md`, run unito `20261103T140513`): LoRA su
  `Qwen/Qwen3.5-4B` (la base di Tev1 4B), solo il campo tipo, CV a 5 fold per thread, 2 epoche, prompt in testo
  libero. Il LoRA resta **sotto** la baseline B (TF-IDF di sole parole) in accuracy sul tipo a **ogni** punto da 50
  a tutte le etichette (da −1,8 a −7,3 punti); la H1 di quella pre-registrazione (≥ 3 punti a 200) è refutata.
  Lì la pendenza del LoRA scende sotto 1 punto/100 a 300 etichette e poi la curva cala. Lo spam ha già F1 0,526 con
  10 esempi (caso contro un'ipotesi come H5).
- Quindi **H1 qui sotto non è un'ipotesi cieca**: chi l'ha scritta (il brief di Federico, prima della C1) e chi la
  registra sanno che la stessa famiglia di modelli, in CV, non ha battuto la baseline sul tipo. Il test fisso, il
  formato nativo di Tev1, i pesi Tev1, le 3 epoche con early stop e l'addestramento sulle cinque domande insieme
  sono le differenze rispetto alla C1.
- Della curva non si è visto nulla: lo split e i sottoinsiemi sono stati generati (solo conteggi per classe, sotto),
  nessun modello è stato addestrato o valutato sul test fisso, neanche la baseline.

## Dati, split e sottoinsiemi (blocco A)

- Corpus: `email_labels_v3.jsonl` (privato, fuori dal repo), 722 righe, sha256 [redatto: hash del corpus privato].
- Thread = il campo `thread` del corpus: **649 thread** (il brief dice 698; il raggruppamento mittente + oggetto
  della C1 ne dà 696). Il campo `thread` è più grossolano: ogni gruppo della C1 sta in un solo thread, quindi
  nessuna coppia dello stesso thread, per l'una o l'altra definizione, sta fra train e test.
- **Test fisso** (`docs/paper3/split_paper3.json`, solo id numerici): 20% dei thread di ciascun tipo (tipo del
  thread = il più frequente), seme 7 → **135 righe, 129 thread**. Per tipo: aggiornamento 79, newsletter 36,
  altro 11, richiesta 6, spam 2, offerta 1. Lo stesso per tutti i run; non si usa mai per scegliere nulla.
- **Pool di training**: le altre 587 righe (520 thread).
- **Sottoinsiemi** (`docs/paper3/curve_subsets.json`, solo id): annidati, presi per thread interi in un ordine
  stratificato per tipo (a ogni passo il tipo più in difetto rispetto alla sua quota nel pool prende il suo
  prossimo thread), semi **7** e **11**. Taglie effettive per entrambi i semi: **50, 100, 200, 300, 400, 587 (all)**.
  L'ultimo thread di una taglia può essere tagliato per arrivare alla taglia esatta (solo dentro il train).
- Generati da `backend/scripts/paper3/curva_dati.py split`.

## Sistemi

- **Baseline (B della curva)**: TF-IDF di parole 1–2 e di caratteri 3–5 (`char_wb`), sublinear, senza tetto di
  feature, + regressione logistica `class_weight='balanced'`, un modello per campo, addestrata su ogni
  sottoinsieme, valutata sul test fisso. Testo = mittente + oggetto + corpo senza tag e URL (come la B del paper v2).
  Iperparametri fissi, nessuna scelta. Con una sola classe nel train predice quella.
- **Maggioranza**: (a) la classe più frequente **del test** per campo (la linea del brief: un riferimento, non un
  sistema addestrabile); (b) la classe più frequente del sottoinsieme di training, per taglia.
- **FT**: LoRA su `togethercomputer/Tev1-0.8B-experimental` (**FT-0.8B**) e su
  `togethercomputer/Tev1-4B-experimental` (**FT-4B**). Un solo LoRA per taglia × seme sulle cinque domande
  (cinque esempi per email). r = 16, alpha = 32, dropout 0,05, moduli `all-linear` (attenzione, compresi i layer
  gated delta, e MLP; torre visiva esclusa), lr 2e-4 costante, AdamW, **3 epoche**, batch effettivo 16 (micro-batch
  1 con accumulo), max 1024 token, bf16 se la GPU lo regge altrimenti fp16 (T4: fp16 per i pesi base, LoRA in fp32).
  **Early stop**: validazione = 10% delle righe del sottoinsieme, per thread interi (seme del run); dopo ogni
  epoca la loss sulle risposte della validazione; resta l'epoca migliore. La validazione fa parte della taglia (il
  LoRA si addestra sul restante 90%). Stessi iperparametri per tutte le taglie.
- **Zero-shot** (taglia 0): lo stesso modello senza LoRA, stessa previsione.
- **Formato** (nativo Tev1, model card e `examples/decide.py` di togethercomputer/tev1): istruzione di sistema
  fissa; messaggio utente = JSON `{state, question, options[{label, key, description}]}`; chat template senza
  thinking; risposta = lettera + EOS; loss solo sui token della risposta. Stato = mittente, oggetto, corpo pulito a
  1.500 caratteri (accorciato se il prompt supera 1024 token); domanda e definizioni italiane del paper
  (Appendice B, `systemone_email.py`). Previsione = la distribuzione sulle lettere delle opzioni nella posizione
  della risposta, rinormalizzata: la scelta vincolata di `/v1/systemone` (choice per tipo e urgenza, choice sì/no
  per gli interruttori). Confidenza = probabilità dell'opzione scelta.
- Esempi esportati, per controllo, da `curva_dati.py esempi` (file locale [redatto: percorso], mai nel repo);
  il trainer li costruisce con le stesse funzioni.

## Metriche

Sul test fisso (135 righe), per campo: accuracy, macro-F1 (sulle classi presenti nel test), ECE a 10 bin, F1 per
classe, latenza mediana per domanda, confidenza media dei corretti e degli errati. **Punti** = differenza assoluta
di accuracy × 100. Con più semi su una taglia si usa la **media dei semi**, confrontando sistemi solo sugli stessi
semi; bande = min/max fra i semi. Output: `results/curve_baseline.json`, `results/curve_<modello>_<seme>.json`.
Le ipotesi si valutano con uno script che legge solo questi JSON (blocco D), niente numeri a mano.

## Ipotesi

- **H1.** FT-4B supera la baseline sul **tipo** di **≥ 3 punti** con **≥ 300 etichette**: si valuta a ogni taglia
  ≥ 300 eseguita per FT-4B (300, 400, all). Confermata se vale a tutte; refutata se manca anche a una.
- **H2.** La curva FT sul **tipo** satura fra **300 e 500** etichette: il primo n della curva con guadagno < 1 punto
  per +100 etichette (pendenza fra n e la taglia successiva, riportata a 100) è 300 o 400. Si valuta per ogni
  modello FT con la curva completa (50, 100, 200, 300, 400, all); confermata se vale per tutti i valutabili. Se
  nessuna pendenza scende sotto 1, refutata (non satura entro 587).
- **H3.** FT-0.8B **non supera** la baseline sul **tipo** a **nessuna** taglia: confermata se FT-0.8B − baseline
  ≤ 0 punti a ogni taglia eseguita (50 … all).
- **H4.** L'ECE del FT sul **tipo** è **< 0,10** con **≥ 200 etichette**: confermata se vale per FT-0.8B e FT-4B a
  ogni taglia ≥ 200 eseguita. L'ECE degli altri quattro campi si riporta ma non decide.
- **H5.** **offerta** e **spam** hanno F1 **< 0,3 ovunque**: per ogni sistema addestrato (baseline, FT-0.8B,
  FT-4B) e ogni taglia ≥ 50. Confermata se vale in tutti i casi. Lo zero-shot si riporta a parte.
  Avvertenza nota al commit: nel test ci sono **1 offerta e 2 spam**; la loro F1 dipende da una o due email e
  dai falsi positivi. Nel pool ci sono 7 offerte e 8 spam: nessuna taglia arriva a 15 esempi per classe.

## Ordine dei run e regole

- Ordine se il tempo GPU è poco: FT-0.8B tutte le taglie seme 7 → FT-4B taglie 0, 100, 400, all seme 7 → secondi
  semi (se non basta, seme 11 solo per 100 e 400). Le taglie non eseguite restano **non valutate**, non contano.
- Un'ipotesi con dati insufficienti (es. FT-4B senza curva completa per H2) è **non valutabile**, non confermata.
- I pesi LoRA non si salvano mai (né su disco né nell'output del notebook); il dataset Kaggle privato
  ([redatto: nome del dataset]) si cancella a fine run; nel notebook nessun testo.

## Deviazioni dal brief ([redatto: percorso locale del brief])

1. **Modelli**: il brief indica «HF `togethercomputer/tev1-0.8b` o quello che il README di Tev1 indica». Verificato
   su Hugging Face il 05/11/2026: i nomi esatti sono `togethercomputer/Tev1-0.8B-experimental` e
   `togethercomputer/Tev1-4B-experimental`, pubblici, non gated, safetensors, online dal 25/09/2026 (fine-tune
   supervisionati di `Qwen/Qwen3.5-0.8B` e `Qwen/Qwen3.5-4B`). La licenza dei pesi fine-tunati è «in via di
   definizione»: qui non si ridistribuiscono. La C1 era partita da `Qwen/Qwen3.5-4B` scrivendo che i pesi di Tev1
   4B non erano pubblici; per la curva questa deviazione non serve e non si applica. Un braccio sulla base Qwen
   (per collegare la curva alla C1) è ammesso solo come **esplorativo**.
2. **Repo**: foyer (gitlab federicolucca/foyer (private)), branch `v3`, `docs/paper3/` e `backend/scripts/paper3/`, non il
   branch `paper3` di `secondbrain`.
3. **Thread e taglia del test**: 649 thread dal campo `thread` (non 698); test 135 righe (non ≈ 145); pool 587.
4. **Nimble 9B**: non previsto.
5. **Formato della risposta**: lettera dell'opzione (formato nativo di Tev1) invece della chiave; la chiamata
   tipizzata è riprodotta con la scelta vincolata sulle lettere in Transformers, non via Ollama. La latenza è
   quella di un forward su T4, non di `/v1/systemone`.
6. **Maggioranza**: oltre a quella del test (brief) si riporta quella del sottoinsieme di training.
7. **Iperparametri non fissati dal brief** e fissati qui: max 1024 token, lr costante, char_wb per i caratteri della
   baseline, nessun tetto di feature, seme del run = seme della curva (sottoinsieme, validazione, init e ordine).

## Codice

- `backend/scripts/paper3/curva_dati.py` — split, sottoinsiemi, esempi (blocco A).
- `backend/scripts/paper3/train_decision_model.py --mode curva` — baseline (`--sistema baseline`) e LoRA.
- `backend/scripts/paper3/kaggle/foyer_paper3_lora.py` con `ESPERIMENTO = 'curva'` — il kernel (non lanciato al
  momento del commit).

## Emendamento 1 (riscritto) — 05/11/2026, prima di qualsiasi risultato LoRA della curva

La prima versione di questo emendamento (commit `046cf50`) attribuiva al proprietario la scelta delle basi Qwen3.5
al posto di Tev1. **Non era una decisione del proprietario**: era un'istruzione sbagliata dell'agente che
coordina il lavoro, basata sulla convinzione che i pesi di Tev1 non fossero pubblici. Quella deviazione è
**ritirata**: valgono i modelli del brief e della sezione «Sistemi», cioè `togethercomputer/Tev1-0.8B-experimental`
(**FT-0.8B**) e `togethercomputer/Tev1-4B-experimental` (**FT-4B**). Le basi Qwen3.5 **non entrano nella curva**,
né ora né come braccio esplorativo successivo.

Un run con la base `Qwen/Qwen3.5-0.8B` era partito su Kaggle (circa 35 minuti) ed è stato fermato cancellando il
kernel prima della fine: **nessun suo output è stato scaricato né aperto**. Al momento di questo commit nessun
risultato LoRA della curva è stato visto; i soli risultati aperti sono quelli della baseline (blocco B, committati
in `e06d414`, dopo la pre-registrazione).

Ordine dei run (resta): FT-0.8B tutte le taglie + zero-shot, seme 7 → FT-4B taglie 0, 100, 400, all, seme 7 →
seme 11 (tutte le taglie se la quota GPU lo permette, altrimenti 100 e 400). Ottimizzazioni già su `v3`: una taglia
per GPU su T4 x2, kernel `flash-linear-attention` se passa la prova numerica. In questo formato la previsione è un
solo forward per domanda (le opzioni sono lettere di un token), quindi la cache del prompt della C1 non serve.
