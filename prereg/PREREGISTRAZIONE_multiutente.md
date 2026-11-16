<!--
English note (added on publication, 16 Nov 2026). This is registration M of "Train After You Measure"
(arXiv companion of 2610.32955). It is the file docs/paper3/PREREGISTRAZIONE_multiutente.md exactly as committed in
the author's private repository foyer (branch v3), commit 56b2970 (2026-10-30 14:47:28 +0100). The text is kept
verbatim in Italian. Only privacy redactions were allowed; none was needed in this file, so the text below is
byte-identical to the original. Hypotheses, thresholds, dates and design are unchanged.
-->

# Pre-registrazione — paper 3, Foyer multi-utente ("Train After You Measure")

Committata il 30/10/2026 sul branch `v3`, **prima di qualsiasi run sulle etichette di un utente diverso dal
proprietario (utente 1)**. Le ipotesi, le soglie e le regole di decisione qui sotto non si cambiano dopo aver
visto i dati di un secondo utente; un'ipotesi aggiunta dopo va segnata come esplorativa, con la data.

Le ipotesi si valutano solo con `backend/scripts/paper3/verifica_ipotesi_multiutente.py`, che legge i
`metrics.json` scritti da `train_decision_model.py` ed `evaluate_transfer.py`. Niente numeri a mano.

## Definizioni comuni

- **Campi**: tipo, urgenza, serve_risposta, azione_48h, e_rumore (le cinque domande tipizzate del paper v2).
- **Etichette di un utente**: solo righe verificate da una persona (conferme e correzioni del proprietario,
  `source` diverso da `system` nell'export; per l'utente 1 anche il corpus v3 del paper). Le decisioni del
  sistema non corrette non sono etichette.
- **Valutazione**: accuracy out-of-fold, StratifiedGroupKFold a 5 fold per thread (mittente + oggetto senza
  prefissi Re/Fwd), shuffle, seme 0, come nel paper v2. Un thread non sta mai fra train e test.
- **Baseline**: TF-IDF di parole (1–2, 20 000 feature, sublinear) + regressione logistica bilanciata, la B del
  paper v2, addestrata sulle etichette dello stesso utente con la stessa CV.
- **Modello per utente**: fine-tuning LoRA di un decision model tipizzato (default Tev1 4B) sulle etichette
  dell'utente, loss solo sui token della risposta.
- **Modello shared**: un solo modello sulle etichette di tutti gli utenti con consenso (c), con l'utente nello
  stato (`"owner": "uN"`).
- **Zero-shot**: lo stesso decision model senza training (Jev, Nimble v4-max, Tev1), con le domande del paper v2.
- **Punti**: differenza assoluta di accuracy × 100.

## Ipotesi

- **H1.** Con ≥ 200 etichette, il modello per utente supera la baseline sul **tipo** di **≥ 3 punti**.
  Si valuta per ogni utente con ≥ 200 etichette, sul punto della curva più vicino a 200 da sopra.
  Confermata se vale per la maggioranza degli utenti valutabili (con un solo utente: per quell'utente).
- **H2.** La curva di apprendimento del modello per utente sul **tipo** satura fra **300 e 500 etichette**:
  il primo punto n con guadagno < 1 punto per +100 etichette (pendenza fra n e il punto successivo, riportata a
  100 etichette) cade in [300, 500]. Confermata se vale per la maggioranza degli utenti con ≥ 500 etichette.
- **H3.** Un modello addestrato sull'utente A, valutato sulle etichette dell'utente B, fa meglio dello
  zero-shot migliore su **tipo** e su **e_rumore**, ma **non** su **serve_risposta**. Confermata se vale per la
  maggioranza delle coppie (A, B) ordinate.
- **H4.** Il modello shared con token utente non perde più di **2 punti** rispetto al modello per utente, sullo
  stesso utente e sullo stesso campo, per tutti e cinque i campi. Confermata se vale per la maggioranza degli
  utenti.
- **H5.** Le classi **offerta** e **spam** restano sotto **F1 0,3** per tutti i sistemi addestrati (baseline,
  per utente, shared) finché un utente non ha **≥ 15 esempi** di ciascuna. Si valuta su ogni (utente, classe)
  con meno di 15 esempi: confermata se F1 < 0,3 in tutti i casi; se nessun caso è valutabile, resta non valutata.

## Cosa si era già visto al momento del commit

Solo dati dell'utente 1: i risultati del paper v2 (B, zero-shot, κ di due annotatori) e, il 30/10 prima di questo
commit, una curva di apprendimento della **baseline** sui suoi 722 esempi (prova dello script). Nessun modello LoRA
addestrato, nessun dato di altri utenti. Quella prova mostra già un caso contro H5: la baseline dell'utente 1 ha F1
0,36 su *offerta* con 8 esempi. H5 resta scritta come era stata proposta prima di vederlo; il caso conta.

## Regole

- D (cross-utente, H3) parte solo quando esiste un secondo utente con consenso.
- Con meno utenti di quelli richiesti un'ipotesi è riportata come **non valutabile**, non come confermata.
- Ogni run registra i consensi verificati (`train_config.json`); un run senza il consenso giusto non parte.
- Seme 0 ovunque; split per thread salvati con il loro hash; hash sha256 dei dati in ingresso.
