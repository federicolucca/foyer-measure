# Istruzioni per l'annotazione — 60 email

Grazie per il tempo. Servono circa 45–60 minuti. Lo scopo è misurare quanto due persone concordano nel leggere la stessa posta, quindi **non esistono risposte giuste da indovinare**: etichetta secondo quello che leggi, senza cercare conferme. Non hai bisogno di conoscere il destinatario: lavora dal testo e dal mittente.

Il file `annotazione_60_cieca.csv` ha 60 righe. Per ciascuna compila le colonne vuote con i valori ammessi qui sotto. Se una riga è davvero ambigua, metti `1` in `dubbio` e scegli comunque. In `note` puoi scrivere quello che vuoi.

## `tipo` — che cosa dice il messaggio (una sola scelta)

- `richiesta` — chi scrive chiede qualcosa al destinatario, oppure serve una sua azione o risposta. Vale anche un avviso automatico che dice «azione richiesta».
- `aggiornamento` — informazione, notifica, conferma o esito, senza richiesta: prenotazioni, codici di sicurezza, avvisi di accesso, inviti di calendario, report di stato, ricevute, fatture, thread in cui si è solo in copia.
- `offerta` — proposta commerciale mirata al destinatario (preventivo, offerta personale, risposta a una sua richiesta di prezzo).
- `newsletter` — invio di massa periodico o promozionale da un brand o servizio a cui ci si è iscritti.
- `spam` — commerciale non richiesto da mittente sconosciuto, o phishing.
- `altro` — tutto il resto, compresi i messaggi **inviati dal destinatario stesso** (mittente = l'indirizzo del destinatario stesso).

Il tipo descrive il messaggio, **non chi lo manda**: un'email da un cliente può essere `richiesta` o `aggiornamento` come qualsiasi altra.

## `urgenza` — quanto presto va fatto qualcosa (una sola scelta)

- `alta` — serve un'azione entro circa 48 ore, oppure c'è un appuntamento in giornata.
- `media` — serve un'azione o una lettura attenta, ma senza scadenza vicina (avvisi di sicurezza, conferme di orario, build falliti).
- `bassa` — nessuna azione richiesta: informativa, ricevuta, notifica, newsletter.

## Tre interruttori — `0` o `1`

Immagina un assistente che gestisce la posta per il destinatario. Per ogni email decidi:

- `serve_risposta` = 1 se il mittente **aspetta una risposta scritta** dal destinatario (l'assistente preparerebbe una bozza).
- `azione_48h` = 1 se c'è **qualcosa da fare entro 48 ore** (l'assistente creerebbe un task): pagare, attivare, confermare, presentarsi, sistemare un guasto.
- `e_rumore` = 1 se l'email **va ignorata** dall'assistente: notifiche automatiche, marketing, ricevute, cose già fatte. Resta solo come contesto.

Le tre colonne sono indipendenti; un'email può avere `serve_risposta=1` e `azione_48h=1` insieme. Un'email con `e_rumore=1` di solito ha gli altri due a 0.

## Due regole pratiche

- Corpo vuoto o solo firma: decidi da mittente e oggetto; se non basta, `dubbio=1`.
- Non cercare il mittente su internet e non chiedere al destinatario: vale solo quello che vedi nel file.

Quando hai finito, rimanda il CSV compilato. Non serve altro.
