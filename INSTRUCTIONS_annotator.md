# Instructions for the annotation — 60 emails

Thank you for your time. This takes about 45–60 minutes. The goal is to measure how much two people agree when reading the same mailbox, so **there are no right answers to guess**: label according to what you read, without looking for confirmation. You do not need to know the recipient: work from the text and the sender.

The file `annotazione_60_cieca.csv` has 60 rows. For each one fill in the empty columns with the allowed values below. If a row is truly ambiguous, put `1` in `dubbio` (doubt) and choose anyway. The `note` column is free text.

Most emails are in Italian, some in English. Column names and label values are in Italian (they must match the system's vocabulary); their meaning is explained here.

## `tipo` — what the message says (one choice)

- `richiesta` (request) — the sender asks the recipient for something, or an action or reply from the recipient is needed. An automated notice saying "action required" counts.
- `aggiornamento` (update) — information, notification, confirmation or outcome, with no request: bookings, security codes, login alerts, calendar invitations, status reports, receipts, invoices, threads where the recipient is only in copy.
- `offerta` (offer) — a commercial proposal aimed at the recipient (a quote, a personal offer, a reply to their price request).
- `newsletter` — periodic or promotional mass mailing from a brand or service the recipient subscribed to.
- `spam` — unsolicited commercial mail from an unknown sender, or phishing.
- `altro` (other) — everything else, including messages **sent by the recipient himself** (sender = the recipient's own address).

The type describes the message, **not who sends it**: an email from a client can be `richiesta` or `aggiornamento` like any other.

## `urgenza` — how soon something must be done (one choice)

- `alta` (high) — an action is needed within about 48 hours, or there is an appointment today.
- `media` (medium) — an action or a careful reading is needed, but with no near deadline (security notices, time confirmations, failed builds).
- `bassa` (low) — no action required: informational, receipt, notification, newsletter.

## Three switches — `0` or `1`

Imagine an assistant that handles the mailbox for the recipient. For each email decide:

- `serve_risposta` (needs reply) = 1 if the sender **expects a written reply** from the recipient (the assistant would draft one).
- `azione_48h` (action within 48h) = 1 if there is **something to do within 48 hours** (the assistant would create a task): pay, activate, confirm, show up, fix an outage.
- `e_rumore` (is noise) = 1 if the email **should be ignored** by the assistant: automated notifications, marketing, receipts, things already done. It is kept only as context.

The three columns are independent; an email can have `serve_risposta=1` and `azione_48h=1` together. An email with `e_rumore=1` usually has the other two at 0.

## Two practical rules

- Empty body or signature only: decide from sender and subject; if that is not enough, `dubbio=1`.
- Do not look the sender up online and do not ask the recipient: only what you see in the file counts.

When you are done, send the completed CSV back. Nothing else is needed.
