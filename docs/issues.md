# Issues

Problem-resolution log: one Problem Record (`PR-N`) per defect found in operation or on review,
tracked to a verified closure. A record holds the durable facts; an investigation trail lives in
a linked note, not here.

## Record structure

- **Problem** — what was observed; frozen at report.
- **Impact / Severity** — who or what is affected, and how badly.
- **Cause** — the confirmed mechanism with evidence; frozen once confirmed, corrections appended.
- **Resolution** — dated, append-only; the corrective action and the change that shipped it.
- **Verification** — how the fix was confirmed, and closure.

Status: `[ ]` open · `[~]` analyzed, fix pending · `[x]` closed (resolved and verified) · `[-]`
rejected (won't fix or not a bug, reason in Resolution).

---

## At a glance

| # | Issue | Severity | Status |
|---|---|---|---|
