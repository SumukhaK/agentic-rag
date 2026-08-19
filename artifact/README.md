# artifact/

A standalone, self-contained copy of the pipeline demo walkthrough that is
also published on Claude's own artifact hosting. `walkthrough.html` needs no
build step and no server — open it directly in a browser and it renders
identically to the hosted version (same content, styling, and light/dark
theme behavior), since it's the same page with an explicit `<!doctype html>`/
`<html>`/`<head>`/`<body>` wrapper added around it (the hosted version omits
that wrapper and lets the artifact host supply it at publish time).

Every request/response, retrieved excerpt, and metric shown in the page is
copied verbatim from a real run of `python -m agentic_rag.evaluation.runner`
against this project's hand-verified corpus (`eval/corpus/`,
`eval/questions.json`) — see [`eval/README.md`](../eval/README.md) for how to
reproduce it. Nothing on the page is mocked or scripted.

This is a point-in-time snapshot, not a live view of the running system — if
the eval corpus or pipeline behavior changes, this file is not automatically
regenerated and may drift from the live examples in `eval/README.md`.
