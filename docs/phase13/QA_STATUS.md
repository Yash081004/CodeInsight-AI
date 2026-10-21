# Phase 13 — QA status

## Latest full-stack run

**Directory:** `test-results/qa_20261021_091943`

| Metric | Value |
|--------|------:|
| passed | **36** |
| failed | 0 |
| blocked | 0 |
| skipped | 0 |
| warning | 2 (model quality — expected weak classifier) |

Backend **reused** at `http://127.0.0.1:8000`; frontend **reused** at `http://127.0.0.1:5173`.

## Harness fix (Phase 13)

On Windows, Vite defaulted to `localhost` only; QA probes `127.0.0.1:5173`. Added `server.host: '127.0.0.1'` in `frontend/vite.config.js`.

**Runner policy:** QA reuses an existing backend on `:8000` and does not start duplicate uvicorn.
