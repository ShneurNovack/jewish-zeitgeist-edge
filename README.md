# jewish-zeitgeist-edge

Cloudflare Worker and offline job behind the Jewish Zeitgeist app (Base44), which answers one question for
Meaningful Minute: what should we post about right now?

How the engine works, which algorithms were chosen and why: [docs/ENGINE.md](docs/ENGINE.md).

## Worker (`src/`)

- `POST /embed` turns text into multilingual embeddings with Workers AI (`@cf/baai/bge-m3`), returned as int8 base64
  vectors. The engine uses them to file headlines, posts and search trends into stories across English and Hebrew.
- `GET /fetch?url=` relays a public feed request for the handful of sources that refuse Base44's servers.
- A cron trigger runs every 10 minutes and calls the pipeline stages in order: `ingest`, `cluster`, `probe`,
  `analyze`, `structure`, `mm`, `enrich`, and `analyze` once more when the language read changed anything.
  `POST /run` does the same on demand. Running the schedule here costs nothing, where a Base44 workflow is billed
  per run.
- `POST /analyst/pull` and `/analyst/push` serve the offline job below.

All routes except `/health` and `/analyst/*` need the shared key in the `x-zg-key` header. The key is stored as the
`ZG_KEY` secret on the Worker and as `EDGE_KEY` in the Base44 app. It is never committed here.

Deploys automatically from `main` through Cloudflare Workers Builds.

## Offline analyst (`analyst/`, `.github/workflows/analyst.yml`)

A daily GitHub Actions job (Python: scikit-learn, LightGBM) that audits the live clustering with HDBSCAN and trains
the model behind "MM fit" once enough of Meaningful Minute's post history has been imported. It holds no secrets:
it presents a GitHub OIDC token and the Worker checks that the token belongs to this repository's workflow on
`main` (`src/oidc.js`). Results appear on the Engine page of the app.

Run it locally with `ZG_KEY` and `APP_URL` set: `python analyst/run.py`.
