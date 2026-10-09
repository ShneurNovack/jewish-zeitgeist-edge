# jewish-zeitgeist-edge

Cloudflare Worker that backs the Jewish Zeitgeist Base44 app.

- `POST /embed` turns text into multilingual embeddings with Workers AI (`@cf/baai/bge-m3`), returned as int8 base64 vectors. The Base44 pipeline uses them to cluster headlines, posts and search trends into topics, across English and Hebrew.
- `GET /fetch?url=` relays a public feed request for the handful of sources that refuse Base44's servers.

- A cron trigger runs every 10 minutes and calls the Base44 pipeline stages in order: `ingest`, `cluster`, `probe` (the search check on hot topics), `score`, `enrich`. `POST /run` does the same on demand. Running the schedule here costs nothing, where a Base44 workflow is billed per run.

All routes except `/health` need the shared key in the `x-zg-key` header. The key is stored as the `ZG_KEY` secret on the Worker and as `EDGE_KEY` in the Base44 app.

Deploys automatically from `main` through Cloudflare Workers Builds.
