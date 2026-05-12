# AGENTS.md

## Repository Reality Check
- This repo is a small single-binary Go app with one Python helper and one embedded frontend file.
- No existing agent/rule files were found (`.cursor/rules`, `.cursorrules`, `.github/copilot-instructions.md`, `claude.md`, `agents.md`).
- No test files (`*_test.go`) or CI workflow files were found.

## Essential Commands

### Run / Build / Dependency Maintenance
- `make run` → runs `go run .`
- `make build` → builds `./vectorview`
- `make tidy` → runs `go mod tidy`
- `make install-dep` → `go get github.com/joho/godotenv`

### Direct Commands Seen in README
- `go mod tidy`
- `python3 -m pip install numpy`
- `go run .`
- `go build -o vectorview .`

### Verification Command (use after changes)
- `go test ./...`
  - There are currently no Go test files, so this mainly validates compile/build integrity.

## Configuration
- Copy `.env.example` to `.env`.
- Observed env vars:
  - `QDRANT_URL` (default `http://localhost:6333`)
  - `QDRANT_API_KEY`
  - `VECTORVIEW_PORT` (default `7433`)
  - `VECTORVIEW_MAX_POINTS` (default `2000`)
  - `VECTORVIEW_REDIS_URL` (optional Redis projection cache)
  - `VECTORVIEW_CACHE_TTL_SECONDS` (default `600`)
  - `VECTORVIEW_SEMANTIC_PROVIDER` (default `ollama`)
  - `VECTORVIEW_OLLAMA_URL` (default `http://localhost:11434`)
  - `VECTORVIEW_EMBED_MODEL` (default `nomic-embed-text`)

## Architecture and Data Flow

### High-level shape
- `main.go`: backend server, Qdrant HTTP client, API handlers, Redis-backed projection cache, semantic embedding calls, static embed.
- `pca_gpu.py`: subprocess worker for projection (`pca` / `random` / `tsne` / `umap`).
- `static/index.html`: entire frontend (HTML + CSS + JS + shaders + controls) in one file.

### Backend request flow
1. Browser loads `/` from Go `http.FileServer` over embedded `static/*`.
2. Frontend calls API routes on same origin:
   - `GET /api/collections`
   - `GET /api/points?collection=&limit=&projection=&vector_name=&append_from=`
   - `GET /api/search?collection=&q=&limit=&projection=&vector_name=`
   - `GET /api/semantic-search?collection=&target_collection=&q=&limit=&projection=&vector_name=`
3. Backend talks to Qdrant using raw HTTP (`/collections`, `/collections/{name}`, `/collections/{name}/points/scroll`, `/collections/{name}/points/search`).

### Projection paths (important)
- `/api/points`: Go serves from Redis cache when available; otherwise spawns `pca_gpu.py` (`exec.CommandContext`) and stores response in Redis.
- `/api/points` also supports incremental append (`append_from`) for random projection when a cached base subset exists.
- `/api/search` and `/api/semantic-search` both call `pca_gpu.py` via stdin for projection consistency.
- Projection worker chooses the most common dense vector dimension (or requested named vector) and skips incompatible vectors.

## Code Organization Notes
- Backend is a single `package main` file; shared logic is not split into packages yet.
- Frontend has no Node/build tooling; it imports Three.js from CDN via import map.
- Static assets are embedded with `//go:embed static/*`, so frontend changes require rebuilding/rerunning Go app.

## Conventions and Patterns Observed
- Go style is `gofmt`-compatible with tabs and minimal abstraction.
- Qdrant access is via handwritten structs and `encoding/json`, not an SDK.
- Handlers call `setCORS(w)` and generally write JSON directly with `json.NewEncoder`.
- Keyword search uses Qdrant `scroll` + `filter.should` over multiple payload keys and case variants.
- Semantic search embeds query text via Ollama and runs Qdrant nearest-neighbor search.
- `/api/collections` includes projection status fields (`projection_ready`, `projection_note`) by probing sample vectors.
- Frontend clustering color key is derived from `payload.file_source` prefix (`extractClusterKey`), not from `entity_type`.

## Gotchas / Non-obvious Behaviors
- `pca_gpu.py` is discovered by `pcaScript()` either:
  - next to executable **only when binary name is exactly `vectorview`**, or
  - fallback path `pca_gpu.py` from current working directory.
- `QDRANT_API_KEY` is applied in Go HTTP client calls, but the Python worker currently receives only `qdrant_url` and does not set API key headers.
  - Result: `/api/points` can fail against API-key-protected Qdrant even if other Go-backed routes work.
- UMAP and t-SNE projections require optional Python deps (`umap-learn`, `scikit-learn`) in runtime environment.
- Redis cache is only active when `VECTORVIEW_REDIS_URL` is configured and ping succeeds.
- Collection picker options are disabled when `projection_ready` is false; labels include point count, vector dim, and projection note.
- `setCORS` sets `Access-Control-Allow-Origin: *` and `Content-Type: application/json` globally for API responses.

## Practical Agent Workflow for This Repo
1. Read `main.go`, `pca_gpu.py`, and `static/index.html` before changing behavior (cross-language flow is tightly coupled).
2. If changing projection behavior, verify both `/api/points` and `/api/search` paths (they are intentionally different).
3. After edits, run `go test ./...`.
4. If you changed Python worker behavior, also run at least one end-to-end manual check via `make run` against a live Qdrant collection.

## File Map
- `main.go` — server + API + inline PCA + embed
- `pca_gpu.py` — full-collection PCA worker
- `static/index.html` — all UI/rendering logic
- `Makefile` — canonical developer shortcuts
- `.env.example` — runtime configuration template
