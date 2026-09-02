# AGENTS.md

## Repository Reality Check
- This repo is a small single-binary Go app with one Python helper and one embedded frontend file.
- No existing agent/rule files were found (`.cursor/rules`, `.cursorrules`, `.github/copilot-instructions.md`, `claude.md`, `agents.md`).
- Go unit test file `main_test.go` is present; CI runs `go test ./...`.
- FrontPocket and MindDrill collections (`frontpocket_memory`, `minddrill_research_thoughts`, `minddrill_chat_memory`, `fp_reflections`) are fully supported alongside meta_bridge.

## Essential Commands

## Operator Preferences
- If the user says "update docs and push", treat it as: update docs + commit + push.

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
  - Validates compile/build integrity and executes unit test suite in `main_test.go`.

## Configuration
- Copy `.env.example` to `.env`.
- Observed env vars:
  - `QDRANT_URL` (default `http://localhost:6333`)
  - `QDRANT_API_KEY`
  - `VECTORVIEW_PORT` (default `7433`)
  - `VECTORVIEW_MAX_POINTS` (default `2000`)
  - `VECTORVIEW_REDIS_URL` (optional Redis projection cache)
  - `VECTORVIEW_CACHE_TTL_SECONDS` (default `600`)
  - `VECTORVIEW_SEMANTIC_PROVIDER` (default `auto`, supports `auto`, `ollama`, `openrouter`, `openai`)
  - `VECTORVIEW_OLLAMA_URL` (default `http://localhost:11434`)
  - `VECTORVIEW_EMBED_MODEL` (default `nomic-embed-text`)
  - `OPENROUTER_API_KEY` / `VECTORVIEW_OPENROUTER_API_KEY` (for FrontPocket / 3072-dim embeddings)
  - `VECTORVIEW_OPENROUTER_MODEL` (default `google/gemini-embedding-2-preview`)
  - `VECTORVIEW_FRONTPOCKET_LIVE` (default `true`)
  - `VECTORVIEW_FRONTPOCKET_COLLECTIONS` (default `frontpocket_memory,minddrill_research_thoughts,minddrill_chat_memory,fp_reflections`)

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
   - `GET /api/highlight?collection=&since=` (poll for external highlight triggers)
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
- Keyword search uses Qdrant `scroll` + `filter.should` over multiple payload keys and case variants across meta_bridge and FrontPocket fields.
- Semantic search auto-detects embedding provider and model per collection (OpenRouter for FrontPocket 3072 dims, Ollama for meta_bridge 768 dims, OpenAI for 1536 dims).
- `/api/highlight` (`POST`) stores in-memory highlight events (`ids`, optional `focus_id`) per collection for UI polling.
- `/api/collections` includes projection status fields (`projection_ready`, `projection_note`) by probing sample vectors.
- Frontend clustering color key is derived from `payload.file_source`, `source_title`, `attachment_filename`, `sources`, or metadata keys (`extractClusterKey`).

## Gotchas / Non-obvious Behaviors
- `pca_gpu.py` is discovered by `pcaScript()` in executable dir (using `filepath.Dir(exe)`) or fallback path `pca_gpu.py` from cwd.
- `QDRANT_API_KEY` is passed via `cmd.Env` and forwarded as an `api-key` header in `pca_gpu.py`.
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
- `main_test.go` — unit tests for collection detection, embeddings, and projection
- `pca_gpu.py` — full-collection PCA worker
- `static/index.html` — all UI/rendering logic
- `.github/workflows/test.yml` — CI test workflow (`go test ./...`)
- `Makefile` — canonical developer shortcuts
- `.env.example` — runtime configuration template
