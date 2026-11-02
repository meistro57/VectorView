# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Added
- On-disk projection cache (`.cache/projections` by default) keyed by collection, limit, method, and vector name, with TTL expiry, atomic writes, and a size-capped LRU prune (`VECTORVIEW_CACHE_DIR`, `VECTORVIEW_CACHE_MAX_MB`). Redis, when configured, now layers on top of the disk cache instead of replacing it.
- Projection concurrency control: a semaphore caps simultaneous `pca_gpu.py` workers (`VECTORVIEW_MAX_PROJECTIONS`, default NumCPU/2) and waiting requests return HTTP 429 after `VECTORVIEW_PROJECTION_QUEUE_SECONDS`. Identical in-flight projections are coalesced so duplicate requests share one worker run.
- Optional bearer-token auth for `/api/*` via `VECTORVIEW_TOKEN`, accepting `Authorization: Bearer`, `X-VectorView-Token`, or `?token=`. The frontend forwards the token automatically once supplied.
- `GET /healthz` reporting Qdrant reachability and latency, active cache backends, projection slot usage, and uptime.
- Structured JSON error envelope (`{"error":{"code","message"}}`) across every API handler, with details logged server-side instead of returned to clients.
- HTTP server timeouts (read-header, read, idle) and graceful shutdown on SIGINT/SIGTERM: in-flight SSE streams and projections drain with a 15s grace period before the process exits.
- Outliers panel: lists every point flagged as a spatial-density outlier (title, snippet, ai_provider/speaker), click-to-inspect-and-focus, plus an "Isolate Outliers" toggle that hides all non-outlier points in the cloud.
- "Color By" payload-field coloring: points can now be categorically colored by `ai_provider`, `ai_model`, `speaker`, `awakening_phase`, `memory_kind`, `project`, `source_type`, or any custom payload key, in addition to the original source-derived clustering (`extractClusterKey` generalized, `recolorByField()` recolors + rebuilds hulls/legend in-place without a refetch).
- UMAP projection mode in both backend and frontend projection selectors (`pca` / `random` / `tsne` / `umap`).
- Named vector selection support end-to-end (`vector_name`) across `/api/points`, `/api/search`, semantic search, and similarity scans.
- Redis-backed projection response caching for `/api/points` via `VECTORVIEW_REDIS_URL` + `VECTORVIEW_CACHE_TTL_SECONDS`.
- Incremental random projection append path (`append_from`) to extend existing clouds without full re-projection when cached base slices exist.
- Semantic search endpoint (`/api/semantic-search`) using query embedding via Ollama (`VECTORVIEW_OLLAMA_URL`, `VECTORVIEW_EMBED_MODEL`).
- Cross-collection semantic search via `target_collection` parameter in `/api/semantic-search` and UI selector.
- Particle trails using off-screen ping-pong render targets and configurable persistence.
- Cluster convex hull overlays for dominant visible clusters (translucent fill + wireframe).
- Density-aware fog adjustment per loaded cloud.
- Payload-field point sizing modes (`score`, `confidence`, `chunk_length`, `text_length`).
- Top HUD load-state indicator (`FETCH`, `PROJECTING`, `RENDERING`, `READY`, `ERROR`) during `/api/points` loads.
- Timeline reveal controls (manual scrub + autoplay) based on ingestion/timestamp ordering.
- Screenshot export button for PNG snapshots.
- Full theme switcher (Deep Space, Bioluminescent, Amber Archaeology, Terminal Green).
- REST highlight trigger endpoint (`POST /api/highlight`) plus polling endpoint (`GET /api/highlight`) to drive UI highlighting from external tools.
- GitHub Actions workflow (`.github/workflows/test.yml`) to run `go test ./...` on push and pull request.

### Changed
- Random projection normalization adjusted to stabilize incremental append behavior.
- `GET /api/points` no longer re-runs the projection worker for concurrent identical requests, and its result is now written to the disk cache even when Redis is not configured.
- API failures now return a stable error envelope instead of plain-text messages, and the UI surfaces `error.message` in the load/scan status line.
- Similarity point-ID lookup now retries string/int64/uint64 forms to avoid Qdrant ID format mismatches on find-similar flows.
- Frontend now polls external highlight events and applies them to selection/highlight state in the Signal Scanner.
- Live ingest now appends new points in-place for random projection mode (with full-sync fallback for non-random projections).
- Size-field payload lookup now resolves nested payload keys (for example nested `score` or `confidence` metadata).
- Point Count slider upper bound increased from 20,000 to 75,000.
- README, ROADMAP, `.env.example`, and AGENTS documentation updated for projection, caching, semantic, visual depth, API highlight triggers, projection limits, token auth, the health probe, and test requirements.
