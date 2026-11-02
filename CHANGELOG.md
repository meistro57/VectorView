# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Added
- Adaptive colouring: **Color By** defaults to `auto`, sampling the loaded points and choosing the most informative payload key (for `discord_channel_logs` that is `channel`), with explicit `channel`, `channel_author`, and `author` options. Palette slots are assigned by cluster size so the largest groups get the most distinct hues, and clusters past the palette length shift hue and dim per lap instead of silently repeating a colour.
- `time` colour mode paints points as a chronological `ts`/`timestamp` gradient with a dated legend ramp, and the Signal Inspector surfaces a chat message's **Posted By** author and channel directly under the point ID.
- Chat-aware list rows: when a point has no `source_title`/`title`, the Outliers panel and Similar Signals now show `Author · Channel` instead of `Signal <uuid>`, and their meta lines fall back to `channel`/`author`.
- `GET /api/collections/{collection}/points/{point_id}` returns a point's full payload, so list responses can trim long text without losing data.
- Adaptive live-watch polling: idle collections back off from the base interval to a 30s floor, then return to base as soon as a count changes.
- Saved lenses: named presets capturing collection, projection, point limit, named vector, theme, color field, and camera, persisted in `localStorage`, with `#lens=` share links that restore the full view on load.
- Optional cluster legend composited into the exported screenshot (swatch, label, count, plus a `COLLECTION · PROJECTION · POINTS` caption), toggled from the top bar.
- Web fonts (Orbitron, Share Tech Mono) are vendored under `static/fonts`, removing the last external asset request.
- Three.js is now vendored under `static/vendor/three` (build + `ConvexGeometry`/`ConvexHull` addons) and upgraded from r128 to r160, so the embedded binary renders without any CDN access. Lighting intensities were rescaled for the modern light units and output color space is pinned to linear to preserve the existing look.
- Camera easing: search results, neighbor scans, and external highlights glide the view to the result centroid over ~650ms, cancel on any drag/scroll, and jump instead when `prefers-reduced-motion` is set.
- Selection history with `Alt+←` / `Alt+→` plus HUD prev/next buttons, and a bookmarks panel that pins inspected points to `localStorage` per collection.
- Container workflow: multi-stage `Dockerfile` (static Go binary plus a numpy-enabled Python runtime, non-root, `/healthz` healthcheck), `compose.yaml` bundling Qdrant with a persistent projection-cache volume, `.dockerignore`, and `make docker-build` / `docker-up` / `docker-down` targets.
- CI now also enforces `gofmt`, `go vet`, `staticcheck`, `go test -race ./...`, and the projection-worker unittest suite.
- Handler layer extracted from `main()` onto a `server` type with per-route methods, plus `httptest` coverage (collections, points, search, semantic search, highlight, similarity, health, inline fallback) backed by a fake Qdrant.
- Inline Go PCA fallback: when `pca_gpu.py` is unavailable or fails and the requested point count is within `VECTORVIEW_INLINE_PCA_MAX`, `/api/points` projects in-process and reports `projection.method = "pca-go-fallback"` instead of failing.
- Startup configuration report covering Qdrant reachability and latency, cache backends, projection slots, worker interpreter and available Python modules, and embedding provider keys. `VECTORVIEW_STRICT_CONFIG=1` turns warnings into a startup failure.
- `VECTORVIEW_PYTHON` to pin the projection-worker interpreter.
- Projection-worker test suite (`tests/test_pca_gpu.py`) with deterministic PCA/random projection checks, named-vector selection, axis padding, CLI round-trip, and a rotation-invariant golden geometry digest. `VECTORVIEW_PCA_BACKEND` pins the worker backend so tests never depend on GPU availability.
- Load progress is now pushed over SSE as `load_progress` events; the UI's 250ms progress poller and 1500ms highlight poller are gone and `/api/load-progress` was removed.
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
- The left Controls panel is wider (300px) and scrolls on desktop, so long sections (Color By, size field, lenses, distance matrix) stay reachable on short viewports; it uses a thin themed scrollbar and keeps child sections from squashing.
- Performance pass: `/api/collections` now parses named-vector configs (dims were reported as 0 for 9 of 30 collections), answers from a 30s catalog cache, and probes collections in parallel instead of one at a time (1.55s → ~60ms cold, ~1ms cached against a 30-collection Qdrant).
- API and static responses are gzip-compressed when accepted (a 2,000-point load drops from 3.2 MB to ~400 KB on the wire), with immutable cache headers for `/vendor` and `/fonts`.
- List responses trim payload strings to `VECTORVIEW_PAYLOAD_MAX_CHARS` (default 400; the inspector only rendered 120 anyway), cutting the same 2,000-point response from 3.2 MB to 1.55 MB decoded.
- The projection worker keeps `auto` on the numpy randomized SVD for small/medium matrices and only reaches for torch/cupy above `VECTORVIEW_PCA_GPU_THRESHOLD` rows×dims (a 2,000×384 PCA went from 0.9s to 0.2s).
- Frontend frame budget: hover raycasts only run after pointer movement (or at 10 Hz while auto-rotating), color-slider recolors coalesce into one per frame, the trail pipeline is skipped when persistence is off, and the nebula layer hides itself while the frame rate is below ~28 FPS.
- The inspector renders long text (up to 1,200 chars, scrollable) and hydrates the full payload for trimmed points on selection.
- Replaced the local `max`/`minInt` helpers with Go's built-in `min`/`max`.
- Frontend split into `static/css/app.css`, `static/js/app.js`, `static/js/shaders.js`, and `static/js/api.js`; `index.html` is now markup plus the import map (4103 → 271 lines) with no bundler or npm dependency.
- Restored the load-progress status line, which had been dropped when the progress poller was replaced by SSE events.
- Random projection normalization adjusted to stabilize incremental append behavior.
- `GET /api/points` no longer re-runs the projection worker for concurrent identical requests, and its result is now written to the disk cache even when Redis is not configured.
- API failures now return a stable error envelope instead of plain-text messages, and the UI surfaces `error.message` in the load/scan status line.
- Similarity point-ID lookup now retries string/int64/uint64 forms to avoid Qdrant ID format mismatches on find-similar flows.
- Frontend now polls external highlight events and applies them to selection/highlight state in the Signal Scanner.
- Live ingest now appends new points in-place for random projection mode (with full-sync fallback for non-random projections).
- Size-field payload lookup now resolves nested payload keys (for example nested `score` or `confidence` metadata).
- Point Count slider upper bound increased from 20,000 to 75,000.
- External highlight updates now arrive over the existing SSE stream, with a single catch-up request when the stream (re)connects.
- README, ROADMAP, `.env.example`, and AGENTS documentation updated for projection, caching, semantic, visual depth, API highlight triggers, projection limits, token auth, the health probe, startup config validation, the inline PCA fallback, and test requirements.
