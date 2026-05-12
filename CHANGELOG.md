# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Added
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
- Timeline reveal controls (manual scrub + autoplay) based on ingestion/timestamp ordering.
- Screenshot export button for PNG snapshots.
- Full theme switcher (Deep Space, Bioluminescent, Amber Archaeology, Terminal Green).
- REST highlight trigger endpoint (`POST /api/highlight`) plus polling endpoint (`GET /api/highlight`) to drive UI highlighting from external tools.
- GitHub Actions workflow (`.github/workflows/test.yml`) to run `go test ./...` on push and pull request.

### Changed
- Random projection normalization adjusted to stabilize incremental append behavior.
- Similarity point-ID lookup now retries string/int64/uint64 forms to avoid Qdrant ID format mismatches on find-similar flows.
- Frontend now polls external highlight events and applies them to selection/highlight state in the Signal Scanner.
- README, ROADMAP, `.env.example`, and AGENTS documentation updated for projection, caching, semantic, visual depth, API highlight triggers, and test requirements.
