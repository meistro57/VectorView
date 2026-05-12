# VectorView — Roadmap

> *The latent space is infinite. The roadmap is not.*

This document tracks where VectorView is headed. Items are roughly ordered by priority within each phase, but this is a living document — things shift as the ecosystem evolves.

---

## ✅ v0.1 — Foundation *(shipped)*

- [x] Go binary with embedded frontend via `//go:embed`
- [x] Qdrant paginated scroll — pulls full collections with vectors
- [x] Unified projection pipeline — Python worker for full collections and filtered search results with PCA/random/t-SNE modes
- [x] Three.js r128 particle cloud with custom GLSL shaders
- [x] Additive blending + dual-layer bloom (core + glow)
- [x] Pulsing vertex animation driven by `uTime` uniform
- [x] OrbitControls — orbit, pan, zoom with damping
- [x] Color clustering by `entity_type` / `source_id` payload field
- [x] HUD: collection picker, stats bar, point inspector, search bar
- [x] Payload keyword search → filtered re-projection
- [x] `.env` support via godotenv
- [x] Deep space aesthetic — starfield, fog, obsidian palette

---

## 🔧 v0.2 — Stability & Polish

- [x] **`go.sum` generation** — ensure clean `go mod tidy` on fresh clone
- [x] **`.gitignore`** — exclude `.env`, binary, `__pycache__`
- [x] **`.env.example`** — committed template with all vars documented
- [x] **Error overlay in UI** — friendly message when Qdrant is unreachable
- [x] **Loading progress bar** — show % of points loaded during scroll
- [x] **Empty collection handling** — picker now shows a graceful "No collections available" state
- [x] **Responsive layout** — HUD panels collapse on narrow viewports
- [x] **Collection metadata sidebar** — vector size, distance metric, point count
- [x] **Projection readiness in collection picker** — show vector dims and disable unsupported collections
- [x] **Color grading controls** — hue/saturation/lightness live remap for cluster palette
- [x] **Point sample slider** — interactive point-count control wired to reload
- [x] **Keyboard shortcuts** — `R` reload, `Space` pause rotation, `Esc` clear inspector

---

## 🧠 v0.3 — Smarter Projection

- [x] **UMAP via subprocess or WASM** — optional upgrade from PCA for non-linear structure
- [x] **Incremental projection** — add new points to existing scene without full re-project
- [x] **Projection caching** — cache PCA result to disk, serve instantly on reload
- [x] **Axis labels** — show what PC1/PC2/PC3 correspond to (variance explained %)
- [x] **Projection selector** — UI toggle between PCA / random / t-SNE (server-side)
- [x] **Advanced named vector support** — explicit vector selection for collections with multiple named vectors (basic first-vector fallback now exists)

---

## 🔍 v0.4 — Vector Search

- [x] **Semantic search** — embed a query string (via Ollama or OpenRouter) and do true nearest-neighbor search against Qdrant
- [x] **Signal Scanner (top-K neighbors)** — click a point, run nearest-neighbor scan, highlight matches, and inspect ranked results
- [x] **Similarity radius** — click a point, highlight all neighbors within cosine distance threshold
- [x] **Distance matrix heatmap** — 2D mini-heatmap of inter-cluster distances in the HUD
- [x] **Outlier detection** — flag points with low neighbor density (rare finds) in a distinct color
- [x] **Cross-collection search** — query one collection, highlight matches in another

---

## 🎨 v0.5 — Visual Depth

- [x] **Particle trails** — ghost trail behind auto-rotating camera, stored in off-screen texture
- [x] **Cluster convex hulls** — translucent mesh wrapping each color cluster
- [x] **Density fog** — thicker fog in low-density regions, clearing near clusters
- [x] **Point size by payload field** — e.g. size by `score`, `confidence`, or chunk length
- [x] **Time axis** — animate points appearing in ingestion order (timeline scrubber)
- [x] **Screenshot export** — capture current view as PNG from HUD button
- [x] **Theme switcher** — deep space (default), bioluminescent, amber archaeology, terminal green

---

## ⚡ v0.6 — Live Streaming

- [ ] **SSE push endpoint** — `/api/stream` pushes new points as they're ingested
- [ ] **Live ingest mode** — frontend subscribes to SSE, animates new points dropping into the cloud
- [ ] **meta_bridge integration** — VectorView watches `mb_chunks` / `mb_claims` collections live during an ingest run
- [ ] **Redis Pub/Sub bridge** — optional Redis subscriber that forwards bot telemetry into the 3D scene
- [ ] **WebSocket ping** — heartbeat keeps connection alive across long ingest sessions

---

## 🤖 v0.7 — Agent Integration

- [ ] **Lewis (Discord) command hook** — `!vectorview snapshot` posts a PNG of current state to Discord channel
- [ ] **KAE integration** — KAE run seeds appear as animated "dig" events in the scene
- [ ] **Reflect loop overlay** — meta_bridge `reflect.py` claims appear as a separate layer, color-coded by confidence
- [ ] **Vectoreologist handoff** — click a cluster in VectorView, launch Vectoreologist TUI focused on that region
- [ ] **REST trigger API** — `POST /api/highlight` to programmatically highlight point IDs from external tools

---

## 🌐 v1.0 — Production Ready

- [ ] **Auth layer** — optional bearer token or basic auth for public-facing deployments
- [ ] **Docker image** — `docker run -e QDRANT_URL=... meistro57/vectorview`
- [ ] **Systemd unit file** — run as a background service on Pop!_OS
- [ ] **Multi-user sessions** — separate view state per browser session
- [ ] **Shareable permalinks** — encode collection + camera position + selected point in URL hash
- [ ] **Plugin system** — drop a `.so` or WASM module into `/plugins` to add custom projection algorithms

---

## 💡 Icebox (someday / maybe)

- WebGPU rendering backend — when Three.js support lands and browser adoption is there
- VR mode — WebXR so you can literally walk through latent space
- Collaborative mode — multiple cursors, shared selection, annotate points in real time
- Embedding generation in-browser — run a small ONNX model via Transformers.js for local semantic search
- Export to Blender — serialize particle positions + colors as a `.ply` point cloud
- iOS / Android app — native Three.js wrapper via Capacitor

---

## 📝 Notes

- Versioning follows the features above, not calendar dates
- PRs welcome — especially for projection algorithms and visual effects
- Open an Issue if your collection breaks something — include vector dimension and point count

---

*Last updated: May 2026*
